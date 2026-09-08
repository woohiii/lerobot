"""Supervised one-item trash-to-bin runner: IR-guided IK + wrist-image IL.

Camera owners remain separate processes. Start ``astra_s_ir_hub.py`` and the
wrist-camera publisher first; this runner only reads their atomically
published frames. A live robot run additionally requires ``--confirm-live-run``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
VISION_DIR = ROOT / "custom_scripts" / "vision_pick_place"
TASK_DIR = VISION_DIR / "task_trash_to_bin"
sys.path.insert(0, str(TASK_DIR))

import config  # noqa: E402
import gripper  # noqa: E402
import il_grasp_skill  # noqa: E402
import perception  # noqa: E402
from kinematics import CollisionDetected, SOArm101  # noqa: E402
from lerobot.configs import FeatureType, PreTrainedConfig  # noqa: E402


HOVER_Z_M = config.TABLE_Z + 0.12
DROP_Z_M = config.TABLE_Z + 0.035
WINDOW = "IR target: click trash, then bin | e=emergency stop | q=cancel"


class SafetyBlocked(RuntimeError):
    """A precondition failed before a new robot motion may be sent."""


class EmergencyStop(RuntimeError):
    """The operator requested torque release; never perform home return."""


def require_fresh_file(path: Path, max_age_s: float) -> None:
    if not path.is_file():
        raise SafetyBlocked(f"missing published frame: {path}")
    age_s = time.time() - path.stat().st_mtime
    if age_s >= max_age_s:
        raise SafetyBlocked(f"stale published frame ({age_s:.1f}s): {path}")


def load_ir_homography(path: Path) -> np.ndarray:
    try:
        data = json.loads(path.read_text())
        matrix = np.asarray(data.get("homography") if isinstance(data, dict) else data, dtype=float)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SafetyBlocked(f"IR homography unreadable: {path} ({exc})") from exc
    if matrix.shape != (3, 3):
        raise SafetyBlocked(f"IR homography must be 3x3, got {matrix.shape}")
    if not np.isfinite(matrix).all() or abs(float(np.linalg.det(matrix))) < 1e-12:
        raise SafetyBlocked("IR homography is non-finite or singular")
    return matrix


def ir_pixel_to_xy(pixel: tuple[int, int], homography: np.ndarray) -> tuple[float, float]:
    mapped = homography @ np.asarray([pixel[0], pixel[1], 1.0], dtype=float)
    if abs(mapped[2]) < 1e-9:
        raise SafetyBlocked("IR click maps to infinity")
    return float(mapped[0] / mapped[2]), float(mapped[1] / mapped[2])


def validate_policy_contract(policy_path: Path) -> None:
    if not policy_path.is_dir():
        raise SafetyBlocked(f"policy checkpoint missing: {policy_path}")
    policy_cfg = PreTrainedConfig.from_pretrained(str(policy_path))
    state = policy_cfg.input_features.get("observation.state")
    wrist = policy_cfg.input_features.get("observation.images.wrist")
    action = policy_cfg.output_features.get("action")
    if state is None or state.type is not FeatureType.STATE or tuple(state.shape) != (6,):
        raise SafetyBlocked("policy must require six-element observation.state")
    if wrist is None or wrist.type is not FeatureType.VISUAL or tuple(wrist.shape) != (3, 480, 640):
        raise SafetyBlocked("policy must require observation.images.wrist with shape (3, 480, 640)")
    if action is None or action.type is not FeatureType.ACTION or tuple(action.shape) != (6,):
        raise SafetyBlocked("policy must output six-element action")


def fresh_frame(path: Path) -> np.ndarray:
    require_fresh_file(path, config.FRAME_STALE_TIMEOUT_S)
    frame = cv2.imread(str(path))
    if frame is None:
        raise SafetyBlocked(f"published frame is not decodable: {path}")
    return frame


def collect_clicks(ir_path: Path, wrist_path: Path) -> tuple[tuple[int, int], tuple[int, int]]:
    clicks: list[tuple[int, int]] = []
    stopped = False

    def on_mouse(event, x, y, _flags, _userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x, y))

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, on_mouse)
    try:
        while len(clicks) < 2:
            ir = fresh_frame(ir_path)
            wrist = fresh_frame(wrist_path)
            wrist = cv2.resize(wrist, (ir.shape[1], ir.shape[0]))
            shown = cv2.hconcat([cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR) if ir.ndim == 2 else ir, wrist])
            prompt = "1/2 click trash in LEFT IR panel" if not clicks else "2/2 click bin in LEFT IR panel"
            cv2.putText(shown, prompt, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow(WINDOW, shown)
            key = cv2.waitKey(20) & 0xFF
            if key == ord("e"):
                stopped = True
                break
            if key in (ord("q"), 27):
                break
            if clicks and clicks[-1][0] >= ir.shape[1]:
                clicks.pop()
                print("[blocked] target clicks must be in the left IR panel")
        if stopped:
            raise EmergencyStop("operator pressed e before motion")
        if len(clicks) != 2:
            raise SafetyBlocked("operator cancelled target selection")
        return clicks[0], clicks[1]
    finally:
        cv2.destroyWindow(WINDOW)


def safe_xy(pixel: tuple[int, int], homography: np.ndarray, label: str) -> tuple[float, float]:
    xy = ir_pixel_to_xy(pixel, homography)
    if not perception.is_xy_within_safe_workspace(*xy):
        raise SafetyBlocked(f"{label} click maps outside calibrated workspace: {xy}")
    return xy


def move_hover(arm: SOArm101, xy: tuple[float, float], label: str) -> None:
    print(f"[IK] {label} hover -> ({xy[0]:.3f}, {xy[1]:.3f}, {HOVER_Z_M:.3f})")
    reached = arm.move_to_xyz_converge((xy[0], xy[1], HOVER_Z_M), tolerance_m=0.015, max_iters=20)
    if float(np.linalg.norm(reached - np.asarray((xy[0], xy[1], HOVER_Z_M)))) > 0.02:
        raise SafetyBlocked(f"{label} IK residual exceeds 20 mm")


def run_live(args: argparse.Namespace, homography: np.ndarray) -> int:
    trash_px, bin_px = collect_clicks(args.ir_frame, args.wrist_frame)
    trash_xy = safe_xy(trash_px, homography, "trash")
    bin_xy = safe_xy(bin_px, homography, "bin")
    arm = SOArm101(port=args.arm_port)
    emergency_stopped = False
    home_xyz: tuple[float, float, float] | None = None
    try:
        arm.connect()
        home_xyz = tuple(arm.gripper_xyz())
        gripper.open_gripper(arm)
        move_hover(arm, trash_xy, "trash")
        wrist = perception.PublishedFrameSource(str(args.wrist_frame))
        if not il_grasp_skill.run_grasp_skill(arm, wrist, str(args.policy_path), max_seconds=args.il_seconds):
            raise SafetyBlocked("IL grasp verification failed; bin motion blocked")
        arm.move_z(config.LIFT_M, steps=20, step_delay_s=0.05)
        move_hover(arm, bin_xy, "bin")
        descent = DROP_Z_M - float(arm.gripper_xyz()[2])
        if descent >= 0:
            raise SafetyBlocked("refusing bin descent: hover is not above conservative drop height")
        arm.move_z(descent, steps=15, step_delay_s=0.05)
        gripper.open_gripper(arm)
        arm.move_z(-descent, steps=15, step_delay_s=0.05)
        print("[success] one item released into bin")
        return 0
    except (EmergencyStop, KeyboardInterrupt) as exc:
        emergency_stopped = True
        raise EmergencyStop("operator interrupted live motion") from exc
    finally:
        try:
            if emergency_stopped:
                arm.release_torque()
                print("[emergency] torque released; home return suppressed")
            elif home_xyz is not None:
                try:
                    arm.move_to_xyz_converge(home_xyz, tolerance_m=0.015, max_iters=20)
                except Exception as exc:  # best-effort only; never hide task result
                    print(f"[warning] home return blocked: {exc}")
        finally:
            arm.disconnect()


def build_parser() -> argparse.ArgumentParser:
    """Keep the live safety contract inspectable without opening hardware."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-port", default=config.FOLLOWER_PORT)
    parser.add_argument("--policy-path", type=Path, required=True, help="trained IL grasp checkpoint (no default: no trash_to_bin checkpoint exists yet)")
    parser.add_argument("--ir-homography", type=Path, required=True, help="IR-pixel to robot-XY calibration JSON")
    parser.add_argument("--ir-frame", type=Path, default=Path(config.ASTRA_IR_FRAME_PATH))
    parser.add_argument("--wrist-frame", type=Path, default=Path(config.WRIST_FRAME_PATH))
    parser.add_argument("--il-seconds", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true", help="validate assets and policy I/O without robot motion")
    parser.add_argument("--confirm-live-run", action="store_true", help="required before any serial connection or robot command")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.il_seconds <= 0:
            raise SafetyBlocked("--il-seconds must be positive")
        homography = load_ir_homography(args.ir_homography)
        fresh_frame(args.ir_frame)
        fresh_frame(args.wrist_frame)
        validate_policy_contract(args.policy_path)
        print("[preflight] IR/wrist frames, IR homography, and policy I/O contract verified")
        if args.dry_run:
            return 0
        if not args.confirm_live_run:
            raise SafetyBlocked("live actuation requires --confirm-live-run (use --dry-run first)")
        return run_live(args, homography)
    except EmergencyStop as exc:
        print(f"[emergency] {exc}")
        return 3
    except (SafetyBlocked, CollisionDetected, RuntimeError) as exc:
        print(f"[blocked] {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
