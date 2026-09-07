"""Adaptive, hover-only wrist-camera alignment loop.

Unlike safe_wrist_align.py (one fixed-Jacobian step), this repeats small
1-3 mm corrections and updates the pixel/meter Jacobian after every step
from the actually observed pixel displacement (Broyden's "good" rank-1
update). Stops on convergence, on iteration limit, or -- exactly like the
single-shot version -- disables torque immediately if the alignment error
grows past the best one seen so far.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

SAFE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SAFE_DIR.parent / "task_red_cube_to_bin_new_gripper"))

from kinematics import build_kinematics, gripper_position, solve_ik  # noqa: E402
from safe_wrist_probe import ARM, PORT, check, emergency_release, frame_and_pixel, move, positions  # noqa: E402

from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig  # noqa: E402
from lerobot.robots.so_follower.so_follower import SOFollower  # noqa: E402

# Same starting estimate the single-shot align.py used; updated in place each
# iteration below using the real measured pixel/meter response.
INITIAL_JACOBIAN_PX_PER_M = np.array([[-495.01798207828074, 3056.9781342166493], [-7614.269750109088, -3461.308750232348]])
MAX_ALIGN_STEP_M = 0.003
MIN_HOVER_Z_M = 0.100
CONVERGED_PX = 15.0
WORSENED_PX = 10.0
DEFAULT_MAX_ITERS = 8
REPORT = SAFE_DIR / "safe_wrist_align_adaptive_report.json"


def broyden_update(j: np.ndarray, step_m: np.ndarray, observed_dpx: np.ndarray) -> np.ndarray:
    """Rank-1 update of J from one real (step -> pixel delta) observation."""
    denom = float(step_m @ step_m)
    if denom < 1e-12:
        return j
    predicted = j @ step_m
    return j + np.outer(observed_dpx - predicted, step_m) / denom


def solve_step(j: np.ndarray, error_px: np.ndarray) -> np.ndarray:
    """Least-squares meter step that would null error_px under J, then clip to the step cap."""
    raw, *_ = np.linalg.lstsq(j, error_px, rcond=None)
    norm = float(np.linalg.norm(raw))
    return raw if norm <= MAX_ALIGN_STEP_M else raw * (MAX_ALIGN_STEP_M / norm)


def main() -> int:
    """Preview or execute the adaptive multi-step hover alignment."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--workspace-clear", action="store_true")
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERS)
    args = parser.parse_args()
    report: dict = {"mode": "PREVIEW_ONLY", "iterations": []}
    bus, cap = None, None
    try:
        jaw = np.array(json.loads((SAFE_DIR / "wrist_jaw_center.json").read_text())["jaw_center_px"], dtype=float)
        robot = SOFollower(SOFollowerRobotConfig(port=PORT, id="follower", use_degrees=True))
        bus = robot.bus
        bus.connect(handshake=True)
        kin = build_kinematics()
        start = positions(bus)
        xyz = gripper_position(kin, np.array([start[name] for name in ARM]))
        if xyz[2] < MIN_HOVER_Z_M:
            raise RuntimeError(f"refuse: current model z={xyz[2]:.3f}m below hover gate")
        cap = cv2.VideoCapture(4, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError("cannot open wrist camera /dev/video4")

        j = INITIAL_JACOBIAN_PX_PER_M.copy()
        pixel = np.array(frame_and_pixel(cap))
        error = jaw - pixel
        best_error_norm = float(np.linalg.norm(error))
        report["jaw_center_px"] = jaw.tolist()
        report["start_px"] = pixel.tolist()
        report["start_error_px"] = error.tolist()
        report["hover_xyz_m"] = xyz.tolist()

        if not args.execute:
            report["result"] = "PREVIEW_ONLY_NO_MOTOR_WRITES"
            report["would_run_up_to_iterations"] = args.max_iterations
            return 0
        if not args.workspace_clear:
            raise RuntimeError("--workspace-clear is required")
        check(bus)

        for i in range(1, args.max_iterations + 1):
            if best_error_norm <= CONVERGED_PX:
                report["result"] = "CONVERGED"
                break
            step = solve_step(j, error)
            current = positions(bus)
            target_arm = solve_ik(kin, np.array([current[name] for name in ARM]), (xyz[0] + step[0], xyz[1] + step[1], xyz[2]))
            target_arm[4] = current["wrist_roll"]
            target = {**current, **dict(zip(ARM, map(float, target_arm), strict=True))}
            move(bus, current, target)
            new_pixel = np.array(frame_and_pixel(cap))
            observed_dpx = new_pixel - pixel
            j = broyden_update(j, step, observed_dpx)
            new_error = jaw - new_pixel
            new_error_norm = float(np.linalg.norm(new_error))
            report["iterations"].append({
                "index": i, "step_m": step.tolist(), "pixel_before": pixel.tolist(), "pixel_after": new_pixel.tolist(),
                "error_after_px": new_error.tolist(), "error_norm_after_px": new_error_norm, "jacobian_px_per_m": j.tolist(),
            })
            if new_error_norm > best_error_norm + WORSENED_PX:
                raise RuntimeError(f"alignment error worsened ({new_error_norm:.1f}px vs best {best_error_norm:.1f}px); stopping")
            pixel, error = new_pixel, new_error
            best_error_norm = min(best_error_norm, new_error_norm)
        else:
            report["result"] = "MAX_ITERATIONS_REACHED"
        report.setdefault("result", "CONVERGED")
        report["final_error_norm_px"] = best_error_norm
        report["final_jacobian_px_per_m"] = j.tolist()
        return 0
    except Exception as exc:
        report["result"] = "EMERGENCY_TORQUE_DISABLED" if args.execute else "PREVIEW_BLOCKED_NO_MOTOR_WRITES"
        report["error"] = str(exc)
        if args.execute and bus is not None and bus.is_connected:
            emergency_release(bus)
        return 2
    finally:
        REPORT.write_text(json.dumps(report, indent=2) + "\n")
        if cap is not None:
            cap.release()
        if bus is not None and bus.is_connected:
            bus.disconnect(disable_torque=False)
        print(f"[saved] {REPORT}; result={report.get('result')}")


if __name__ == "__main__":
    raise SystemExit(main())
