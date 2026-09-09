"""One-step RGB/depth XY correction for a provisional hand-eye transform."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def compute_xy_nudge(
    current_camera_xyz: np.ndarray,
    target_camera_xyz: np.ndarray,
    camera_to_base: np.ndarray,
    *,
    max_nudge_m: float,
) -> tuple[float, float]:
    current = np.asarray(current_camera_xyz, dtype=float)
    target = np.asarray(target_camera_xyz, dtype=float)
    transform = np.asarray(camera_to_base, dtype=float)
    if current.shape != (3,) or target.shape != (3,) or transform.shape != (4, 4):
        raise ValueError("camera points and transform have invalid shapes")
    if not np.isfinite(current).all() or not np.isfinite(target).all() or not np.isfinite(transform).all():
        raise ValueError("visual-servo inputs must be finite")
    current_base = (transform @ np.r_[current, 1.0])[:3]
    target_base = (transform @ np.r_[target, 1.0])[:3]
    delta = target_base[:2] - current_base[:2]
    magnitude = float(np.linalg.norm(delta))
    if magnitude > max_nudge_m:
        raise ValueError(f"visual-servo correction {magnitude:.4f} m exceeds {max_nudge_m:.4f} m")
    return float(delta[0]), float(delta[1])


def validate_args(*, multi_step: bool, step_nudge_m: float, max_nudge_m: float) -> None:
    """Validate step sizing only for the multi-step execution path."""
    if multi_step and (step_nudge_m <= 0 or step_nudge_m > max_nudge_m):
        raise ValueError("--step-nudge-m must be > 0 and <= --max-nudge-m")


def _select_valid_depth_pixel(
    depth: np.ndarray, pixel: tuple[int, int], region: tuple[int, int, int, int] | None = None
) -> tuple[int, int] | None:
    """Select the valid depth pixel nearest the target, optionally inside a box."""
    u, v = pixel
    if region is None:
        x0, y0, x1, y1 = 0, 0, depth.shape[1], depth.shape[0]
    else:
        rx, ry, rw, rh = region
        x0, y0 = max(0, rx), max(0, ry)
        x1, y1 = min(depth.shape[1], rx + rw), min(depth.shape[0], ry + rh)
    if x1 <= x0 or y1 <= y0:
        return None
    rows, cols = np.where(depth[y0:y1, x0:x1] > 0)
    if len(cols) == 0:
        return None
    cols, rows = cols + x0, rows + y0
    index = int(np.argmin((cols - u) ** 2 + (rows - v) ** 2))
    return int(cols[index]), int(rows[index])


def _capture_point(
    camera, depth_stream, color, pixel: tuple[int, int], region: tuple[int, int, int, int] | None = None
) -> np.ndarray:
    depth = camera.read_raw_depth_mm()
    if depth is None:
        raise ValueError("depth frame unavailable")
    u = int(pixel[0] * depth.shape[1] / color.shape[1])
    v = int(pixel[1] * depth.shape[0] / color.shape[0])
    if not (0 <= u < depth.shape[1] and 0 <= v < depth.shape[0]):
        raise ValueError("selected pixel is outside depth frame")
    depth_region = None
    if region is not None:
        rx, ry, rw, rh = region
        depth_region = (
            int(rx * depth.shape[1] / color.shape[1]),
            int(ry * depth.shape[0] / color.shape[0]),
            int(rw * depth.shape[1] / color.shape[1]),
            int(rh * depth.shape[0] / color.shape[0]),
        )
    resolved = _select_valid_depth_pixel(depth, (u, v), depth_region)
    if resolved is None:
        raise ValueError("selected detection box has no valid depth")
    u, v = resolved
    from primesense import openni2

    xyz_mm = openni2.convert_depth_to_world(depth_stream, u, v, float(depth[v, u]))
    return np.asarray(xyz_mm, dtype=float) / 1000.0


def _load_local_qwen_detector():
    """Load the adjacent local Qwen detector without requiring repo-root imports."""
    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root))
    from custom_scripts.vision_pick_place.task_red_cube_to_bin_new_gripper.perception_qwen import detect_qwen

    return detect_qwen


def _build_execution_arm(port: str):
    """Create the calibration-ID-isolated left follower for physical motion."""
    from .left_arm import LeftSOArm101

    return LeftSOArm101(port=port)


def hold_tcp_pose(port: str, *, arm_factory=_build_execution_arm) -> np.ndarray:
    """Enable torque at the arm's measured pose after an operator selects the TCP."""
    arm = arm_factory(port)
    connected = False
    try:
        arm.connect()
        connected = True
        arm.enable_torque()
        return arm.get_joint_deg()
    finally:
        if connected:
            arm.disconnect()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--port", required=True)
    parser.add_argument("--max-rms-m", type=float, default=0.03)
    parser.add_argument("--max-nudge-m", type=float, default=0.04)
    parser.add_argument(
        "--multi-step",
        action="store_true",
        help="apply bounded XY corrections repeatedly, re-reading FK TCP after each step",
    )
    parser.add_argument("--step-nudge-m", type=float, default=0.04)
    parser.add_argument(
        "--max-steps-per-run",
        type=int,
        default=1,
        help="maximum multi-step corrections to apply before requiring a new camera measurement",
    )
    parser.add_argument(
        "--max-z-drift-m",
        type=float,
        default=0.02,
        help="abort if measured FK TCP Z drifts more than this during XY nudges",
    )
    parser.add_argument(
        "--max-step-joint-deg",
        type=float,
        default=15.0,
        help="refuse a substep if IK requests more than this per-joint change",
    )
    parser.add_argument("--require-home", action="store_true", help="require safe home pose before motion")
    parser.add_argument("--home-tolerance-deg", type=float, default=5.0)
    parser.add_argument("--local-prompt", help="detect the box automatically with local Qwen2.5-VL")
    parser.add_argument(
        "--auto-tcp",
        action="store_true",
        help="read the robot joints and estimate TCP via FK; no RGB click is needed",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--hold-position", action="store_true")
    parser.add_argument(
        "--preview",
        action="store_true",
        help="show the Qwen box marker and wait for Space before continuing",
    )
    args = parser.parse_args(argv)
    if args.hold_position and not args.execute:
        parser.error("--hold-position requires --execute")
    try:
        validate_args(
            multi_step=args.multi_step, step_nudge_m=args.step_nudge_m, max_nudge_m=args.max_nudge_m
        )
    except ValueError as exc:
        parser.error(str(exc))
    if not 1 <= args.max_steps_per_run <= 20:
        parser.error("--max-steps-per-run must be between 1 and 20")
    if args.max_z_drift_m <= 0:
        parser.error("--max-z-drift-m must be positive")
    if args.max_step_joint_deg <= 0:
        parser.error("--max-step-joint-deg must be positive")
    if args.home_tolerance_deg <= 0:
        parser.error("--home-tolerance-deg must be positive")
    payload = json.loads(args.calibration.read_text(encoding="utf-8"))
    rms = float(payload["rms_error_m"])
    if rms > args.max_rms_m:
        parser.error(f"RMS {rms:.6f} m exceeds visual-servo limit {args.max_rms_m:.6f} m")

    vision_dir = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(vision_dir / "task_trash_to_bin"))
    sys.path.insert(0, str(vision_dir))
    from orbbec_color_camera import ThreadedOrbbecRGBDCamera

    camera = ThreadedOrbbecRGBDCamera(width=640, height=480, fps=30)
    selected: list[tuple[int, int] | None] = [None]
    labels: dict[str, np.ndarray] = {}
    label_pixels: dict[str, tuple[int, int]] = {}
    window_created = False
    try:
        if not camera.isOpened():
            raise RuntimeError("Astra S did not open")

        def on_click(event, x, y, _flags, _userdata):
            if event == cv2.EVENT_LBUTTONDOWN:
                selected[0] = (x, y)

        cv2.namedWindow("visual servo RGB")
        cv2.setMouseCallback("visual servo RGB", on_click)
        window_created = True
        if args.auto_tcp and not args.local_prompt:
            parser.error("--auto-tcp currently requires --local-prompt")
        print("TCP 자동 계산 모드" if args.auto_tcp else "그리퍼 TCP 클릭 후 G, 계산/종료는 Q")
        if args.local_prompt:
            detect_qwen = _load_local_qwen_detector()

            detected = None
            while detected is None:
                ret, color, _ = camera.read()
                if not ret or color is None:
                    time.sleep(0.05)
                    continue
                detected = detect_qwen(color, args.local_prompt)
            x, y, width, height = detected.bbox
            box_pixel = (int(round(x + width / 2)), int(round(y + height / 2)))
            labels["box"] = _capture_point(camera, camera.depth_stream, color, box_pixel, (int(x), int(y), int(width), int(height)))
            label_pixels["box"] = box_pixel
            print(f"local Qwen box camera point: {tuple(labels['box'])}")
            if args.preview:
                preview = color.copy()
                cv2.rectangle(
                    preview,
                    (int(x), int(y)),
                    (int(x + width), int(y + height)),
                    (0, 255, 0),
                    2,
                )
                cv2.circle(preview, box_pixel, 8, (0, 255, 0), 2)
                cv2.putText(preview, "Qwen box - Space=continue, Q=quit", (10, 25), 0, 0.65, (0, 255, 255), 2)
                cv2.imshow("visual servo RGB", preview)
                while True:
                    key = cv2.waitKey(30) & 0xFF
                    if key == ord("q"):
                        return 2
                    if key == ord(" "):
                        break
            if args.auto_tcp:
                from .left_arm import LeftSOArm101

                auto_arm = LeftSOArm101(port=args.port)
                auto_arm.connect()
                try:
                    joints = auto_arm.get_joint_deg()
                    if args.require_home:
                        from .home_pose import home_pose_error_deg

                        error_deg = home_pose_error_deg(joints)
                        if error_deg > args.home_tolerance_deg:
                            raise RuntimeError(
                                f"home pose required: largest joint error {error_deg:.2f} deg exceeds "
                                f"{args.home_tolerance_deg:.2f} deg"
                            )
                        print(f"home pose check: passed (largest error {error_deg:.2f} deg)")
                    flange_pose = auto_arm.kin.forward_kinematics(joints[:5])
                    offset = np.asarray(payload.get("tcp_offset_m", [0.0, 0.0, 0.0]), dtype=float)
                    tcp_base = flange_pose[:3, 3] + flange_pose[:3, :3] @ offset
                    camera_to_base = np.asarray(payload["camera_to_base"], dtype=float)
                    tcp_camera = np.linalg.inv(camera_to_base) @ np.r_[tcp_base, 1.0]
                    labels["tcp"] = tcp_camera[:3]
                    print(f"auto TCP camera point: {tuple(labels['tcp'])}")
                finally:
                    auto_arm.disconnect()
                delta = compute_xy_nudge(
                    labels["tcp"],
                    labels["box"],
                    np.asarray(payload["camera_to_base"]),
                    max_nudge_m=(1.0 if args.multi_step else args.max_nudge_m),
                )
                print(f"suggested base XY nudge: dx={delta[0]:.4f}, dy={delta[1]:.4f} m")
                if not args.execute:
                    if args.multi_step:
                        import math

                        steps = math.ceil(np.linalg.norm(delta) / args.step_nudge_m)
                        print(f"multi-step dry plan: {steps} step(s)")
                        print(f"per-run execution limit: {args.max_steps_per_run} step(s)")
                    print("dry visual-servo plan: no hardware command was issued")
                    return 0
                if input("Type MOVE_NUDGE to apply one bounded XY correction: ").strip() != "MOVE_NUDGE":
                    print("aborted: confirmation did not match")
                    return 2
                from .left_arm import LeftSOArm101

                arm = LeftSOArm101(port=args.port)
                connected = False
                holding = False
                try:
                    arm.connect()
                    connected = True
                    if args.require_home:
                        from .home_pose import home_pose_error_deg

                        error_deg = home_pose_error_deg(arm.get_joint_deg())
                        if error_deg > args.home_tolerance_deg:
                            raise RuntimeError(
                                f"home pose required: largest joint error {error_deg:.2f} deg exceeds "
                                f"{args.home_tolerance_deg:.2f} deg"
                            )
                    initial_gripper_z = float(arm.gripper_xyz()[2])

                    def safe_nudge(dx: float, dy: float) -> np.ndarray:
                        """Apply a small XY move while checking Z after every substep."""
                        reached = arm.gripper_xyz()
                        substeps = 12
                        for _ in range(substeps):
                            current_xyz = arm.gripper_xyz()
                            sub_dx, sub_dy = dx / substeps, dy / substeps
                            plan = arm.preview_move(
                                (current_xyz[0] + sub_dx, current_xyz[1] + sub_dy, current_xyz[2])
                            )
                            if plan["max_abs_delta_deg"] > args.max_step_joint_deg:
                                raise RuntimeError(
                                    "IK step refused: requested joint change "
                                    f"{plan['max_abs_delta_deg']:.1f} deg exceeds "
                                    f"{args.max_step_joint_deg:.1f} deg"
                                )
                            reached = arm.nudge_xy(sub_dx, sub_dy, steps=1, step_delay_s=0.10)
                            measured_z = float(arm.gripper_xyz()[2])
                            z_drift = measured_z - initial_gripper_z
                            if abs(z_drift) > args.max_z_drift_m:
                                raise RuntimeError(
                                    f"vertical drift {z_drift:+.4f} m exceeds "
                                    f"{args.max_z_drift_m:.4f} m safety limit"
                                )
                        return reached

                    if args.multi_step:
                        camera_to_base = np.asarray(payload["camera_to_base"], dtype=float)
                        inverse = np.linalg.inv(camera_to_base)
                        step_index = 0
                        while True:
                            joints = arm.get_joint_deg()
                            flange_pose = arm.kin.forward_kinematics(joints[:5])
                            offset = np.asarray(payload.get("tcp_offset_m", [0.0, 0.0, 0.0]), dtype=float)
                            tcp_base = flange_pose[:3, 3] + flange_pose[:3, :3] @ offset
                            tcp_camera = (inverse @ np.r_[tcp_base, 1.0])[:3]
                            remaining = compute_xy_nudge(
                                tcp_camera, labels["box"], camera_to_base, max_nudge_m=1.0
                            )
                            remaining_norm = float(np.linalg.norm(remaining))
                            if remaining_norm < 0.005:
                                print("multi-step correction complete")
                                break
                            scale = min(1.0, args.step_nudge_m / remaining_norm)
                            step = (remaining[0] * scale, remaining[1] * scale)
                            step_index += 1
                            reached = safe_nudge(step[0], step[1])
                            measured_z = float(arm.gripper_xyz()[2])
                            z_drift = measured_z - initial_gripper_z
                            if abs(z_drift) > args.max_z_drift_m:
                                raise RuntimeError(
                                    f"vertical drift {z_drift:+.4f} m exceeds "
                                    f"{args.max_z_drift_m:.4f} m safety limit"
                                )
                            print(
                                f"step {step_index}: dx={step[0]:.4f}, dy={step[1]:.4f} m; "
                                f"reached base TCP={tuple(float(v) for v in reached)}; "
                                f"z drift={z_drift:+.4f} m"
                            )
                            if step_index >= args.max_steps_per_run:
                                print("batch limit reached; re-run for a fresh camera measurement")
                                break
                    else:
                        reached = safe_nudge(delta[0], delta[1])
                        measured_z = float(arm.gripper_xyz()[2])
                        z_drift = measured_z - initial_gripper_z
                        if abs(z_drift) > args.max_z_drift_m:
                            raise RuntimeError(
                                f"vertical drift {z_drift:+.4f} m exceeds "
                                f"{args.max_z_drift_m:.4f} m safety limit"
                            )
                        print(f"reached base TCP: {tuple(float(v) for v in reached)}")
                    if args.hold_position:
                        holding = True
                        print("holding corrected position with torque enabled; press Ctrl+C to stop")
                        while True:
                            time.sleep(1.0)
                except KeyboardInterrupt:
                    print("visual-servo hold interrupted")
                finally:
                    if connected and not holding:
                        arm.disconnect()
                return 0
        else:
            print("박스 클릭 후 B 또는 E, 그리퍼 TCP 클릭 후 G")
        while True:
            ret, color, _ = camera.read()
            if not ret or color is None:
                time.sleep(0.05)
                continue
            display = color.copy()
            for label, pixel in label_pixels.items():
                cv2.circle(display, pixel, 8, (0, 255, 0), 2)
                cv2.putText(display, label, pixel, 0, 0.7, (0, 255, 255), 2)
            if selected[0] is not None:
                cv2.circle(display, selected[0], 8, (0, 0, 255), 2)
            cv2.putText(display, "click then B=box, G=TCP, Q=finish", (10, 25), 0, 0.65, (0, 255, 255), 2)
            cv2.imshow("visual servo RGB", display)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break
            allowed_keys = (ord("g"),) if args.local_prompt else (ord("b"), ord("e"), ord("g"))
            if selected[0] is None or key not in allowed_keys:
                continue
            clicked = selected[0]
            try:
                point = _capture_point(camera, camera.depth_stream, color, clicked)
            except ValueError as exc:
                print(f"{exc}")
                selected[0] = None
                continue
            label = "tcp" if args.local_prompt or key == ord("g") else "box"
            if label == "tcp":
                try:
                    held_joints = hold_tcp_pose(args.port)
                except Exception as exc:
                    print(f"TCP torque hold failed: {exc}")
                    selected[0] = None
                    continue
                print(f"TCP selected; torque holding current joints: {tuple(float(v) for v in held_joints)}")
            labels[label] = point
            label_pixels[label] = clicked
            print(f"{label} camera point: {tuple(point)}")
            selected[0] = None
            if "box" not in labels or "tcp" not in labels:
                continue
            delta = compute_xy_nudge(
                labels["tcp"],
                labels["box"],
                np.asarray(payload["camera_to_base"]),
                max_nudge_m=args.max_nudge_m,
            )
            print(f"suggested base XY nudge: dx={delta[0]:.4f}, dy={delta[1]:.4f} m")
            if not args.execute:
                print("dry visual-servo plan: no hardware command was issued")
                return 0
            if input("Type MOVE_NUDGE to apply one bounded XY correction: ").strip() != "MOVE_NUDGE":
                print("aborted: confirmation did not match")
                return 2
            arm = _build_execution_arm(args.port)
            connected = False
            holding = False
            try:
                arm.connect()
                connected = True
                reached = arm.nudge_xy(delta[0], delta[1], steps=12, step_delay_s=0.10)
                print(f"reached base TCP: {tuple(float(v) for v in reached)}")
                if args.hold_position:
                    holding = True
                    print("holding corrected position with torque enabled; press Ctrl+C to stop")
                    while True:
                        time.sleep(1.0)
            except KeyboardInterrupt:
                print("visual-servo hold interrupted")
            finally:
                if connected and not holding:
                    arm.disconnect()
            return 0
    finally:
        camera.release()
        if window_created:
            cv2.destroyWindow("visual servo RGB")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
