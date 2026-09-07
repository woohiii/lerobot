"""Apply one bounded, hover-only wrist-camera alignment correction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

SAFE_DIR = Path(__file__).resolve().parent
TASK_DIR = SAFE_DIR.parent / "task_red_cube_to_bin_new_gripper"
sys.path.insert(0, str(TASK_DIR))

from kinematics import build_kinematics, gripper_position, solve_ik  # noqa: E402
from safe_wrist_probe import (  # noqa: E402
    ARM,
    PORT,
    check,
    emergency_release,
    frame_and_pixel,
    move,
    positions,
)

from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig  # noqa: E402
from lerobot.robots.so_follower.so_follower import SOFollower  # noqa: E402

# Measured today by the returned 2 mm X/Y probes, columns=[dpx/dx, dpx/dy].
JACOBIAN_PX_PER_M = np.array([[-495.01798207828074, 3056.9781342166493], [-7614.269750109088, -3461.308750232348]])
MAX_ALIGN_STEP_M = 0.003
MIN_HOVER_Z_M = 0.100
REPORT = SAFE_DIR / "safe_wrist_align_report.json"


def main() -> int:
    """Preview or execute one camera-guided XY correction at fixed height."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--workspace-clear", action="store_true")
    args = parser.parse_args()
    report, bus, cap = {"mode": "PREVIEW_ONLY"}, None, None
    try:
        jaw = np.array(json.loads((SAFE_DIR / "wrist_jaw_center.json").read_text())["jaw_center_px"], dtype=float)
        robot = SOFollower(SOFollowerRobotConfig(port=PORT, id="follower", use_degrees=True))
        bus = robot.bus
        bus.connect(handshake=True)
        start = positions(bus)
        kin = build_kinematics()
        xyz = gripper_position(kin, np.array([start[name] for name in ARM]))
        if xyz[2] < MIN_HOVER_Z_M:
            raise RuntimeError(f"refuse: current model z={xyz[2]:.3f}m below hover gate")
        cap = cv2.VideoCapture(4, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError("cannot open wrist camera /dev/video4")
        before = np.array(frame_and_pixel(cap))
        error = jaw - before
        raw_step = np.linalg.solve(JACOBIAN_PX_PER_M, error)
        step = raw_step * min(1.0, MAX_ALIGN_STEP_M / float(np.linalg.norm(raw_step)))
        report.update(before_px=before.tolist(), jaw_center_px=jaw.tolist(), error_before_px=error.tolist(),
                      requested_xy_step_m=raw_step.tolist(), bounded_xy_step_m=step.tolist(), hover_xyz_m=xyz.tolist())
        if not args.execute:
            report["result"] = "PREVIEW_ONLY_NO_MOTOR_WRITES"
            return 0
        if not args.workspace_clear:
            raise RuntimeError("--workspace-clear is required")
        check(bus)
        target_arm = solve_ik(kin, np.array([start[name] for name in ARM]), (xyz[0] + step[0], xyz[1] + step[1], xyz[2]))
        target_arm[4] = start["wrist_roll"]
        target = {**start, **dict(zip(ARM, map(float, target_arm), strict=True))}
        move(bus, start, target)
        after = np.array(frame_and_pixel(cap))
        after_error = jaw - after
        if float(np.linalg.norm(after_error)) > float(np.linalg.norm(error)) + 10.0:
            raise RuntimeError("alignment error worsened by more than 10 px; stopping")
        report.update(mode="EXECUTED_ONE_BOUNDED_HOVER_ALIGNMENT", after_px=after.tolist(), error_after_px=after_error.tolist(),
                      result="ALIGNMENT_STEP_COMPLETE_TORQUE_HOLDING")
        return 0
    except Exception as exc:
        report.update(result="EMERGENCY_TORQUE_DISABLED" if args.execute else "PREVIEW_BLOCKED_NO_MOTOR_WRITES", error=str(exc))
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
