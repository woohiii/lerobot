"""One conservative eye-in-hand calibration probe, with an automatic return.

It is intentionally limited to a 2 mm horizontal X or Y nudge at the already
verified hover height.  It neither descends nor changes the gripper.  The
pixel displacement measured before/after the nudge is the first column of
the wrist-camera Jacobian used for later closed-loop alignment.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

SAFE_DIR = Path(__file__).resolve().parent
TASK_DIR = SAFE_DIR.parent / "task_red_cube_to_bin_new_gripper"
sys.path.insert(0, str(TASK_DIR))

from kinematics import build_kinematics, gripper_position, solve_ik  # noqa: E402
from perception import detect_red_cube  # noqa: E402

from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig  # noqa: E402
from lerobot.robots.so_follower.so_follower import SOFollower  # noqa: E402

PORT = "/dev/ttyACM0"
ARM = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
ALL = ARM + ("gripper",)
PROBE_M = 0.002
# The commanded coarse hover is 130 mm; the live FK readback carries about
# 17 mm model/calibration bias.  100 mm keeps a conservative 10 cm gate while
# accepting that validated high pose, far above the measured home/table level.
MIN_HOVER_Z_M = 0.100
MAX_STEP_DEG, STEP_DELAY_S, MAX_LAG_DEG = 1.5, 0.18, 7.5
REPORT = SAFE_DIR / "safe_wrist_probe_report.json"


def positions(bus, names=ALL):
    """Read calibrated motor positions without commanding motion."""
    return {name: float(bus.read("Present_Position", name, normalize=True, num_retry=2)) for name in names}


def emergency_release(bus):
    """Best-effort immediate torque release after a safety failure."""
    for name in ALL:
        with contextlib.suppress(Exception):
            bus.disable_torque(name, num_retry=2)


def check(bus):
    """Raise when torque or electrical telemetry is unsafe for a nudge."""
    failures = []
    for name in ALL:
        if bus.read("Torque_Enable", name, normalize=False, num_retry=2) != 1:
            failures.append(f"{name}: torque off")
        if bus.read("Present_Temperature", name, normalize=False, num_retry=2) > 65:
            failures.append(f"{name}: over 65C")
        voltage = bus.read("Present_Voltage", name, normalize=False, num_retry=2)
        if not 90 <= voltage <= 140:
            failures.append(f"{name}: unsafe voltage")
    if failures:
        raise RuntimeError("; ".join(failures))


def frame_and_pixel(cap):
    """Require a current red-cube detection from the exclusive wrist camera."""
    # UVC commonly needs a short warm-up after a previous preview window
    # releases it; do not mistake its initial stale/black frames for loss.
    warmup_deadline = time.monotonic() + 1.5
    while time.monotonic() < warmup_deadline:
        cap.read()
        time.sleep(0.02)
    for _ in range(80):
        ok, frame = cap.read()
        if ok and frame is not None:
            det = detect_red_cube(frame)
            if det is not None:
                return [float(det.cx), float(det.cy)]
        time.sleep(0.03)
    raise RuntimeError("wrist camera did not produce a red-cube detection")


def move(bus, start, target):
    """Move only arm joints in short increments while checking every step."""
    largest = max(abs(target[name] - start[name]) for name in ARM)
    steps = max(1, int(np.ceil(largest / MAX_STEP_DEG)))
    for i in range(1, steps + 1):
        command = {name: start[name] + (target[name] - start[name]) * i / steps for name in ARM}
        for name in ARM:
            bus.write("Goal_Position", name, command[name], num_retry=2)
        time.sleep(STEP_DELAY_S)
        check(bus)
        actual = positions(bus, ARM)
        lag = max(abs(actual[name] - command[name]) for name in ARM)
        if lag > MAX_LAG_DEG:
            raise RuntimeError(f"tracking lag {lag:.2f}deg exceeds {MAX_LAG_DEG:.1f}deg")


def main():
    """Preview or execute the 2 mm probe and return to the same hover pose."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--workspace-clear", action="store_true")
    parser.add_argument("--axis", choices=("x", "y"), default="x")
    args = parser.parse_args()
    report, bus, cap = {"mode": "PREVIEW_ONLY"}, None, None
    try:
        robot = SOFollower(SOFollowerRobotConfig(port=PORT, id="follower", use_degrees=True))
        bus = robot.bus
        bus.connect(handshake=True)
        start = positions(bus)
        kin = build_kinematics()
        xyz = gripper_position(kin, np.array([start[name] for name in ARM]))
        if xyz[2] < MIN_HOVER_Z_M:
            raise RuntimeError(f"refuse: current model z={xyz[2]:.3f}m is below hover gate")
        target_xyz = (xyz[0] + PROBE_M, xyz[1], xyz[2]) if args.axis == "x" else (xyz[0], xyz[1] + PROBE_M, xyz[2])
        target_arm = solve_ik(kin, np.array([start[name] for name in ARM]), target_xyz)
        target_arm[4] = start["wrist_roll"]
        target = {**start, **dict(zip(ARM, map(float, target_arm), strict=True))}
        report.update(start_xyz_m=xyz.tolist(), target_joint_deg=target, max_delta_deg=max(abs(target[n]-start[n]) for n in ARM))
        if not args.execute:
            report["result"] = "PREVIEW_ONLY_NO_MOTOR_WRITES"
            return 0
        if not args.workspace_clear:
            raise RuntimeError("--workspace-clear is required")
        check(bus)
        cap = cv2.VideoCapture(4, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError("cannot open wrist camera /dev/video4")
        before = frame_and_pixel(cap)
        move(bus, start, target)
        after = frame_and_pixel(cap)
        move(bus, positions(bus), start)
        report.update(mode=f"EXECUTED_2MM_{args.axis.upper()}_PROBE_AND_RETURN", axis=args.axis, pixel_before=before, pixel_after=after,
                      pixel_delta_per_meter=((np.array(after)-np.array(before))/PROBE_M).tolist(), result="PROBE_COMPLETE_RETURNED_TO_HOVER")
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
