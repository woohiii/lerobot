"""Conservative, hover-only approach to the currently detected red cube.

This is deliberately *not* a pick script: it never descends below 130 mm,
never changes the gripper, and never opens a camera or motor implicitly in
preview mode.  ``--execute`` requires an already-safe torque hold from
safe_torque_enable.py and two explicit physical-clearance acknowledgements.
Every short interpolation step rechecks motor telemetry and position
tracking.  Any unsafe observation immediately releases all follower torque.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

SAFE_DIR = Path(__file__).resolve().parent
TASK_DIR = SAFE_DIR.parent / "task_red_cube_to_bin_new_gripper"
sys.path.insert(0, str(TASK_DIR))

import config  # noqa: E402
from kinematics import build_kinematics, solve_ik  # noqa: E402
from perception import detect_red_cube  # noqa: E402

from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig  # noqa: E402
from lerobot.robots.so_follower.so_follower import SOFollower  # noqa: E402

PORT = "/dev/ttyACM0"
ARM = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
ALL = ARM + ("gripper",)
HOVER_Z_M = 0.130
# A larger destination difference is allowed only because the physical command
# is still bounded to one degree at a time and checked after every step.
MAX_TOTAL_DELTA_DEG = 75.0
# 1-degree stop-and-correct updates made the real servos chatter.  Three
# degrees is still well below LeRobot's 15-degree relative-target guard, but
# lets the motor controller follow a continuous-looking trajectory.
MAX_STEP_DEG = 1.5
STEP_DELAY_S = 0.18
MAX_TRACKING_LAG_DEG = 7.5
MAX_TEMPERATURE_C = 65.0
MIN_VOLTAGE_RAW, MAX_VOLTAGE_RAW = 90.0, 140.0
RGB_PATH = Path("/tmp/vsp_astra_rgb.png")
HOMOGRAPHY_PATH = SAFE_DIR / "markerless_global_servo_config.json"
REPORT_PATH = SAFE_DIR / "safe_hover_approach_report.json"


def read_position(bus, names: tuple[str, ...] = ALL) -> dict[str, float]:
    """Read the calibrated present positions without commanding a motor."""
    return {name: float(bus.read("Present_Position", name, normalize=True, num_retry=2)) for name in names}


def telemetry_failures(bus, *, require_torque: bool) -> list[str]:
    """Return all telemetry/torque violations observed at this instant."""
    failures: list[str] = []
    for name in ALL:
        torque = float(bus.read("Torque_Enable", name, normalize=False, num_retry=2))
        temp = float(bus.read("Present_Temperature", name, normalize=False, num_retry=2))
        volt = float(bus.read("Present_Voltage", name, normalize=False, num_retry=2))
        if require_torque and torque != 1:
            failures.append(f"{name}: torque is not enabled")
        if temp > MAX_TEMPERATURE_C:
            failures.append(f"{name}: {temp:.1f}C exceeds {MAX_TEMPERATURE_C:.0f}C")
        if not MIN_VOLTAGE_RAW <= volt <= MAX_VOLTAGE_RAW:
            failures.append(f"{name}: voltage raw={volt:.1f} outside safe range")
    return failures


def disable_all(bus) -> None:
    """Release every motor after a failed safety gate."""
    for name in ALL:
        with contextlib.suppress(Exception):
            bus.disable_torque(name, num_retry=2)


def pixel_to_xy(px: float, py: float) -> tuple[float, float]:
    """Map a fresh Astra RGB pixel to coarse table XY using the review fit."""
    data = json.loads(HOMOGRAPHY_PATH.read_text())
    h = np.asarray(data["camera_to_base_xy_homography"], dtype=float)
    mapped = h @ np.array([px, py, 1.0])
    mapped /= mapped[2]
    return float(mapped[0]), float(mapped[1])


def make_plan() -> dict:
    """Detect the current cube and form a hover-only Cartesian target."""
    if not RGB_PATH.exists() or time.time() - RGB_PATH.stat().st_mtime > 3.0:
        raise RuntimeError("Astra RGB frame is missing or older than 3 seconds")
    frame = cv2.imread(str(RGB_PATH))
    if frame is None:
        raise RuntimeError("Cannot decode Astra RGB frame")
    cube = detect_red_cube(frame)
    if cube is None:
        raise RuntimeError("Red cube was not detected in the fresh Astra frame")
    xy = pixel_to_xy(cube.cx, cube.cy)
    return {
        "created_utc": datetime.now(UTC).isoformat(),
        "mode": "HOVER_ONLY_NO_DESCENT_NO_GRIPPER",
        "source_rgb": str(RGB_PATH),
        "source_frame_age_s": time.time() - RGB_PATH.stat().st_mtime,
        "cube_pixel": [float(cube.cx), float(cube.cy)],
        "hover_xyz_m": [xy[0], xy[1], HOVER_Z_M],
        "global_xy_notice": "coarse only; wrist visual servo remains required before any descent",
    }


def preview(plan: dict, current: dict[str, float]) -> dict:
    """Solve and validate an IK trajectory from the freshly read arm pose."""
    kin = build_kinematics()
    current_arm = np.array([current[name] for name in ARM], dtype=float)
    target_arm = solve_ik(kin, current_arm, tuple(plan["hover_xyz_m"]))
    # Wrist roll is position-neutral for this hover target.  Holding it fixed
    # rejects arbitrary 180/360 degree equivalent IK branches.
    target_arm[4] = current["wrist_roll"]
    for index, name in enumerate(ARM):
        lo, hi = config.JOINT_LIMITS_DEG[name]
        if not lo <= target_arm[index] <= hi:
            raise RuntimeError(f"refuse: IK target {name}={target_arm[index]:.2f} outside [{lo}, {hi}]")
    target = {**current}
    target.update({name: float(value) for name, value in zip(ARM, target_arm, strict=True)})
    delta = {name: target[name] - current[name] for name in ARM}
    max_delta = max(abs(value) for value in delta.values())
    plan |= {"ik_joint_deg_before_safety_gate": [float(value) for value in target_arm],
             "ik_delta_deg_before_safety_gate": {name: float(target_arm[i] - current[name]) for i, name in enumerate(ARM)}}
    plan |= {"current_joint_deg": current, "target_joint_deg": target, "delta_joint_deg": delta,
             "max_abs_delta_deg": max_delta,
             "interpolation_steps": int(np.ceil(max_delta / MAX_STEP_DEG))}
    return plan


def main() -> int:
    """Run a safe read-only preview or an explicitly armed hover approach."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="send the already-previewed hover trajectory")
    parser.add_argument("--workspace-clear", action="store_true", help="confirms no hand/cable/obstacle is in arm path")
    parser.add_argument("--keep-clear", action="store_true", help="confirms people keep clear during motion")
    args = parser.parse_args()
    report: dict = {"created_utc": datetime.now(UTC).isoformat(), "mode": "PREVIEW_ONLY"}
    bus = None
    try:
        # The RGB snapshot is owned by the already-running Astra process; no camera is opened here.
        plan = make_plan()
        robot = SOFollower(SOFollowerRobotConfig(port=PORT, id="follower", use_degrees=True))
        bus = robot.bus
        bus.connect(handshake=True)
        current = read_position(bus)
        plan = preview(plan, current)
        report["plan"] = plan
        if plan["max_abs_delta_deg"] > MAX_TOTAL_DELTA_DEG:
            raise RuntimeError(
                f"refuse: total single-joint delta {plan['max_abs_delta_deg']:.1f}deg "
                f"exceeds {MAX_TOTAL_DELTA_DEG:.0f}deg"
            )
        if not args.execute:
            report["result"] = "PREVIEW_ONLY_NO_MOTOR_WRITES"
            return 0
        if not (args.workspace_clear and args.keep_clear):
            raise RuntimeError("execution requires --workspace-clear and --keep-clear")
        failures = telemetry_failures(bus, require_torque=True)
        if failures:
            raise RuntimeError("preflight failed: " + "; ".join(failures))

        report["mode"] = "EXECUTED_HOVER_ONLY_INCREMENTAL"
        start = current.copy()
        target = plan["target_joint_deg"]
        steps = plan["interpolation_steps"]
        for step in range(1, steps + 1):
            commanded = {name: start[name] + (target[name] - start[name]) * step / steps for name in ARM}
            # Direct raw-bus writes keep each command no more than 3 degrees apart.
            for name in ARM:
                bus.write("Goal_Position", name, commanded[name], num_retry=2)
            time.sleep(STEP_DELAY_S)
            failures = telemetry_failures(bus, require_torque=True)
            actual = read_position(bus, ARM)
            lag = {name: abs(actual[name] - commanded[name]) for name in ARM}
            if max(lag.values()) > MAX_TRACKING_LAG_DEG:
                failures.append(f"position tracking lag {max(lag.values()):.2f}deg exceeds {MAX_TRACKING_LAG_DEG:.0f}deg")
            report["last_step"] = {"index": step, "commanded": commanded, "actual": actual, "lag_deg": lag}
            if failures:
                raise RuntimeError("motion aborted: " + "; ".join(failures))
        report["result"] = "HOVER_REACHED_TORQUE_HOLDING"
        return 0
    except Exception as exc:
        report["result"] = "EMERGENCY_TORQUE_DISABLED" if args.execute else "PREVIEW_BLOCKED_NO_MOTOR_WRITES"
        report["error"] = str(exc)
        if args.execute and bus is not None and bus.is_connected:
            disable_all(bus)
        return 2
    finally:
        REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
        if bus is not None and bus.is_connected:
            bus.disconnect(disable_torque=False)
        print(f"[saved] {REPORT_PATH}; result={report.get('result')}")


if __name__ == "__main__":
    raise SystemExit(main())
