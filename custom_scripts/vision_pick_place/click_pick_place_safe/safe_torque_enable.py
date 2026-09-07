"""Conservative SO-101 torque re-enable with a no-motion hold guarantee.

Default mode is read-only.  Enabling requires both explicit CLI flags.  The
only permitted Goal_Position values are freshly-read Present_Position values;
any fault, unsafe telemetry, or unexpected post-enable motion disables torque
on every motor immediately.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SOFollower

PORT = "/dev/ttyACM0"
SAFE_DIR = Path(__file__).resolve().parent
REPORT = SAFE_DIR / "safe_torque_enable_report.json"
MOTORS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
MAX_TEMPERATURE_C = 65
MIN_VOLTAGE_RAW, MAX_VOLTAGE_RAW = 90, 140  # STS3215 reports 0.1 V units.
MAX_POST_ENABLE_DELTA = dict.fromkeys(MOTORS, 3.0) | {"gripper": 5.0}
MONITOR_SECONDS = 3.0


def _read(bus, field: str, *, normalize: bool) -> dict[str, float]:
    return {name: float(bus.read(field, name, normalize=normalize, num_retry=2)) for name in MOTORS}


def _telemetry(bus) -> dict:
    return {
        "torque_enable": _read(bus, "Torque_Enable", normalize=False),
        "present_position": _read(bus, "Present_Position", normalize=True),
        "temperature_c": _read(bus, "Present_Temperature", normalize=False),
        "voltage_raw_tenths_v": _read(bus, "Present_Voltage", normalize=False),
        "current_raw": _read(bus, "Present_Current", normalize=False),
    }


def _telemetry_safe(snapshot: dict) -> list[str]:
    failures: list[str] = []
    for name in MOTORS:
        if snapshot["temperature_c"][name] > MAX_TEMPERATURE_C:
            failures.append(f"{name}: temperature above {MAX_TEMPERATURE_C}C")
        voltage = snapshot["voltage_raw_tenths_v"][name]
        if not MIN_VOLTAGE_RAW <= voltage <= MAX_VOLTAGE_RAW:
            failures.append(f"{name}: voltage raw={voltage} outside safe range")
    return failures


def _disable_all(bus) -> None:
    """Best-effort emergency torque release; no position command is sent."""
    for name in MOTORS:
        with contextlib.suppress(Exception):
            bus.disable_torque(name, num_retry=2)


def main() -> int:
    """Run a read-only preflight or explicit conservative torque enable."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--enable-torque", action="store_true", help="allow the controlled torque-enable phase")
    parser.add_argument("--arm-supported", action="store_true", help="confirms a person is supporting the limp arm")
    args = parser.parse_args()
    requested_enable = args.enable_torque and args.arm_supported
    robot = SOFollower(SOFollowerRobotConfig(port=PORT, id="follower", use_degrees=True))
    bus = robot.bus
    report: dict = {"created_utc": datetime.now(UTC).isoformat(), "mode": "READ_ONLY_PREFLIGHT"}
    success = False
    try:
        bus.connect(handshake=True)
        before = _telemetry(bus)
        report["before"] = before
        failures = _telemetry_safe(before)
        if any(value != 0 for value in before["torque_enable"].values()):
            failures.append("one or more joints already have torque enabled; refusing state change")
        report["preflight_failures"] = failures
        if not requested_enable:
            report["result"] = "PREFLIGHT_ONLY_NO_MOTOR_WRITES"
            success = not failures
            return 0 if success else 2
        if failures:
            report["result"] = "BLOCKED_BEFORE_ENABLE"
            return 2

        # All writes below happen while torque is off.  Goals are exactly the
        # observed positions, followed by deliberately conservative limits.
        for name, position in before["present_position"].items():
            bus.write("Goal_Position", name, position)
            bus.write("Max_Torque_Limit", name, 350)      # 35% maximum torque
            bus.write("Protection_Current", name, 180)    # lower than existing 250 safety setting
            bus.write("Overload_Torque", name, 20)        # 20% overload threshold
        report["mode"] = "CONSERVATIVE_HOLD_ENABLE"
        report["goal_position_equals_present_position"] = before["present_position"]

        # Enable quickly only after every goal/limit write completed; the arm
        # should hold the exact pose it was manually placed in.
        for name in MOTORS:
            bus.enable_torque(name, num_retry=2)
        baseline = _read(bus, "Present_Position", normalize=True)
        deadline = time.monotonic() + MONITOR_SECONDS
        max_delta = dict.fromkeys(MOTORS, 0.0)
        while time.monotonic() < deadline:
            now = _read(bus, "Present_Position", normalize=True)
            for name in MOTORS:
                max_delta[name] = max(max_delta[name], abs(now[name] - baseline[name]))
                if max_delta[name] > MAX_POST_ENABLE_DELTA[name]:
                    raise RuntimeError(f"unexpected post-enable position change: {name} moved {max_delta[name]:.2f}")
            time.sleep(0.05)
        report["post_enable_max_delta"] = max_delta
        report["after"] = _telemetry(bus)
        report["result"] = "TORQUE_ENABLED_HOLD_STABLE"
        success = True
        return 0
    except Exception as exc:
        report["result"] = "EMERGENCY_TORQUE_DISABLED"
        report["error"] = str(exc)
        if bus.is_connected:
            _disable_all(bus)
        return 2
    finally:
        REPORT.write_text(json.dumps(report, indent=2) + "\n")
        if bus.is_connected:
            bus.disconnect(disable_torque=False)
        print(f"[saved] {REPORT}; result={report.get('result')}")
        if not success and requested_enable:
            print("[safety] torque was disabled after a failed enable attempt")


if __name__ == "__main__":
    raise SystemExit(main())
