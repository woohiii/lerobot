"""Read and save the follower's current pose as a waypoint; no motor writes."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SOFollower


def main() -> int:
    """Save a serial read-only waypoint from the follower arm."""
    robot = SOFollower(SOFollowerRobotConfig(port="/dev/ttyACM0", id="follower", use_degrees=True))
    bus = robot.bus
    try:
        bus.connect(handshake=True)
        names = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
        joints = [float(bus.read("Present_Position", name)) for name in names]
    finally:
        if bus.is_connected:
            bus.disconnect(disable_torque=False)
    path = Path(__file__).resolve().parent / "safe_transit_waypoint.json"
    path.write_text(json.dumps({"created_utc": datetime.now(UTC).isoformat(), "mode": "READ_ONLY_NO_MOTOR_WRITES", "arm_joint_deg": joints}, indent=2) + "\n")
    print(f"[saved] {path}: {joints}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
