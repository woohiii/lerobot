"""Capture the SO-101 follower's present joint positions without actuation.

Only bus ping/firmware reads and ``Present_Position`` read requests are sent.
This deliberately does NOT call ``SOFollower.connect()``, because that method
configures PID/torque settings.  It never writes a motor register, never
enables/disables torque, and disconnects with ``disable_torque=False``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SOFollower

PORT = "/dev/ttyACM0"
OUTPUT = Path(__file__).resolve().parent / "home_pose_readonly.json"


def main() -> int:
    """Read normalized joint degrees without sending any motor write."""
    robot = SOFollower(SOFollowerRobotConfig(port=PORT, id="follower", use_degrees=True))
    bus = robot.bus
    try:
        bus.connect(handshake=True)  # ping/model/firmware reads only
        joint_deg = {name: float(bus.read("Present_Position", name)) for name in bus.motors}
    finally:
        if bus.is_connected:
            bus.disconnect(disable_torque=False)

    payload = {
        "created_utc": datetime.now(UTC).isoformat(),
        "port": PORT,
        "mode": "READ_ONLY_NO_MOTOR_WRITES",
        "joint_deg": joint_deg,
        "arm_joint_deg": [joint_deg[name] for name in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")],
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[saved] {OUTPUT}")
    print("[read-only] no torque or goal-position register was written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
