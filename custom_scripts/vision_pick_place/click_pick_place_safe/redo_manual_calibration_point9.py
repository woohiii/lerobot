"""Replace only point 9 of a saved manual calibration, with no motor writes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SOFollower

SAFE_DIR = Path(__file__).resolve().parent
TASK_DIR = SAFE_DIR.parent / "task_red_cube_to_bin_new_gripper"
sys.path.insert(0, str(TASK_DIR))

from kinematics import build_kinematics, gripper_position  # noqa: E402, I001


PORT = "/dev/ttyACM0"
RGB_PATH = Path("/tmp/vsp_astra_rgb.png")
POINTS_PATH = SAFE_DIR / "manual_table_calibration_points.json"
WINDOW = "Redo calibration point 9 - READ ONLY - torque OFF"


def main() -> int:
    """Hand-align the jaw at grid point 9 and replace just that saved sample."""
    data = json.loads(POINTS_PATH.read_text())
    samples = data["samples"]
    if len(samples) != 9:
        raise RuntimeError("Expected exactly nine saved samples")
    plan = json.loads((SAFE_DIR / "latest_plan.json").read_text())
    cube, box = plan["inputs"]["grasp_center_px"], plan["inputs"]["drop_px"]
    image = cv2.imread(str(RGB_PATH))
    if image is None:
        raise RuntimeError("Cannot read current Astra RGB frame")
    x1 = min(image.shape[1] - 10, max(cube[0], box[0]) + 60)
    y1 = min(image.shape[0] - 10, max(cube[1], box[1]) + 75)
    target = [x1, y1]
    robot = SOFollower(SOFollowerRobotConfig(port=PORT, id="follower", use_degrees=True))
    bus, kin = robot.bus, build_kinematics()
    try:
        bus.connect(handshake=True)
        if any(int(bus.read("Torque_Enable", name, normalize=False)) for name in bus.motors):
            raise RuntimeError("Torque must remain OFF for manual point redo")
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, 1000, 700)
        while True:
            frame = cv2.imread(str(RGB_PATH))
            if frame is None:
                raise RuntimeError("Astra RGB stream stopped")
            cv2.drawMarker(frame, tuple(target), (0, 165, 255), cv2.MARKER_CROSS, 28, 2)
            cv2.putText(frame, "Move jaw center to orange point 9 on table; SPACE replaces point 9", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 255), 2)
            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(25) & 0xFF
            if key in (27, ord("q")):
                return 0
            if key == ord(" "):
                names = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
                joints = np.array([float(bus.read("Present_Position", name)) for name in names])
                xyz = gripper_position(kin, joints)
                samples[8] = {"pixel": target, "joint_deg": joints.tolist(), "gripper_xyz_m": xyz.tolist()}
                POINTS_PATH.write_text(json.dumps(data, indent=2) + "\n")
                print(f"[replaced point 9] xyz={np.round(xyz, 4)}")
                return 0
    finally:
        if bus.is_connected:
            bus.disconnect(disable_torque=False)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
