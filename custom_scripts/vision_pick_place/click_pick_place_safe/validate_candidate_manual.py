"""Read-only held-out validation of the manual homography candidate.

Torque must already be off.  This script performs no motor write: it opens
the bus, reads present positions at two user hand-guided image targets, then
compares forward kinematics with the candidate homography.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
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
WINDOW = "Candidate validation - READ ONLY - torque remains OFF"
OUTPUT = SAFE_DIR / "homography_candidate_heldout_validation.json"


def _read_arm_deg(bus) -> np.ndarray:
    names = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
    return np.array([float(bus.read("Present_Position", name)) for name in names])


def main() -> int:
    """Collect cube and box held-out references without motor writes."""
    plan = json.loads((SAFE_DIR / "latest_plan.json").read_text())
    candidate = json.loads((SAFE_DIR / "homography_candidate_ransac_review.json").read_text())
    matrix = np.array(candidate["homography"], dtype=np.float64)
    targets = [("cube", plan["inputs"]["grasp_center_px"]), ("box", plan["inputs"]["drop_px"])]
    robot = SOFollower(SOFollowerRobotConfig(port=PORT, id="follower", use_degrees=True))
    bus, kin = robot.bus, build_kinematics()
    records: list[dict] = []
    try:
        bus.connect(handshake=True)  # read-only ping/model/firmware handshake
        torque = {name: int(bus.read("Torque_Enable", name, normalize=False)) for name in bus.motors}
        if any(torque.values()):
            raise RuntimeError("Torque is not off on every joint; refusing manual validation")
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, 1000, 700)
        while len(records) < len(targets):
            image = cv2.imread(str(RGB_PATH))
            if image is None:
                raise RuntimeError("Cannot read current Astra RGB frame")
            canvas = image.copy()
            label, point = targets[len(records)]
            cv2.drawMarker(canvas, tuple(point), (0, 165, 255), cv2.MARKER_CROSS, 24, 2)
            cv2.putText(canvas, f"Hand-align jaw center with {label} cross; SPACE to read only", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            cv2.imshow(WINDOW, canvas)
            key = cv2.waitKey(25) & 0xFF
            if key in (27, ord("q")):
                return 0
            if key == ord(" "):
                observed = gripper_position(kin, _read_arm_deg(bus))
                predicted = cv2.perspectiveTransform(np.array(point, dtype=np.float64).reshape(1, 1, 2), matrix).reshape(2)
                error_mm = float(np.linalg.norm(observed[:2] - predicted) * 1000.0)
                records.append({"target": label, "pixel": point, "observed_xyz_m": observed.tolist(), "candidate_xy_m": predicted.tolist(), "xy_error_mm": error_mm})
                print(f"[recorded] {label}: xy error={error_mm:.2f} mm")
    finally:
        if bus.is_connected:
            bus.disconnect(disable_torque=False)
        cv2.destroyAllWindows()

    errors = [item["xy_error_mm"] for item in records]
    passed = len(records) == 2 and max(errors) <= 8.0
    OUTPUT.write_text(json.dumps({
        "created_utc": datetime.now(UTC).isoformat(),
        "mode": "MANUAL_HELDOUT_READ_ONLY_NO_MOTOR_WRITES",
        "torque_writes": 0,
        "records": records,
        "pass_threshold_mm": 8.0,
        "passed": passed,
        "activation_allowed": False,
        "next_gate": "explicit review; candidate is never applied automatically",
    }, indent=2) + "\n")
    print(f"[saved] {OUTPUT}; held-out {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
