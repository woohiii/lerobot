"""Hand-guided, torque-off table calibration for the SO-101, against the
Astra S **IR** view (not RGB - see astra_s_ir_hub.py's docstring for why IR
and Depth can't run together, and why this is a separate camera source from
guided_manual_table_calibration.py's RGB one). IR and RGB are different
sensors on the same Astra S unit with their own parallax/extrinsics, so the
existing RGB-fit homography.json is not reusable here - this produces its
own IR-pixel <-> robot-xy fit.

The only motor write is a single, user-authorized Torque_Enable=0 operation
at startup. Every later robot operation is Present_Position reading. This
program never sends a goal position, never enables torque, and leaves the arm
limp on exit. It writes a *candidate* homography; nothing existing is
overwritten.
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
sys.path.insert(0, str(SAFE_DIR.parent))  # camera_utils.py lives in vision_pick_place/, not task_dir

from camera_utils import ASTRA_IR_FRAME_PATH  # noqa: E402
from kinematics import build_kinematics, gripper_position  # noqa: E402

PORT = "/dev/ttyACM0"
POINTS_PATH = SAFE_DIR / "manual_table_calibration_points_ir.json"
CANDIDATE_PATH = SAFE_DIR / "homography_candidate_ir.json"
WINDOW = "Manual IR table calibration - torque OFF"
MIN_POINTS = 9
# 15% margin on each side of the 640x480 IR frame, 3x3 grid - deliberately not
# derived from any RGB-camera plan's pixel positions (a different sensor with
# different parallax; those pixels don't mean the same thing here).
MARGIN_X, MARGIN_Y = 96, 72
_RASTER_POINTS = [(x, y) for y in (MARGIN_Y, 240, 480 - MARGIN_Y) for x in (MARGIN_X, 320, 640 - MARGIN_X)]
# 2026-09-02: user reported the first raster point (a far corner) was an
# uncomfortably long hand-reach to START with, torque off. Center-out ordering
# instead - nearest/easiest point (frame center) first, corners (farthest
# reach) last, so difficulty ramps up instead of starting at the worst case.
GUIDE_POINTS = sorted(_RASTER_POINTS, key=lambda p: (p[0] - 320) ** 2 + (p[1] - 240) ** 2)


def _read_arm_deg(bus) -> np.ndarray:
    names = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
    return np.array([float(bus.read("Present_Position", name)) for name in names])


def _write_candidate(samples: list[dict]) -> dict:
    pixels = np.array([sample["pixel"] for sample in samples], dtype=np.float32)
    robot = np.array([sample["gripper_xyz_m"][:2] for sample in samples], dtype=np.float32)
    homography, inlier_mask = cv2.findHomography(pixels, robot, cv2.RANSAC, 0.006)
    if homography is None:
        raise RuntimeError("Could not fit a homography from the recorded points")
    predicted = cv2.perspectiveTransform(pixels.reshape(-1, 1, 2), homography).reshape(-1, 2)
    errors_mm = np.linalg.norm(predicted - robot, axis=1) * 1000.0
    return {
        "created_utc": datetime.now(UTC).isoformat(),
        "mode": "CANDIDATE_ONLY_MANUAL_TORQUE_OFF_CALIBRATION_IR",
        "camera_source": str(ASTRA_IR_FRAME_PATH),
        "homography": homography.tolist(),
        "table_z_median_m": float(np.median([sample["gripper_xyz_m"][2] for sample in samples])),
        "pixel_points": pixels.tolist(),
        "robot_points": robot.tolist(),
        "inliers": [bool(value) for value in inlier_mask.ravel()],
        "fit_error_mm": {"mean": float(errors_mm.mean()), "max": float(errors_mm.max())},
        "activation_allowed": False,
        "required_before_activation": [
            "Review point coverage around where the object/drop target actually sit in IR.",
            "Perform held-out validation with at least two new hand-guided points.",
            "Explicitly approve promoting this to homography_ir.json.",
        ],
    }


def main() -> int:
    """Guide manual table contacts against the live IR frame and save a non-active candidate."""
    robot = SOFollower(SOFollowerRobotConfig(port=PORT, id="follower", use_degrees=True))
    bus = robot.bus
    kin = build_kinematics()
    samples: list[dict] = []
    reference_z: float | None = None
    status = "Place the jaw center on the table at orange point 1, then press SPACE"

    try:
        bus.connect(handshake=True)
        bus.disable_torque()  # explicitly requested: this is the only write in this program
        print("[torque OFF] Move the arm by hand. No automated movement will occur.")
        initial_image = cv2.imread(str(ASTRA_IR_FRAME_PATH))
        if initial_image is None:
            raise RuntimeError(f"Cannot read live Astra IR frame: {ASTRA_IR_FRAME_PATH} (is astra_s_ir_hub.py running?)")
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, 960, 720)
        cv2.imshow(WINDOW, initial_image)
        cv2.waitKey(1)
        while len(samples) < MIN_POINTS:
            image = cv2.imread(str(ASTRA_IR_FRAME_PATH))
            if image is None:
                raise RuntimeError(f"Cannot read live Astra IR frame: {ASTRA_IR_FRAME_PATH}")
            canvas = image.copy()
            for index, point in enumerate(GUIDE_POINTS, start=1):
                color = (0, 165, 255) if index == len(samples) + 1 else (140, 140, 140)
                cv2.drawMarker(canvas, point, color, cv2.MARKER_CROSS, 18, 2)
                cv2.putText(canvas, str(index), (point[0] + 5, point[1] + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)
            for index, sample in enumerate(samples, start=1):
                point = tuple(sample["pixel"])
                cv2.circle(canvas, point, 5, (0, 255, 255), -1)
                cv2.putText(canvas, str(index), (point[0] + 6, point[1] + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            message = f"{len(samples) + 1}/{MIN_POINTS}: align TCP with ORANGE cross on table, then SPACE"
            cv2.putText(canvas, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            cv2.putText(canvas, status, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # 2026-09-02: user reported hitting REJECTED 3x in a row on real hardware -
            # the check only ever ran (and reported the miss) AFTER a SPACE press, so
            # there was no way to see "how close am I" while still nudging the arm by
            # hand. Read + show it live every frame instead, so the user can watch the
            # number settle before pressing SPACE rather than finding out after.
            joints = _read_arm_deg(bus)
            xyz = gripper_position(kin, joints)
            if reference_z is None:
                live_text, live_color = "Live height: this will become the reference at point 1", (0, 255, 255)
            else:
                delta_mm = (float(xyz[2]) - reference_z) * 1000.0
                ok = abs(delta_mm) <= 15.0
                live_color = (0, 220, 0) if ok else (0, 100, 255)
                live_text = f"Live height vs reference: {delta_mm:+.1f} mm ({'OK' if ok else 'adjust, target +-15mm'})"
            cv2.putText(canvas, live_text, (10, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.55, live_color, 2)
            cv2.imshow(WINDOW, canvas)
            key = cv2.waitKey(25) & 0xFF
            if key in (27, ord("q")):
                print("[cancelled] torque remains OFF; no calibration candidate written")
                return 0
            if key == ord(" "):
                if reference_z is None:
                    reference_z = float(xyz[2])
                elif abs(float(xyz[2]) - reference_z) > 0.015:
                    delta_mm = (float(xyz[2]) - reference_z) * 1000.0
                    status = f"REJECTED: model TCP height differs {delta_mm:+.1f} mm. Re-seat jaw center on table, then SPACE."
                    print(f"[rejected] TCP height differs by {abs(delta_mm):.1f} mm; keep the tip on the same table plane")
                    continue
                samples.append({"pixel": list(GUIDE_POINTS[len(samples)]), "joint_deg": joints.tolist(), "gripper_xyz_m": xyz.tolist()})
                status = f"Recorded point {len(samples)}/{MIN_POINTS}. Keep the same jaw-center/table contact for next point."
                print(f"[recorded] {len(samples)}/{MIN_POINTS}: xyz={np.round(xyz, 4)}")
        POINTS_PATH.write_text(json.dumps({"samples": samples}, indent=2) + "\n")
        candidate = _write_candidate(samples)
        CANDIDATE_PATH.write_text(json.dumps(candidate, indent=2) + "\n")
        print(f"[saved] {POINTS_PATH}\n[saved candidate only] {CANDIDATE_PATH}")
        print(f"[fit] mean={candidate['fit_error_mm']['mean']:.2f} mm max={candidate['fit_error_mm']['max']:.2f} mm")
        return 0
    finally:
        if bus.is_connected:
            bus.disconnect(disable_torque=False)
        cv2.destroyAllWindows()
        print("[safety] Torque was NOT re-enabled. The arm remains hand-movable.")


if __name__ == "__main__":
    raise SystemExit(main())
