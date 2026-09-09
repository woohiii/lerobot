"""Hand-guided, torque-off IR-pixel <-> robot-xy calibration for the
task_trash_to_bin task (right arm only).

Same algorithm/output-shape/safety pattern as
``click_pick_place_safe/guided_manual_table_calibration_ir.py`` (that script
targets ``task_red_cube_to_bin_new_gripper``; this is the equivalent for the
new trimmed task_trash_to_bin, reading ground truth through this task's own
``SOArm101`` instead of a bare bus + build_kinematics()).

Ground truth is the robot's own reported end-effector xy
(``SOArm101.gripper_xyz()``, forward kinematics off Present_Position) - so
this DOES open the robot serial port, unlike a pure-camera calibration. The
only motor write is one user-authorized ``SOArm101.release_torque()`` call at
startup; every later robot read is Present_Position via ``gripper_xyz()``.
This never sends a goal position and leaves the arm limp on exit. Writes a
*candidate* file only - nothing named ``ir_homography.json`` is auto
-overwritten; promoting the candidate is a manual step after reviewing
``fit_error_mm``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

import config
from kinematics import SOArm101

TASK_DIR = Path(__file__).resolve().parent
POINTS_PATH = TASK_DIR / "ir_homography_calibration_points.json"
CANDIDATE_PATH = TASK_DIR / "ir_homography_candidate.json"
WINDOW = "IR homography calibration (trash_to_bin) - torque OFF"
MIN_POINTS = 9
# 30% margin on each side of the 640x480 IR frame, 3x3 grid - widened from the
# original 15% (96, 72) because the arm cannot reach the table plane at the
# outer raster points in this workspace (measured 55-59mm height offset there).
MARGIN_X, MARGIN_Y = 192, 144
_RASTER_POINTS = [(x, y) for y in (MARGIN_Y, 240, 480 - MARGIN_Y) for x in (MARGIN_X, 320, 640 - MARGIN_X)]
# Center-out ordering: nearest/easiest point first, corners (farthest hand
# reach) last, so difficulty ramps up instead of starting at the worst case.
GUIDE_POINTS = sorted(_RASTER_POINTS, key=lambda p: (p[0] - 320) ** 2 + (p[1] - 240) ** 2)


def fit_homography_candidate(samples: list[dict]) -> dict:
    """Pure function: fit an IR-pixel -> robot-xy homography from recorded
    samples. No file I/O, no hardware - split out so both main() and the
    self-test can exercise the same math."""
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
        "camera_source": str(config.ASTRA_IR_FRAME_PATH),
        "homography": homography.tolist(),
        "table_z_median_m": float(np.median([sample["gripper_xyz_m"][2] for sample in samples])),
        "pixel_points": pixels.tolist(),
        "robot_points": robot.tolist(),
        "inliers": [bool(value) for value in inlier_mask.ravel()],
        "fit_error_mm": {"mean": float(errors_mm.mean()), "max": float(errors_mm.max())},
        "activation_allowed": False,
        "required_before_activation": [
            "Review point coverage around where the trash/bin actually sit in IR.",
            "Perform held-out validation with at least two new hand-guided points.",
            "Explicitly approve promoting this to ir_homography.json (the file "
            "ir_wrist_hybrid_trash_to_bin.py --ir-homography should point at).",
        ],
    }


def main() -> int:
    """Guide manual IR-frame contacts against the live robot pose and save a
    non-active candidate."""
    arm = SOArm101(port=config.FOLLOWER_PORT)
    samples: list[dict] = []
    reference_z: float | None = None
    status = "Place the jaw center on the table at orange point 1, then press SPACE"

    try:
        arm.connect()
        arm.release_torque()  # explicitly requested: this is the only write in this program
        print("[torque OFF] Move the arm by hand. No automated movement will occur.")
        initial_image = cv2.imread(str(config.ASTRA_IR_FRAME_PATH))
        if initial_image is None:
            raise RuntimeError(
                f"Cannot read live Astra IR frame: {config.ASTRA_IR_FRAME_PATH} (is astra_s_ir_hub.py running?)"
            )
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, 960, 720)
        cv2.imshow(WINDOW, initial_image)
        cv2.waitKey(1)
        while len(samples) < MIN_POINTS:
            image = cv2.imread(str(config.ASTRA_IR_FRAME_PATH))
            if image is None:
                raise RuntimeError(f"Cannot read live Astra IR frame: {config.ASTRA_IR_FRAME_PATH}")
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

            xyz = arm.gripper_xyz()
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
                joints = arm.get_joint_deg()
                samples.append({"pixel": list(GUIDE_POINTS[len(samples)]), "joint_deg": joints.tolist(), "gripper_xyz_m": xyz.tolist()})
                status = f"Recorded point {len(samples)}/{MIN_POINTS}. Keep the same jaw-center/table contact for next point."
                print(f"[recorded] {len(samples)}/{MIN_POINTS}: xyz={np.round(xyz, 4)}")
        POINTS_PATH.write_text(json.dumps({"samples": samples}, indent=2) + "\n")
        candidate = fit_homography_candidate(samples)
        CANDIDATE_PATH.write_text(json.dumps(candidate, indent=2) + "\n")
        print(f"[saved] {POINTS_PATH}\n[saved candidate only] {CANDIDATE_PATH}")
        print(f"[fit] mean={candidate['fit_error_mm']['mean']:.2f} mm max={candidate['fit_error_mm']['max']:.2f} mm")
        return 0
    finally:
        if arm.robot.is_connected:
            arm.disconnect()
        cv2.destroyAllWindows()
        print("[safety] Torque was NOT re-enabled. The arm remains hand-movable.")


if __name__ == "__main__":
    raise SystemExit(main())
