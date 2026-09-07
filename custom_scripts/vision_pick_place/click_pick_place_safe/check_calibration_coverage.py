"""Check whether clicked pick/place pixels are inside measured camera calibration coverage.

This uses files only.  It neither opens a camera nor connects to the robot.
For a first physical run, extrapolating a table homography outside its taught
pixel polygon is rejected even if an older, permissive workspace margin would
allow it.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

SAFE_DIR = Path(__file__).resolve().parent
VISION_DIR = SAFE_DIR.parent
PLAN_PATH = SAFE_DIR / "latest_plan.json"
HOMOGRAPHY_PATH = VISION_DIR / "homography.json"
OUTPUT_PATH = SAFE_DIR / "calibration_coverage_report.json"
OVERLAY_PATH = SAFE_DIR / "calibration_coverage.png"


def main() -> int:
    """Write strict image-space calibration coverage results for the current plan."""
    plan = json.loads(PLAN_PATH.read_text())
    homography = json.loads(HOMOGRAPHY_PATH.read_text())
    image = cv2.imread(plan["inputs"]["rgb"])
    if image is None:
        raise RuntimeError("The RGB frame used by the plan is no longer readable")

    taught = np.array(homography["pixel_points"], dtype=np.float32)
    hull = cv2.convexHull(taught).reshape(-1, 2)
    result: dict[str, dict] = {}
    for name, point in {"grasp_center": plan["inputs"]["grasp_center_px"], "drop": plan["inputs"]["drop_px"]}.items():
        signed_distance = float(cv2.pointPolygonTest(hull, tuple(float(v) for v in point), True))
        result[name] = {
            "pixel": point,
            "inside_strict_calibration_hull": signed_distance >= 0.0,
            "signed_distance_to_hull_px": signed_distance,
        }

    canvas = image.copy()
    cv2.polylines(canvas, [hull.astype(np.int32)], True, (0, 255, 255), 2)
    for name, item in result.items():
        x, y = item["pixel"]
        color = (0, 255, 0) if item["inside_strict_calibration_hull"] else (0, 0, 255)
        cv2.circle(canvas, (x, y), 7, color, -1)
        cv2.putText(canvas, f"{name}: {'inside' if item['inside_strict_calibration_hull'] else 'OUTSIDE'}", (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2)
    cv2.putText(canvas, "YELLOW: taught homography region. RED: re-calibration required.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 255), 2)
    cv2.imwrite(str(OVERLAY_PATH), canvas)

    all_inside = all(item["inside_strict_calibration_hull"] for item in result.values())
    report = {
        "mode": "OFFLINE_CALIBRATION_COVERAGE_CHECK_NO_HARDWARE",
        "strict_coverage_pass": all_inside,
        "targets": result,
        "required_before_real_motion": [] if all_inside else [
            "Capture additional camera-to-base calibration points covering the red cube and black-box pixels.",
            "Validate the new homography with held-out points before any descent.",
        ],
        "overlay": str(OVERLAY_PATH),
    }
    OUTPUT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[saved] {OUTPUT_PATH}")
    print(f"[strict coverage] {'PASS' if all_inside else 'FAIL — re-calibration required'}")
    return 0 if all_inside else 2


if __name__ == "__main__":
    raise SystemExit(main())
