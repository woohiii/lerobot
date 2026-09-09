"""Hardware-free self-test for calibrate_ir_homography's homography math and
its compatibility with ir_wrist_hybrid_trash_to_bin.py's loader.

Run: uv run python3 custom_scripts/vision_pick_place/task_trash_to_bin/test_calibrate_ir_homography.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

import calibrate_ir_homography as cih

VISION_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(VISION_DIR))
import ir_wrist_hybrid_trash_to_bin as hybrid  # noqa: E402

# A known affine map (rotate + scale + translate) standing in for a real
# camera<->robot extrinsic, so the fitted homography has a ground truth to be
# checked against.
_A = np.array([[0.0004, -0.00005], [0.00003, 0.00035]])
_B = np.array([0.02, -0.05])


def _synthetic_samples() -> list[dict]:
    pixels = cih.GUIDE_POINTS  # 9 known pixel points, real script's own raster
    samples = []
    for px in pixels:
        xy = _A @ np.array(px, dtype=float) + _B
        samples.append({"pixel": list(px), "joint_deg": [0.0] * 6, "gripper_xyz_m": [float(xy[0]), float(xy[1]), 0.1]})
    return samples


def test_round_trip_synthetic_points():
    candidate = cih.fit_homography_candidate(_synthetic_samples())
    homography = np.asarray(candidate["homography"])
    assert homography.shape == (3, 3), f"homography shape 잘못됨: {homography.shape}"
    assert np.isfinite(homography).all(), "homography에 non-finite 값 존재"
    assert candidate["fit_error_mm"]["max"] < 1.0, f"합성 점 왕복 오차가 너무 큼: {candidate['fit_error_mm']}"


def test_compatible_with_hybrid_loader():
    candidate = cih.fit_homography_candidate(_synthetic_samples())
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ir_homography_candidate.json"
        path.write_text(json.dumps(candidate))
        homography = hybrid.load_ir_homography(path)  # raises SafetyBlocked on any format mismatch

        test_px = cih.GUIDE_POINTS[0]
        expected_xy = _A @ np.array(test_px, dtype=float) + _B
        got_xy = hybrid.ir_pixel_to_xy(test_px, homography)
        error_mm = float(np.linalg.norm(np.array(got_xy) - expected_xy)) * 1000.0
        assert error_mm < 1.0, f"hybrid 로더를 거친 왕복 오차가 너무 큼: {error_mm:.3f} mm"


if __name__ == "__main__":
    failures = 0
    for test in (test_round_trip_synthetic_points, test_compatible_with_hybrid_loader):
        try:
            print(f"[{test.__name__}]")
            test()
            print("  PASS")
        except Exception as e:  # noqa: BLE001 - a dry-run harness, report and continue
            failures += 1
            print(f"  FAIL: {type(e).__name__}: {e}")
    print("\n" + ("FAIL" if failures else "PASS") + f" ({failures} failing)")
    sys.exit(1 if failures else 0)
