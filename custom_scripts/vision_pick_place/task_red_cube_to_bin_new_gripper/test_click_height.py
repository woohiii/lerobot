"""Hardware-free unit tests for click-based Astra height estimation.

Run: uv run python3 custom_scripts/vision_pick_place/task_red_cube_to_bin_new_gripper/test_click_height.py
"""

from __future__ import annotations

import os
import sys
import tempfile

import cv2
import numpy as np

import config
import perception
import task_state_machine as tsm


def test_estimate_height_at_px():
    """An arbitrary-color clicked patch yields its known depth delta."""
    color = np.full((480, 640, 3), 200, dtype=np.uint8)
    depth_mm = np.full((240, 320), 1000, dtype=np.uint16)
    known_height_mm = 30
    depth_mm[110:130, 150:170] = 1000 - known_height_mm

    with tempfile.TemporaryDirectory() as tmpdir:
        rgb_path = os.path.join(tmpdir, "rgb.png")
        depth_path = os.path.join(tmpdir, "depth_mm.npy")
        assert cv2.imwrite(rgb_path, color)
        np.save(depth_path, depth_mm)
        height = perception.estimate_height_at_px_m(320, 240, rgb_path=rgb_path, depth_path=depth_path)

    assert height is not None, "클릭 패치 높이 추정이 None을 반환함"
    assert abs(height - known_height_mm / 1000.0) < 0.003, f"추정치 {height}가 기대값과 다름"


class _FakeArm:
    def __init__(self):
        self.xyz = np.array((0.1, 0.0, 0.1), dtype=float)

    def gripper_xyz(self):
        return self.xyz.copy()

    def move_to_xyz(self, xyz, **_kwargs):
        self.xyz = np.array(xyz, dtype=float)


def test_descend_without_click_uses_legacy_estimator():
    """Omitting click_px must retain the red-cube estimate call path."""
    calls = []
    original_cube = perception.estimate_cube_height_m
    original_click = perception.estimate_height_at_px_m
    original_close = tsm.gripper.close_gripper
    original_success = tsm.gripper.is_grasp_success
    original_sleep = tsm.time.sleep
    perception.estimate_cube_height_m = lambda: calls.append("cube") or None
    perception.estimate_height_at_px_m = lambda *_args: (_ for _ in ()).throw(AssertionError("click estimator called"))
    tsm.gripper.close_gripper = lambda _arm: 0.0
    tsm.gripper.is_grasp_success = lambda _pct: False
    tsm.time.sleep = lambda *_args: None
    try:
        tsm.descend_and_grasp(_FakeArm())
    finally:
        perception.estimate_cube_height_m = original_cube
        perception.estimate_height_at_px_m = original_click
        tsm.gripper.close_gripper = original_close
        tsm.gripper.is_grasp_success = original_success
        tsm.time.sleep = original_sleep
    assert calls == ["cube"], f"기존 색 기반 경로가 아닌 호출: {calls}"


if __name__ == "__main__":
    failures = 0
    for test in (test_estimate_height_at_px, test_descend_without_click_uses_legacy_estimator):
        try:
            print(f"[{test.__name__}]")
            test()
            print("  PASS")
        except Exception as e:  # noqa: BLE001 - dry-run harness, report and continue
            failures += 1
            print(f"  FAIL: {type(e).__name__}: {e}")
    print("\n" + ("FAIL" if failures else "PASS") + f" ({failures} failing)")
    sys.exit(1 if failures else 0)
