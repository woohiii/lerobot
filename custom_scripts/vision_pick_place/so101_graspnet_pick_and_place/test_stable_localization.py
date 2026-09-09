from __future__ import annotations

import numpy as np
import pytest

from .models import CameraIntrinsics, ObjectRoi, RgbdFrame
from .stable_localization import localize_stable_roi


def _frame(depth_mm: int) -> RgbdFrame:
    return RgbdFrame(
        rgb=np.zeros((10, 10, 3), dtype=np.uint8),
        depth=np.full((10, 10), depth_mm, dtype=np.uint16),
        intrinsics=CameraIntrinsics(100, 100, 0, 0),
    )


def test_localize_stable_roi_transforms_three_consistent_depth_frames() -> None:
    frames = [_frame(600), _frame(601), _frame(599)]
    roi = ObjectRoi(1, 1, 5, 5)
    transform = np.eye(4)
    transform[:3, 3] = (0.002, -0.004, 0.0)

    pose = localize_stable_roi(
        frames, roi, transform, min_valid_depth_ratio=0.5, max_spread_m=0.005
    )

    assert pose.camera_xyz == pytest.approx((0.015, 0.015, 0.6), abs=0.002)
    assert pose.base_xyz == pytest.approx((0.017, 0.011, 0.6), abs=0.002)
    assert pose.max_spread_m < 0.005


def test_localize_stable_roi_rejects_depth_jitter() -> None:
    frames = [_frame(600), _frame(610), _frame(590)]

    with pytest.raises(ValueError, match="unstable"):
        localize_stable_roi(
            frames, ObjectRoi(1, 1, 5, 5), np.eye(4), min_valid_depth_ratio=0.5, max_spread_m=0.005
        )


def test_localize_stable_roi_requires_registered_three_frame_capture() -> None:
    frames = [_frame(600), _frame(600)]

    with pytest.raises(ValueError, match="exactly three"):
        localize_stable_roi(
            frames, ObjectRoi(1, 1, 5, 5), np.eye(4), min_valid_depth_ratio=0.5, max_spread_m=0.005
        )
