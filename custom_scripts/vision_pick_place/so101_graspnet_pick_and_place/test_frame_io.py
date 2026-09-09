from __future__ import annotations

import numpy as np

from .frame_io import load_rgbd_frame, save_rgbd_snapshot
from .models import CameraIntrinsics, RgbdFrame


def test_save_rgbd_snapshot_round_trips_rgb_depth_and_metadata(tmp_path) -> None:
    frame = RgbdFrame(
        rgb=np.zeros((3, 4, 3), dtype=np.uint8),
        depth=np.full((3, 4), 700, dtype=np.uint16),
        intrinsics=CameraIntrinsics(500, 501, 2, 1),
    )

    directory = save_rgbd_snapshot(frame, tmp_path)
    loaded = load_rgbd_frame(directory)

    assert loaded.rgb.shape == (3, 4, 3)
    assert loaded.depth.dtype == np.uint16
    np.testing.assert_array_equal(loaded.depth, frame.depth)
    assert loaded.intrinsics == frame.intrinsics

