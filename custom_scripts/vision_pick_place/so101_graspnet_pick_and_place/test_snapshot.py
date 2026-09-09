from __future__ import annotations

import numpy as np

from .frame_io import load_rgbd_frame
from .models import CameraIntrinsics
from .snapshot import capture_published_snapshot


def test_capture_published_snapshot_reads_rgb_and_depth_files(tmp_path) -> None:
    import cv2

    rgb_path = tmp_path / "rgb.png"
    depth_path = tmp_path / "depth.npy"
    rgb = np.zeros((4, 5, 3), dtype=np.uint8)
    assert cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    np.save(depth_path, np.full((4, 5), 600, dtype=np.uint16))

    frame = capture_published_snapshot(
        rgb_path, depth_path, CameraIntrinsics(500, 500, 2, 2), tmp_path / "snapshots"
    )

    assert frame.exists()
    assert (frame / "rgb.png").exists()
    assert (frame / "depth_mm.npy").exists()
    assert (frame / "frame.json").exists()


def test_capture_published_snapshot_resizes_registered_depth_to_rgb_grid(tmp_path) -> None:
    import cv2

    rgb_path = tmp_path / "rgb.png"
    depth_path = tmp_path / "depth.npy"
    rgb = np.zeros((4, 6, 3), dtype=np.uint8)
    assert cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    np.save(depth_path, np.full((2, 3), 600, dtype=np.uint16))

    frame = capture_published_snapshot(
        rgb_path, depth_path, CameraIntrinsics(500, 500, 2, 2), tmp_path / "snapshots"
    )

    loaded = load_rgbd_frame(frame)
    assert loaded.depth.shape == (4, 6)
    assert loaded.depth.dtype == np.uint16
    assert np.all(loaded.depth == 600)
