"""Create reproducible RGB-D snapshots from the existing Astra publisher."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .frame_io import save_rgbd_snapshot
from .models import CameraIntrinsics, RgbdFrame


def capture_published_snapshot(
    rgb_path: str | Path,
    depth_path: str | Path,
    intrinsics: CameraIntrinsics,
    output_root: str | Path,
) -> Path:
    """Read one atomically-published Astra RGB/depth pair and persist it."""
    bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"RGB frame not found: {rgb_path}")
    depth = np.asarray(np.load(depth_path), dtype=np.uint16)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError("published depth frame must be a 2-D array")
    if depth.shape != rgb.shape[:2]:
        # Astra S commonly publishes registered depth at 320x240 and RGB at
        # 640x480. Nearest-neighbour keeps the metric uint16 depth values and
        # preserves the registered pixel geometry without inventing values.
        depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    return save_rgbd_snapshot(RgbdFrame(rgb=rgb, depth=depth, intrinsics=intrinsics), output_root)
