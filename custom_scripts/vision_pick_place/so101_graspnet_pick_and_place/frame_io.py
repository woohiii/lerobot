"""RGB-D snapshot persistence for reproducible perception and dry-runs."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from .models import CameraIntrinsics, RgbdFrame


def save_rgbd_snapshot(frame: RgbdFrame, root: str | Path) -> Path:
    root = Path(root)
    directory = root / datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    directory.mkdir(parents=True, exist_ok=False)
    if not cv2.imwrite(str(directory / "rgb.png"), cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)):
        raise OSError("failed to write RGB snapshot")
    np.save(directory / "depth_mm.npy", frame.depth)
    (directory / "frame.json").write_text(
        json.dumps(
            {
                "rgb": "rgb.png",
                "depth": "depth_mm.npy",
                "depth_unit": "mm",
                "intrinsics": frame.intrinsics.__dict__,
            },
            indent=2,
        )
    )
    return directory


def load_rgbd_frame(directory: str | Path) -> RgbdFrame:
    directory = Path(directory)
    metadata = json.loads((directory / "frame.json").read_text())
    bgr = cv2.imread(str(directory / metadata["rgb"]), cv2.IMREAD_COLOR)
    depth = np.load(directory / metadata["depth"])
    if bgr is None:
        raise OSError(f"failed to read RGB snapshot from {directory}")
    return RgbdFrame(
        rgb=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
        depth=np.asarray(depth, dtype=np.uint16),
        intrinsics=CameraIntrinsics(**metadata["intrinsics"]),
    )
