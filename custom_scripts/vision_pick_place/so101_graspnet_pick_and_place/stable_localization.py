"""Three-frame validation for registered RGB-D ROI localization."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .geometry import build_roi_point_cloud, transform_points
from .models import ObjectRoi, RgbdFrame


@dataclass(frozen=True)
class StablePose:
    """A camera-space ROI location validated across three registered frames."""

    camera_xyz: np.ndarray
    base_xyz: np.ndarray
    max_spread_m: float


def _roi_center_point(frame: RgbdFrame, roi: ObjectRoi, *, min_valid_ratio: float) -> np.ndarray:
    cloud = build_roi_point_cloud(frame, roi)
    area = (roi.x1 - roi.x0) * (roi.y1 - roi.y0)
    if len(cloud) < area * min_valid_ratio:
        raise ValueError("ROI has insufficient valid depth")
    return np.median(cloud[:, :3], axis=0)


def _validate_registered_frames(frames: Sequence[RgbdFrame]) -> None:
    if len(frames) != 3:
        raise ValueError("stable localization requires exactly three registered RGB-D frames")
    reference = frames[0]
    for frame in frames[1:]:
        if frame.depth.shape != reference.depth.shape or frame.intrinsics != reference.intrinsics:
            raise ValueError("RGB-D frames must have matching dimensions and intrinsics for registration")


def localize_stable_roi(
    frames: Sequence[RgbdFrame],
    roi: ObjectRoi,
    camera_to_base: np.ndarray,
    *,
    min_valid_depth_ratio: float,
    max_spread_m: float,
) -> StablePose:
    """Return a stable ROI median only after validating the registered capture."""

    _validate_registered_frames(frames)
    points = np.stack(
        [_roi_center_point(frame, roi, min_valid_ratio=min_valid_depth_ratio) for frame in frames]
    )
    camera_xyz = np.median(points, axis=0)
    spread = float(np.max(np.linalg.norm(points - camera_xyz, axis=1)))
    if spread > max_spread_m:
        raise ValueError(f"ROI localization is unstable: {spread:.4f} m")
    base_xyz = transform_points(camera_xyz[None, :], camera_to_base)[0]
    return StablePose(camera_xyz=camera_xyz, base_xyz=base_xyz, max_spread_m=spread)
