"""Small, dependency-light data contracts for the RGB-D geometry pipeline."""

from dataclasses import dataclass
from collections.abc import Callable
from math import isfinite

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole camera intrinsics in pixel units."""

    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self) -> None:
        if not all(isfinite(value) for value in (self.fx, self.fy, self.cx, self.cy)):
            raise ValueError("camera intrinsics must be finite")
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("fx and fy must be positive")


@dataclass(frozen=True)
class ObjectRoi:
    """Image-space bounding box, with x1/y1 exclusive."""

    x0: int
    y0: int
    x1: int
    y1: int


@dataclass(frozen=True)
class DetectedObject:
    """An object or destination region returned by the local vision model."""

    label: str
    box_xyxy: tuple[int, int, int, int]
    confidence: float
    kind: str = "object"

    @property
    def roi(self) -> ObjectRoi:
        return ObjectRoi(*self.box_xyxy)


@dataclass(frozen=True)
class TaskCommand:
    """Resolved natural-language task against one captured scene."""

    source: DetectedObject
    destination: DetectedObject
    arm: str | None = None


@dataclass(frozen=True)
class KoreanTaskQuery:
    """Source and destination phrases extracted from a Korean task command."""

    source_query: str
    destination_query: str


@dataclass(frozen=True)
class RgbdFrame:
    """A color image and aligned uint16 depth image from the same frame."""

    rgb: np.ndarray
    depth: np.ndarray
    intrinsics: CameraIntrinsics | None
    # Optional SDK-produced XYZ (metres), aligned with rgb/depth.  Astra frames
    # use this instead of guessed pinhole intrinsics.
    xyz_camera: np.ndarray | None = None
    xyz_provider: Callable[[int, int, int], tuple[float, float, float]] | None = None
    capture_id: int | None = None

    def __post_init__(self) -> None:
        if self.rgb.dtype != np.uint8 or self.rgb.ndim != 3 or self.rgb.shape[-1] != 3:
            raise ValueError("rgb must be an HxWx3 uint8 array")
        if self.depth.dtype != np.uint16 or self.depth.ndim != 2:
            raise ValueError("depth must be an HxW uint16 array")
        if self.rgb.shape[:2] != self.depth.shape:
            raise ValueError("rgb and depth dimensions must match")
        if self.xyz_camera is not None and (
            self.xyz_camera.shape != (*self.depth.shape, 3)
            or not np.isfinite(self.xyz_camera[self.depth > 0]).all()
        ):
            raise ValueError("xyz_camera must be an aligned finite HxWx3 array")
