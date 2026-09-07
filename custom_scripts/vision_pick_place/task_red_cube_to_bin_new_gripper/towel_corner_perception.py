"""Pure, table-view towel-corner perception for the IL+IK pre-grasp phase.

This module intentionally only reads an in-memory RGB image.  It does not
open cameras, load robot state, apply the Astra homography, or command a
robot.  The caller should use the fixed external Astra table view (the same
``/tmp/vsp_astra_rgb.png`` convention as :mod:`perception`) so the complete
towel outline is visible; wrist views are too close for this first phase.

The default mask is tuned for the pale pink/orange towel currently seen on a
neutral table.  Tune ``lower_hsv``/``upper_hsv`` for the actual towel colour
before using its output in a grasp trigger.  Pixel coordinates are ``(x, y)``
in the supplied RGB frame, with ``x`` increasing right and ``y`` down.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class TowelCornerDetection:
    """Corners and simple image-plane quality measures for one towel mask."""

    points: tuple[tuple[int, int], ...]
    confidence: float
    contour_area: float
    mask_area_fraction: float


def _order_clockwise(points: np.ndarray) -> tuple[tuple[int, int], ...]:
    """Make four image points stable: clockwise, beginning at top-left."""
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angles)]
    start = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
    ordered = np.roll(ordered, -start, axis=0)
    return tuple((int(round(x)), int(round(y))) for x, y in ordered)


def detect_towel_corners(
    rgb_frame: np.ndarray,
    *,
    lower_hsv: tuple[int, int, int] = (0, 15, 80),
    upper_hsv: tuple[int, int, int] = (35, 255, 255),
    min_area_fraction: float = 0.01,
    kernel_size: int = 5,
) -> TowelCornerDetection | None:
    """Return four contour-derived towel corners plus a 0..1 confidence.

    ``rgb_frame`` must be an ``H x W x 3`` RGB uint8-style image.  The largest
    pink/orange contour is selected, then a convex four-vertex polygon is used
    when available.  A minimum-area rectangle is the deliberate fallback for
    noisy/frayed outlines: it remains easy to inspect and gives four usable
    extreme points, but is less faithful for a folded towel.
    """
    if rgb_frame.ndim != 3 or rgb_frame.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB frame, got shape {rgb_frame.shape}")
    if rgb_frame.size == 0:
        return None
    if not (0 <= lower_hsv[0] <= upper_hsv[0] <= 179 and 0 <= lower_hsv[1] <= upper_hsv[1] <= 255
            and 0 <= lower_hsv[2] <= upper_hsv[2] <= 255):
        raise ValueError("HSV bounds must be ordered OpenCV HSV values")
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")

    rgb_u8 = np.asarray(rgb_frame, dtype=np.uint8)
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array(lower_hsv, dtype=np.uint8), np.array(upper_hsv, dtype=np.uint8))
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    frame_area = float(rgb_frame.shape[0] * rgb_frame.shape[1])
    contour = max(contours, key=cv2.contourArea)
    contour_area = float(cv2.contourArea(contour))
    if contour_area < frame_area * min_area_fraction:
        return None

    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    if hull_area <= 0:
        return None
    perimeter = cv2.arcLength(hull, True)
    polygon = None
    # A small epsilon preserves perspective-skewed rectangular towel edges;
    # increasing it only when necessary suppresses small cloth-edge noise.
    for epsilon_fraction in (0.01, 0.02, 0.03, 0.04, 0.05):
        candidate = cv2.approxPolyDP(hull, epsilon_fraction * perimeter, True)
        if len(candidate) == 4 and cv2.isContourConvex(candidate):
            polygon = candidate.reshape(4, 2).astype(float)
            break
    used_rectangle = polygon is None
    if polygon is None:
        polygon = cv2.boxPoints(cv2.minAreaRect(hull)).astype(float)

    solidity = min(1.0, contour_area / hull_area)
    rectangle_penalty = 0.15 if used_rectangle else 0.0
    confidence = float(np.clip(solidity - rectangle_penalty, 0.0, 1.0))
    return TowelCornerDetection(
        points=_order_clockwise(polygon),
        confidence=confidence,
        contour_area=contour_area,
        mask_area_fraction=contour_area / frame_area,
    )


def find_towel_corners(rgb_frame: np.ndarray, **kwargs) -> list[tuple[int, int]]:
    """Return candidate ``(x, y)`` corner pixels, or an empty list if absent."""
    detection = detect_towel_corners(rgb_frame, **kwargs)
    return list(detection.points) if detection is not None else []


def _synthetic_towel_frame() -> tuple[np.ndarray, np.ndarray]:
    """Return an RGB neutral-table image and its known skewed towel corners."""
    image = np.full((360, 480, 3), (92, 92, 92), dtype=np.uint8)
    expected = np.array([(92, 80), (382, 58), (407, 282), (68, 305)], dtype=np.int32)
    cv2.fillConvexPoly(image, expected, color=(225, 125, 35))  # orange RGB towel
    return image, expected


def self_check() -> None:
    image, expected = _synthetic_towel_frame()
    detection = detect_towel_corners(image)
    assert detection is not None, "synthetic towel was not segmented"
    assert len(detection.points) == 4, detection.points
    got = np.asarray(detection.points)
    # Both sets use stable clockwise/top-left ordering, but nearest-neighbor
    # matching keeps this check about geometry instead of ordering details.
    nearest = np.min(np.linalg.norm(got[:, None, :] - expected[None, :, :], axis=2), axis=0)
    assert float(nearest.max()) <= 12.0, (detection.points, expected.tolist(), nearest.tolist())
    assert detection.confidence > 0.8, detection
    print(f"PASS: detected 4 synthetic towel corners within {nearest.max():.1f}px: {detection.points}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-check", action="store_true", help="run the in-memory synthetic contour test")
    parser.add_argument("--image", type=Path, help="read-only external Astra/table-view image to inspect")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return 0
    if args.image is None:
        parser.error("pass --self-check or --image PATH")

    bgr = cv2.imread(str(args.image))
    if bgr is None:
        parser.error(f"could not read image: {args.image}")
    detection = detect_towel_corners(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    if detection is None:
        print("no towel-sized saturated contour found")
    else:
        print(f"corners={list(detection.points)} confidence={detection.confidence:.2f} "
              f"area_fraction={detection.mask_area_fraction:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
