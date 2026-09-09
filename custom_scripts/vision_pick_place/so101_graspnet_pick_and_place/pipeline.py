"""Safe hand-off from Gemini ROI geometry to a uniquely routed SO-101 arm."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from .geometry import build_roi_point_cloud, transform_points
from .graspnet_adapter import GraspCandidate, filter_candidates
from .models import ObjectRoi, RgbdFrame
from .routing import ArmRoute, route_point


class PlanningError(RuntimeError):
    """Raised when perception cannot produce one safe, reachable grasp."""


@dataclass(frozen=True)
class PlannedGrasp:
    candidate: GraspCandidate
    base_position_xyz: tuple[float, float, float]
    arm: ArmRoute


def plan_grasp(
    frame: RgbdFrame,
    roi: ObjectRoi,
    camera_to_base: np.ndarray,
    infer_candidates: Callable[[np.ndarray], Iterable[GraspCandidate]],
    arms: Sequence[ArmRoute],
    *,
    center_exclusion_half_width: float,
    min_score: float,
) -> PlannedGrasp:
    """Infer candidates from one ROI and return the best safely routed result."""

    cloud = build_roi_point_cloud(frame, roi)
    if len(cloud) == 0:
        raise PlanningError("ROI contains no valid depth points")
    candidates = filter_candidates(infer_candidates(cloud), min_score=min_score)
    if not candidates:
        raise PlanningError("GraspNet returned no valid candidates")
    for candidate in candidates:
        point = transform_points(np.asarray([candidate.position_xyz], dtype=float), camera_to_base)[0]
        try:
            arm = route_point(point, arms, center_exclusion_half_width)
        except ValueError:
            continue
        return PlannedGrasp(candidate, tuple(float(value) for value in point), arm)
    raise PlanningError("no reachable grasp candidate outside the central exclusion zone")
