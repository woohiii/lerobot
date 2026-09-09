"""Workspace-based routing for SO-101 pick-and-place arms."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class WorkspaceBounds:
    """Inclusive axis-aligned bounds in the shared base frame."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    def contains(self, point: Sequence[float]) -> bool:
        """Return whether a 3D point lies within all bounds (inclusive)."""
        x, y, z = point
        return (
            self.x_min <= x <= self.x_max
            and self.y_min <= y <= self.y_max
            and self.z_min <= z <= self.z_max
        )


@dataclass(frozen=True)
class ArmRoute:
    """An arm's name, reachable workspace, and destination bin pose."""

    name: str
    workspace: WorkspaceBounds
    bin_pose: tuple[float, ...]
    port: str | None = None


def route_point(
    point: Sequence[float],
    arms: Sequence[ArmRoute],
    center_exclusion_half_width: float,
) -> ArmRoute:
    """Select the unique arm whose workspace contains ``point``.

    The shared center strip is intentionally rejected before workspace lookup so
    that overlapping arm bounds cannot silently decide a potentially unsafe
    handoff.  Workspace boundaries are inclusive.
    """
    if len(point) != 3:
        raise ValueError("point must contain exactly three coordinates")
    if not isfinite(center_exclusion_half_width) or center_exclusion_half_width < 0:
        raise ValueError("center exclusion half-width must be non-negative")

    if abs(point[0]) <= center_exclusion_half_width:
        raise ValueError("point lies in central exclusion zone")

    candidates = [arm for arm in arms if arm.workspace.contains(point)]
    if not candidates:
        raise ValueError("point is outside every arm workspace")
    if len(candidates) > 1:
        names = ", ".join(arm.name for arm in candidates)
        raise ValueError(f"ambiguous workspace overlap: {names}")
    return candidates[0]
