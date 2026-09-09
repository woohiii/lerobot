"""Depth-backed planning for the local VLM command path."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .models import TaskCommand
from .routing import ArmRoute, route_point
from .stable_localization import StablePose


@dataclass(frozen=True)
class LocalPickPlacePlan:
    command: TaskCommand
    source_xyz: tuple[float, float, float]
    destination_xyz: tuple[float, float, float]
    arm: ArmRoute


def plan_local_task(
    source_pose: StablePose,
    destination_pose: StablePose,
    command: TaskCommand,
    arms: Sequence[ArmRoute],
    *,
    center_exclusion_half_width: float,
    min_confidence: float = 0.6,
) -> LocalPickPlacePlan:
    if command.source.confidence < min_confidence or command.destination.confidence < min_confidence:
        raise ValueError("source and destination confidence are below the configured threshold")
    source = source_pose.base_xyz
    destination = destination_pose.base_xyz
    arm = next((item for item in arms if item.name == command.arm), None) if command.arm else None
    if arm is None:
        arm = route_point(source, arms, center_exclusion_half_width)
    elif not arm.workspace.contains(source):
        raise ValueError(f"source point is outside requested arm workspace: {arm.name}")
    if not arm.workspace.contains(destination):
        raise ValueError(f"destination point is outside selected arm workspace: {arm.name}")
    return LocalPickPlacePlan(
        command=command,
        source_xyz=tuple(float(value) for value in source),
        destination_xyz=tuple(float(value) for value in destination),
        arm=arm,
    )
