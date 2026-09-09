from __future__ import annotations

import numpy as np
import pytest

from .local_planner import plan_local_task
from .models import CameraIntrinsics, DetectedObject, RgbdFrame, TaskCommand
from .routing import ArmRoute, WorkspaceBounds
from .stable_localization import StablePose


def _frame() -> RgbdFrame:
    return RgbdFrame(
        rgb=np.zeros((10, 10, 3), dtype=np.uint8),
        depth=np.full((10, 10), 500, dtype=np.uint16),
        intrinsics=CameraIntrinsics(100, 100, 0, 0),
    )


def _command() -> TaskCommand:
    return TaskCommand(
        source=DetectedObject("red block", (1, 1, 5, 5), 0.9, "object"),
        destination=DetectedObject("blue tray", (5, 5, 9, 9), 0.9, "destination"),
    )


def test_plan_local_task_uses_stable_base_poses_and_routes_arm() -> None:
    arms = [ArmRoute("left", WorkspaceBounds(-1, -0.05, -1, 1, 0, 1), (0, 0, 0))]
    source = StablePose(np.array((0.01, 0.01, 0.5)), np.array((-0.19, 0.01, 0.5)), 0.001)
    destination = StablePose(np.array((0.07, 0.01, 0.5)), np.array((-0.13, 0.01, 0.5)), 0.001)

    plan = plan_local_task(
        source, destination, _command(), arms, center_exclusion_half_width=0.02, min_confidence=0.6
    )

    assert plan.arm.name == "left"
    assert plan.source_xyz == pytest.approx((-0.2 + 0.01, 0.01, 0.5))
    assert plan.destination_xyz == pytest.approx((-0.2 + 0.07, 0.01, 0.5))


def test_plan_local_task_rejects_low_confidence_detection() -> None:
    command = TaskCommand(
        source=DetectedObject("block", (1, 1, 5, 5), 0.5),
        destination=DetectedObject("tray", (5, 5, 9, 9), 0.9, "destination"),
    )
    arms = [ArmRoute("left", WorkspaceBounds(-1, -0.05, -1, 1, 0, 1), (0, 0, 0))]

    with pytest.raises(ValueError, match="confidence"):
        plan_local_task(
            StablePose(np.zeros(3), np.zeros(3), 0.0),
            StablePose(np.zeros(3), np.zeros(3), 0.0),
            command,
            arms,
            center_exclusion_half_width=0.02,
            min_confidence=0.6,
        )
