"""Hardware-free tests for the safe sequential pick-and-place executor."""

from __future__ import annotations

import numpy as np
import pytest

from custom_scripts.vision_pick_place.so101_graspnet_pick_and_place.orchestrator import (
    PickAndPlaceExecutor,
    SafetyAbortError,
)
from custom_scripts.vision_pick_place.so101_graspnet_pick_and_place.left_pick_executor import PickPlaceWaypoints


class FakeArm:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[float, float, float] | None]] = []

    def move_to(self, point: tuple[float, float, float]) -> None:
        self.calls.append(("move", point))

    def open_gripper(self) -> None:
        self.calls.append(("open", None))

    def close_gripper(self) -> None:
        self.calls.append(("close", None))

    def retreat(self) -> None:
        self.calls.append(("retreat", None))


def test_executor_places_only_after_dual_grasp_verification() -> None:
    arm = FakeArm()
    executor = PickAndPlaceExecutor(
        arm=arm,
        verify_gripper=lambda: True,
        verify_wrist=lambda: True,
        home=(0.0, 0.0, 0.3),
        bin_pose=(0.2, -0.2, 0.15),
        approach_height_m=0.08,
        lift_height_m=0.10,
    )

    executor.execute(np.array([0.1, 0.05, 0.02]))

    assert arm.calls == [
        ("move", (0.1, 0.05, 0.1)),
        ("open", None),
        ("move", (0.1, 0.05, 0.02)),
        ("close", None),
        ("move", (0.1, 0.05, 0.12)),
        ("move", (0.2, -0.2, 0.15)),
        ("open", None),
        ("retreat", None),
        ("move", (0.0, 0.0, 0.3)),
    ]


def test_executor_retreats_without_placing_when_wrist_verification_fails() -> None:
    arm = FakeArm()
    executor = PickAndPlaceExecutor(
        arm=arm,
        verify_gripper=lambda: True,
        verify_wrist=lambda: False,
        home=(0.0, 0.0, 0.3),
        bin_pose=(0.2, -0.2, 0.15),
        approach_height_m=0.08,
        lift_height_m=0.10,
    )

    with pytest.raises(SafetyAbortError, match="wrist"):
        executor.execute(np.array([0.1, 0.05, 0.02]))

    assert ("move", (0.2, -0.2, 0.15)) not in arm.calls
    assert arm.calls[-2:] == [("retreat", None), ("move", (0.0, 0.0, 0.3))]


def test_executor_retreats_when_arm_motion_raises() -> None:
    class FailingArm(FakeArm):
        def move_to(self, point: tuple[float, float, float]) -> None:
            super().move_to(point)
            raise RuntimeError("motor failure")

    arm = FailingArm()
    executor = PickAndPlaceExecutor(
        arm=arm,
        verify_gripper=lambda: True,
        verify_wrist=lambda: True,
        home=(0.0, 0.0, 0.3),
        bin_pose=(0.2, -0.2, 0.15),
        approach_height_m=0.08,
        lift_height_m=0.10,
    )

    with pytest.raises(SafetyAbortError, match="execution failed"):
        executor.execute(np.array([0.1, 0.05, 0.02]))
    assert arm.calls[-2:] == [("retreat", None), ("move", (0.0, 0.0, 0.3))]


def test_executor_uses_validated_destination_hover_before_release() -> None:
    """Fails if a validated destination hover is skipped before releasing the object."""
    arm = FakeArm()
    executor = PickAndPlaceExecutor(
        arm=arm,
        verify_gripper=lambda: True,
        verify_wrist=lambda: True,
        home=(0.0, 0.0, 0.3),
        bin_pose=(9.0, 9.0, 9.0),
        approach_height_m=0.08,
        lift_height_m=0.10,
    )
    waypoints = PickPlaceWaypoints(
        source_hover=(0.1, 0.0, 0.1),
        source_grasp=(0.1, 0.0, 0.02),
        source_lift=(0.1, 0.0, 0.12),
        destination_hover=(0.2, -0.2, 0.16),
        destination_release=(0.2, -0.2, 0.04),
        retreat=(0.2, -0.2, 0.14),
    )

    executor.execute_validated(waypoints)

    assert arm.calls == [
        ("move", waypoints.source_hover),
        ("open", None),
        ("move", waypoints.source_grasp),
        ("close", None),
        ("move", waypoints.source_lift),
        ("move", waypoints.destination_hover),
        ("move", waypoints.destination_release),
        ("open", None),
        ("retreat", None),
        ("move", (0.0, 0.0, 0.3)),
    ]
