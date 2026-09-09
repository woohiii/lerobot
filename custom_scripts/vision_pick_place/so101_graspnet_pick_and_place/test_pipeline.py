"""Integration tests for the hardware-free perception-to-arm planning path."""

from __future__ import annotations

import numpy as np
import pytest

from custom_scripts.vision_pick_place.so101_graspnet_pick_and_place.graspnet_adapter import GraspCandidate
from custom_scripts.vision_pick_place.so101_graspnet_pick_and_place.models import (
    CameraIntrinsics,
    ObjectRoi,
    RgbdFrame,
)
from custom_scripts.vision_pick_place.so101_graspnet_pick_and_place.pipeline import (
    PlanningError,
    plan_grasp,
)
from custom_scripts.vision_pick_place.so101_graspnet_pick_and_place.routing import ArmRoute, WorkspaceBounds


def _frame() -> RgbdFrame:
    return RgbdFrame(
        rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        depth=np.full((2, 2), 500, dtype=np.uint16),
        intrinsics=CameraIntrinsics(fx=100.0, fy=100.0, cx=0.0, cy=0.0),
    )


def _arms() -> list[ArmRoute]:
    return [
        ArmRoute("left", WorkspaceBounds(-1.0, -0.05, -1.0, 1.0, 0.0, 1.0), (-0.3, 0.0, 0.2)),
        ArmRoute("right", WorkspaceBounds(0.05, 1.0, -1.0, 1.0, 0.0, 1.0), (0.3, 0.0, 0.2)),
    ]


def test_plan_grasp_transforms_best_reachable_candidate_and_routes_arm() -> None:
    candidates = [
        GraspCandidate((0.0, 0.0, 0.5), np.eye(3).tolist(), 0.95),
        GraspCandidate((0.2, 0.0, 0.5), np.eye(3).tolist(), 0.80),
    ]
    camera_to_base = np.eye(4)
    camera_to_base[0, 3] = 0.1

    plan = plan_grasp(
        _frame(),
        ObjectRoi(0, 0, 2, 2),
        camera_to_base,
        lambda _: candidates,
        _arms(),
        center_exclusion_half_width=0.05,
        min_score=0.5,
    )

    assert plan.arm.name == "right"
    assert plan.base_position_xyz == pytest.approx((0.1, 0.0, 0.5))
    assert plan.candidate.score == 0.95


def test_plan_grasp_rejects_when_every_candidate_lands_in_unsafe_zone() -> None:
    candidate = GraspCandidate((0.0, 0.0, 0.5), np.eye(3).tolist(), 0.95)

    with pytest.raises(PlanningError, match="reachable"):
        plan_grasp(
            _frame(),
            ObjectRoi(0, 0, 2, 2),
            np.eye(4),
            lambda _: [candidate],
            _arms(),
            center_exclusion_half_width=0.05,
            min_score=0.5,
        )
