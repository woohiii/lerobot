"""Hardware-free tests for left-arm pick waypoint preflight."""

from __future__ import annotations

import numpy as np
import pytest

from custom_scripts.vision_pick_place.so101_graspnet_pick_and_place.adapters import SO101ArmAdapter
from custom_scripts.vision_pick_place.so101_graspnet_pick_and_place.left_pick_executor import (
    PickPlaceWaypoints,
    _dry_solve_ik,
    build_and_validate_waypoints,
    build_waypoints,
    execute_validated_pick_place,
)
from custom_scripts.vision_pick_place.so101_graspnet_pick_and_place.orchestrator import SafetyAbortError


class _FakeKin:
    def __init__(self, solutions: list[np.ndarray]) -> None:
        self.solutions = iter(solutions)

    def solve_ik(self, current_joint_deg: np.ndarray, target_xyz: tuple[float, float, float]) -> np.ndarray:
        return next(self.solutions)


class _FakeArm:
    def __init__(self, solutions: list[np.ndarray]) -> None:
        self.kin = _FakeKin(solutions)
        self.sent: list[np.ndarray] = []
        self.move_targets: list[tuple[float, float, float]] = []

    def get_joint_deg(self) -> np.ndarray:
        return np.zeros(6)

    def send_joint_deg(self, joints: np.ndarray) -> None:
        self.sent.append(joints)

    def move_to_xyz_converge(self, target: tuple[float, float, float]) -> None:
        self.move_targets.append(target)


class _ExecutionArm:
    def __init__(self, waypoints: PickPlaceWaypoints) -> None:
        self.events: list[str] = []
        self.raise_on: str | None = None
        self.joints = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.sent: list[np.ndarray] = []

    def enable_torque(self) -> None:
        self.events.append("enable_torque")

    def get_joint_deg(self) -> np.ndarray:
        return self.joints.copy()

    def send_joint_deg(self, joints: np.ndarray) -> None:
        self.joints = np.asarray(joints, dtype=float).copy()
        self.sent.append(self.joints.copy())
        if self.joints[-1] in (0.0, 99.0):
            self.events.append("open" if self.joints[-1] == 99.0 else "close")

    def send_arm_joint_deg(self, joints: np.ndarray) -> None:
        target = np.concatenate((np.asarray(joints, dtype=float), self.joints[-1:]))
        self.joints = target
        self.sent.append(target.copy())
        event = {
            1.0: "source_hover", 2.0: "source_grasp", 3.0: "source_lift",
            4.0: "destination_hover", 5.0: "destination_release", 6.0: "retreat",
        }[target[0]]
        self.events.append(event)
        if event == self.raise_on:
            raise RuntimeError("motion failed")

    def move_validated_arm_joint_deg(self, joints: np.ndarray) -> None:
        self.send_arm_joint_deg(joints)

    def hold_current_pose(self) -> None:
        self.sent.append(self.joints.copy())


@pytest.fixture
def execution_waypoints() -> PickPlaceWaypoints:
    return PickPlaceWaypoints(
        source_hover=(0.1, 0.0, 0.1),
        source_grasp=(0.1, 0.0, 0.02),
        source_lift=(0.1, 0.0, 0.12),
        destination_hover=(0.2, -0.2, 0.16),
        destination_release=(0.2, -0.2, 0.04),
        retreat=(0.2, -0.2, 0.14),
        joint_waypoints=(
            (1.0, 1.0, 1.0, 1.0, 1.0),
            (2.0, 2.0, 2.0, 2.0, 2.0),
            (3.0, 3.0, 3.0, 3.0, 3.0),
            (4.0, 4.0, 4.0, 4.0, 4.0),
            (5.0, 5.0, 5.0, 5.0, 5.0),
            (6.0, 6.0, 6.0, 6.0, 6.0),
        ),
    )


def test_execution_uses_prevalidated_waypoint_order(execution_waypoints: PickPlaceWaypoints) -> None:
    """Fails if execution re-plans or changes the preflight-approved state order."""
    arm = _ExecutionArm(execution_waypoints)

    execute_validated_pick_place(arm, execution_waypoints)

    assert arm.events == [
        "enable_torque",
        "source_hover",
        "open",
        "source_grasp",
        "close",
        "source_lift",
        "destination_hover",
        "destination_release",
        "open",
        "retreat",
    ]


def test_execution_failure_retreats_before_reraising(execution_waypoints: PickPlaceWaypoints) -> None:
    """Fails if any command error can escape without the validated retreat command."""
    arm = _ExecutionArm(execution_waypoints)
    arm.raise_on = "source_grasp"

    with pytest.raises(SafetyAbortError):
        execute_validated_pick_place(arm, execution_waypoints)

    assert arm.sent[-1][0] == 2.0
    assert all(target[0] != 6.0 for target in arm.sent)


def test_execution_holds_delayed_mid_leg_feedback_without_reversing() -> None:
    """Fails if an incomplete edge abort commands a historical waypoint instead of current feedback."""
    waypoints = PickPlaceWaypoints(
        *( (0.0, 0.0, 0.0),) * 6,
        joint_waypoints=((5.0,) * 5,) * 6,
    )

    class DelayedFailingArm(_ExecutionArm):
        def move_validated_arm_joint_deg(self, joints: np.ndarray) -> None:
            self.joints[:5] = 2.5
            raise RuntimeError("stall")

    arm = DelayedFailingArm(waypoints)
    with pytest.raises(SafetyAbortError):
        execute_validated_pick_place(arm, waypoints)

    assert arm.sent[-1][:5] == pytest.approx(np.full(5, 2.5))


def test_validated_joint_motion_rejects_a_lagging_motor_before_the_next_leg() -> None:
    """Catches a paced joint primitive that advances despite unchanged motor feedback."""
    from .left_arm import LeftSOArm101

    arm = object.__new__(LeftSOArm101)
    arm.get_joint_deg = lambda: np.zeros(6)  # type: ignore[method-assign]
    arm.send_arm_joint_deg = lambda _target: None  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="stalled"):
        arm.move_validated_arm_joint_deg(np.full(5, 2.0), timeout_s=0.3)


def test_execution_uses_gripper_values_within_so101_percent_range(execution_waypoints: PickPlaceWaypoints) -> None:
    """Fails if opening commands 100 or relies on the arm's 10-degree motion cap."""
    arm = _ExecutionArm(execution_waypoints)

    execute_validated_pick_place(arm, execution_waypoints)

    gripper_targets = [target[-1] for target in arm.sent]
    assert max(gripper_targets) == 99.0
    assert min(gripper_targets) == 0.0


def test_preflight_preserves_the_approved_joint_target_for_each_cartesian_leg() -> None:
    """Fails if execution would need to solve IK again after preflight."""
    arm = _FakeArm([np.full(5, value) for value in range(1, 7)])

    waypoints = build_and_validate_waypoints(
        arm, np.array([0.2, 0.0, 0.01]), np.array([0.14, -0.1, 0.02]),
        table_z=-0.04, hover_height_m=0.03, lift_height_m=0.05, max_joint_delta_deg=10.0,
    )

    np.testing.assert_allclose(waypoints.joint_waypoints, tuple(tuple([value] * 5) for value in range(1, 7)))


def test_waypoints_use_vertical_source_approach_and_destination_release() -> None:
    """Fails if the waypoint builder offsets the source or destination incorrectly."""
    waypoints = build_waypoints(
        np.array([0.20, 0.0, 0.01]),
        np.array([0.14, -0.1, 0.02]),
        hover_height_m=0.03,
        lift_height_m=0.05,
    )

    assert waypoints.source_hover == pytest.approx((0.20, 0.0, 0.04))
    assert waypoints.source_grasp == pytest.approx((0.20, 0.0, 0.01))
    assert waypoints.source_lift == pytest.approx((0.20, 0.0, 0.06))
    assert waypoints.destination_hover == pytest.approx((0.14, -0.1, 0.05))
    assert waypoints.destination_release == pytest.approx((0.14, -0.1, 0.02))
    assert waypoints.retreat == pytest.approx((0.14, -0.1, 0.07))


def test_preflight_rejects_an_ik_step_above_joint_limit() -> None:
    """Fails if sequential IK jumps above the allowed per-joint motion limit."""
    arm = _FakeArm([np.zeros(5), np.full(5, 11.0), np.full(5, 11.0), np.full(5, 11.0), np.full(5, 11.0), np.full(5, 11.0)])

    with pytest.raises(ValueError, match="joint"):
        build_and_validate_waypoints(
            arm,
            np.array([0.20, 0.0, 0.01]),
            np.array([0.14, -0.1, 0.02]),
            table_z=-0.04,
            hover_height_m=0.03,
            lift_height_m=0.05,
            max_joint_delta_deg=10.0,
        )


def test_preflight_rejects_table_penetration_before_solving_ik() -> None:
    """Fails if an unsafe Cartesian waypoint reaches the IK solver."""
    arm = _FakeArm([])

    with pytest.raises(ValueError, match="table"):
        build_and_validate_waypoints(
            arm,
            np.array([0.20, 0.0, -0.05]),
            np.array([0.14, -0.1, 0.02]),
            table_z=-0.04,
            hover_height_m=0.03,
            lift_height_m=0.05,
            max_joint_delta_deg=10.0,
        )


def test_adapter_retreat_commands_the_validated_waypoint() -> None:
    """Fails if retreat is a no-op or commands a waypoint that was not stored."""
    arm = _FakeArm([])
    adapter = SO101ArmAdapter(arm)
    adapter.set_validated_retreat_waypoint((0.14, -0.1, 0.07))

    adapter.retreat()

    assert len(arm.sent) == 0
    assert arm.move_targets == [(0.14, -0.1, 0.07)]


def test_dry_ik_uses_the_preceding_planned_joint_target_as_its_seed(monkeypatch) -> None:
    """Fails if production dry IK re-reads a stale live arm state for each leg."""
    arm = _FakeArm([])
    arm.kin = object()
    seen_seeds: list[np.ndarray] = []

    def solve_left_ik(kin, current: np.ndarray, waypoint: tuple[float, float, float]) -> np.ndarray:
        seen_seeds.append(current.copy())
        return current[:5] + 1.0

    monkeypatch.setattr(
        "custom_scripts.vision_pick_place.so101_graspnet_pick_and_place.left_pick_executor._solve_left_ik",
        solve_left_ik,
    )
    current = np.zeros(6)
    first = _dry_solve_ik(arm, current, (0.2, 0.0, 0.04))
    second = _dry_solve_ik(arm, np.concatenate((first, current[5:])), (0.2, 0.0, 0.01))

    assert seen_seeds == [pytest.approx(np.zeros(6)), pytest.approx(np.array([1, 1, 1, 1, 1, 0]))]
    assert second == pytest.approx(np.full(5, 2.0))
