"""Preflight construction for a safe, left-arm pick-and-place trajectory."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .left_arm import LeftSOArm101


@dataclass(frozen=True)
class PickPlaceWaypoints:
    source_hover: tuple[float, float, float]
    source_grasp: tuple[float, float, float]
    source_lift: tuple[float, float, float]
    destination_hover: tuple[float, float, float]
    destination_release: tuple[float, float, float]
    retreat: tuple[float, float, float]
    joint_waypoints: tuple[tuple[float, ...], ...] = ()


def build_waypoints(
    source_xyz: np.ndarray,
    destination_xyz: np.ndarray,
    *,
    hover_height_m: float,
    lift_height_m: float,
) -> PickPlaceWaypoints:
    """Build vertical approach, lift, release, and retreat Cartesian waypoints."""
    source = _point(source_xyz, "source_xyz")
    destination = _point(destination_xyz, "destination_xyz")
    _positive_height(hover_height_m, "hover_height_m")
    _positive_height(lift_height_m, "lift_height_m")
    return PickPlaceWaypoints(
        source_hover=(source[0], source[1], source[2] + hover_height_m),
        source_grasp=source,
        source_lift=(source[0], source[1], source[2] + lift_height_m),
        destination_hover=(destination[0], destination[1], destination[2] + hover_height_m),
        destination_release=destination,
        retreat=(destination[0], destination[1], destination[2] + lift_height_m),
    )


def build_and_validate_waypoints(
    arm: "LeftSOArm101",
    source_xyz: np.ndarray,
    destination_xyz: np.ndarray,
    *,
    table_z: float,
    hover_height_m: float,
    lift_height_m: float,
    max_joint_delta_deg: float,
) -> PickPlaceWaypoints:
    """Dry-solve each sequential left-arm leg and reject unsafe trajectories."""
    if not np.isfinite(table_z):
        raise ValueError("table_z must be finite")
    _positive_height(max_joint_delta_deg, "max_joint_delta_deg")
    waypoints = build_waypoints(
        source_xyz,
        destination_xyz,
        hover_height_m=hover_height_m,
        lift_height_m=lift_height_m,
    )
    current = np.asarray(arm.get_joint_deg(), dtype=float)
    if current.ndim != 1 or current.size < 6 or not np.isfinite(current).all():
        raise ValueError("current joint state must be finite")
    ordered_waypoints = _ordered_waypoints(waypoints)
    joint_waypoints: list[tuple[float, ...]] = []
    for waypoint in ordered_waypoints:
        if waypoint[2] < table_z:
            raise ValueError(f"waypoint penetrates table: z={waypoint[2]:.6f} < {table_z:.6f}")
    for waypoint in ordered_waypoints:
        target = _dry_solve_ik(arm, current, waypoint)
        if target.ndim != 1 or target.size == 0 or not np.isfinite(target).all():
            raise ValueError("IK returned non-finite joint values")
        if target.size > current.size:
            raise ValueError("IK returned more joints than the arm state")
        delta = np.abs(target - current[: target.size])
        if float(np.max(delta)) > max_joint_delta_deg:
            raise ValueError("joint delta exceeds max_joint_delta_deg")
        current = np.concatenate((target, current[target.size :]))
        joint_waypoints.append(tuple(float(value) for value in target))
    return PickPlaceWaypoints(**{**waypoints.__dict__, "joint_waypoints": tuple(joint_waypoints)})


def execute_validated_pick_place(arm: "LeftSOArm101", waypoints: PickPlaceWaypoints) -> None:
    """Execute one already-approved route once, preserving holding torque on abort."""
    from .orchestrator import SafetyAbortError

    _require_validated_joint_waypoints(waypoints)
    try:
        arm.enable_torque()
        _send_joint_waypoint(arm, waypoints.joint_waypoints[0])
        _set_gripper(arm, 99.0)
        _send_joint_waypoint(arm, waypoints.joint_waypoints[1])
        _set_gripper(arm, 0.0, allow_contact=True)
        _send_joint_waypoint(arm, waypoints.joint_waypoints[2])
        _require_verified_grasp(arm)
        _send_joint_waypoint(arm, waypoints.joint_waypoints[3])
        _send_joint_waypoint(arm, waypoints.joint_waypoints[4])
        _set_gripper(arm, 99.0)
        _send_joint_waypoint(arm, waypoints.joint_waypoints[5])
    except SafetyAbortError:
        raise
    except Exception as exc:
        try:
            arm.hold_current_pose()
        except Exception as hold_exc:
            raise SafetyAbortError(f"execution failed: {exc}; current-pose hold failed: {hold_exc}") from exc
        raise SafetyAbortError(f"execution failed: {exc}") from exc


def _send_joint_waypoint(arm: "LeftSOArm101", joint_waypoint: tuple[float, ...]) -> None:
    """Send the exact joint target approved by preflight; never solve IK during execution."""
    arm.move_validated_arm_joint_deg(np.asarray(joint_waypoint, dtype=float))


def _set_gripper(arm: "LeftSOArm101", value: float, *, allow_contact: bool = False) -> None:
    if value not in (0.0, 99.0):
        raise ValueError("SO-101 gripper commands must use its 0-99 percent range")
    initial = _finite_joint_feedback(arm)[-1]
    stalled = 0
    for _ in range(60):
        joints = _finite_joint_feedback(arm)
        delta = value - joints[-1]
        if abs(delta) <= 0.5:
            return
        joints[-1] += float(np.clip(delta, -10.0, 10.0))
        arm.send_joint_deg(joints)
        time.sleep(0.05)
        actual = _finite_joint_feedback(arm)[-1]
        if abs(actual - joints[-1]) > 2.0:
            stalled += 1
            if stalled >= 3:
                if allow_contact and initial - actual >= 15.0 and actual <= 90.0:
                    return
                raise RuntimeError("gripper stalled before safe acceptance")
        else:
            stalled = 0
    raise RuntimeError("gripper did not settle before deadline")


def _require_validated_joint_waypoints(waypoints: PickPlaceWaypoints) -> None:
    if len(waypoints.joint_waypoints) != 6:
        raise ValueError("execution requires six prevalidated joint waypoints")
    for target in waypoints.joint_waypoints:
        if len(target) != 5 or not np.isfinite(target).all():
            raise ValueError("prevalidated joint waypoint must contain finite five-axis arm data")


def _require_verified_grasp(arm: "LeftSOArm101") -> None:
    joints = _finite_joint_feedback(arm)
    if joints[-1] >= 95.0:
        raise RuntimeError("conservative grasp check failed: gripper is not closed")


def _finite_joint_feedback(arm: "LeftSOArm101") -> np.ndarray:
    joints = np.asarray(arm.get_joint_deg(), dtype=float)
    if joints.ndim != 1 or joints.size < 6 or not np.isfinite(joints).all():
        raise RuntimeError("joint feedback must be finite six-axis data")
    return joints.copy()


def _dry_solve_ik(arm: Any, current: np.ndarray, waypoint: tuple[float, float, float]) -> np.ndarray:
    solver = getattr(getattr(arm, "kin", None), "solve_ik", None)
    if callable(solver):
        return np.asarray(solver(current, waypoint), dtype=float)
    return np.asarray(_solve_left_ik(arm.kin, current, waypoint), dtype=float)


def _solve_left_ik(kin: Any, current: np.ndarray, waypoint: tuple[float, float, float]) -> np.ndarray:
    """Use the isolated left-arm module's seed-aware IK wrapper."""
    from .left_arm import _kinematics

    return _kinematics.solve_ik(kin, current, waypoint)


def _ordered_waypoints(waypoints: PickPlaceWaypoints) -> tuple[tuple[float, float, float], ...]:
    return (
        waypoints.source_hover,
        waypoints.source_grasp,
        waypoints.source_lift,
        waypoints.destination_hover,
        waypoints.destination_release,
        waypoints.retreat,
    )


def _point(values: np.ndarray, name: str) -> tuple[float, float, float]:
    point = np.asarray(values, dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError(f"{name} must contain three finite coordinates")
    return tuple(float(value) for value in point)  # type: ignore[return-value]


def _positive_height(value: float, name: str) -> None:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
