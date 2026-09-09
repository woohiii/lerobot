"""Hardware-neutral, fail-closed SO-101 pick-and-place state executor."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from .left_pick_executor import PickPlaceWaypoints


class Arm(Protocol):
    def move_to(self, point: tuple[float, float, float]) -> None: ...

    def open_gripper(self) -> None: ...

    def close_gripper(self) -> None: ...

    def retreat(self) -> None: ...


class SafetyAbortError(RuntimeError):
    """Raised after a failed grasp check has been retreated safely."""


@dataclass
class PickAndPlaceExecutor:
    """Execute one arm at a time; never place an unverified grasp."""

    arm: Arm
    verify_gripper: Callable[[], bool]
    verify_wrist: Callable[[], bool]
    home: tuple[float, float, float]
    bin_pose: tuple[float, float, float]
    approach_height_m: float
    lift_height_m: float

    def execute(self, grasp_xyz: np.ndarray) -> None:
        target = tuple(float(value) for value in grasp_xyz)
        if len(target) != 3 or not np.isfinite(target).all():
            raise ValueError("grasp_xyz must contain three finite coordinates")
        hover = self._point(target[0], target[1], target[2] + self.approach_height_m)
        lift = self._point(target[0], target[1], target[2] + self.lift_height_m)
        try:
            self.arm.move_to(hover)
            self.arm.open_gripper()
            self.arm.move_to(target)
            self.arm.close_gripper()
            self.arm.move_to(lift)
            if not self.verify_gripper():
                self._abort("gripper verification failed")
            if not self.verify_wrist():
                self._abort("wrist verification failed")
            self.arm.move_to(self.bin_pose)
            self.arm.open_gripper()
            self.arm.retreat()
            self.arm.move_to(self.home)
        except SafetyAbortError:
            raise
        except Exception as exc:
            self._abort(f"execution failed: {exc}")

    def execute_validated(self, waypoints: "PickPlaceWaypoints") -> None:
        """Execute only the Cartesian legs approved by left-arm preflight."""
        try:
            self.arm.move_to(waypoints.source_hover)
            self.arm.open_gripper()
            self.arm.move_to(waypoints.source_grasp)
            self.arm.close_gripper()
            self.arm.move_to(waypoints.source_lift)
            if not self.verify_gripper():
                self._abort("gripper verification failed")
            if not self.verify_wrist():
                self._abort("wrist verification failed")
            self.arm.move_to(waypoints.destination_hover)
            self.arm.move_to(waypoints.destination_release)
            self.arm.open_gripper()
            self.arm.retreat()
            self.arm.move_to(self.home)
        except SafetyAbortError:
            raise
        except Exception as exc:
            self._abort(f"execution failed: {exc}")

    def _abort(self, reason: str) -> None:
        try:
            self.arm.retreat()
        except Exception as exc:
            stop = getattr(self.arm, "stop", None)
            if callable(stop):
                try:
                    stop()
                except Exception:
                    pass
            raise SafetyAbortError(f"{reason}; retreat failed: {exc}") from exc
        try:
            self.arm.move_to(self.home)
        except Exception as exc:
            raise SafetyAbortError(f"{reason}; home return failed: {exc}") from exc
        raise SafetyAbortError(reason)

    @staticmethod
    def _point(x: float, y: float, z: float) -> tuple[float, float, float]:
        return tuple(round(value, 6) for value in (x, y, z))  # type: ignore[return-value]
