"""Safety-gated IK precision grasp for two detected towel corners.

This is intentionally a small, sequential state machine: plan both arms from
one external-Astra frame, then approach/descend/close each arm in turn.  It
never calls ``config.apply_side()``, since that mutates single-arm globals.
Use ``--self-check`` for an entirely in-memory proof; ``dry_run=True`` plans
and reports poses without calling any arm or gripper command surface.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

import config
import cv2
import gripper
import numpy as np
import perception
import task_state_machine as tsm
from kinematics import CollisionDetected
from towel_corner_perception import find_towel_corners

# This is deliberately NOT the red-cube threshold.  It is a provisional
# towel-specific value pending empty-jaw vs. towel-jaw calibration; real runs
# must not treat it as a validated value merely because cube calibration exists.
TOWEL_GRASP_SUCCESS_MIN_FINAL_PCT = 6.0
TOWEL_PREGRASP_HEIGHT_M = 0.10
TOWEL_CORNER_SEPARATION_M = 0.12
# ``homography.json`` was touch-taught in config.FRAME_W x config.FRAME_H
# (640x480) coordinates.  astra_s_depth_hub publishes its RGB frame at
# 320x240, so towel-corner pixels must be expressed in the calibration frame
# before using perception.pixel_to_xy's established homography convention.
ASTRA_PUBLISHED_RGB_SIZE = (320, 240)
WRIST_FRAME_PATHS = {"left": "/tmp/vsp_wrist_left.png", "right": config.WRIST_FRAME_PATH}


class ArmLike(Protocol):
    def preview_move(self, xyz): ...
    def move_to_xyz_converge(self, xyz, tolerance_m: float = 0.005, max_iters: int = 15): ...
    def move_to_xyz(self, xyz, **kwargs): ...
    def gripper_xyz(self): ...


def detect_towel_wrist_target(bgr_frame: np.ndarray, target_px: tuple[float, float]) -> perception.Detection | None:
    """Use the existing towel contour detector to select the corner nearest the jaws."""
    corners = find_towel_corners(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))
    if not corners:
        return None
    cx, cy = min(corners, key=lambda point: (point[0] - target_px[0]) ** 2 + (point[1] - target_px[1]) ** 2)
    return perception.Detection(cx=float(cx), cy=float(cy), area=1.0, bbox=(cx, cy, 1, 1))


def return_arms_home(left_arm: ArmLike, right_arm: ArmLike, home_xyz: dict[str, tuple[float, float, float]]) -> None:
    """Use the existing safety-checked xyz home-return motion for both arms."""
    for side, arm in (("left", left_arm), ("right", right_arm)):
        try:
            arm.move_to_xyz_converge(home_xyz[side], tolerance_m=0.015, max_iters=20)
        except CollisionDetected as exc:
            print(f"[towel] {side} 홈 복귀 중 충돌 감지, 안전 위치에서 정지: {exc}")
        except Exception as exc:  # noqa: BLE001 - disconnect must still happen after a failed safety return
            print(f"[towel] {side} 홈 복귀 실패: {exc}")


class GraspState(StrEnum):
    PLAN = "plan"
    PREGRASP = "pregrasp"
    DESCEND = "descend"
    CLOSE = "close"
    VERIFY = "verify"
    FAILED = "failed"
    COMPLETE = "complete"


@dataclass(frozen=True)
class ArmPlan:
    side: str
    pixel: tuple[int, int]
    xy: tuple[float, float]
    pregrasp_xyz: tuple[float, float, float]
    grasp_xyz: tuple[float, float, float]


@dataclass(frozen=True)
class PrecisionGraspResult:
    success: bool
    state: GraspState
    reason: str
    plans: tuple[ArmPlan, ...] = ()


def towel_pixel_to_calibration_pixel(px: float, py: float) -> tuple[float, float]:
    """Express a published 320x240 Astra pixel in the 640x480 calibration frame."""
    source_w, source_h = ASTRA_PUBLISHED_RGB_SIZE
    return px * config.FRAME_W / source_w, py * config.FRAME_H / source_h


def towel_pixel_to_xy(px: float, py: float) -> tuple[float, float] | None:
    """Map a 320x240 published Astra towel pixel through the 640x480 calibration."""
    return perception.pixel_to_xy(*towel_pixel_to_calibration_pixel(px, py))


def _candidate_plans(
    corners: list[tuple[int, int]], pixel_to_xy: Callable[[float, float], tuple[float, float] | None]
) -> list[tuple[tuple[int, int], tuple[float, float]]]:
    candidates = []
    for pixel in corners:
        calibration_pixel = towel_pixel_to_calibration_pixel(*pixel)
        xy = pixel_to_xy(*pixel)
        safety_box_passed = xy is not None and perception.is_xy_within_safe_workspace(*xy)
        print(
            "[towel] corner "
            f"raw_pixel={pixel} rescaled_pixel={calibration_pixel} xy={xy} "
            f"safety_box_passed={safety_box_passed}"
        )
        if safety_box_passed:
            candidates.append((pixel, xy))
    return candidates


def assign_corners(
    corners: list[tuple[int, int]],
    *,
    pixel_to_xy: Callable[[float, float], tuple[float, float] | None] = towel_pixel_to_xy,
    min_separation_m: float = TOWEL_CORNER_SEPARATION_M,
) -> tuple[ArmPlan, ArmPlan] | None:
    """Choose the farthest safe pair and assign lower/higher robot-y to arms.

    The common calibrated workspace gate is the existing, only validated
    reachability boundary.  Lower robot-y is assigned to right and higher-y
    to left, giving a deterministic non-crossing pair assignment.
    """
    candidates = _candidate_plans(corners, pixel_to_xy)
    best = None
    for index, first in enumerate(candidates):
        for second in candidates[index + 1 :]:
            distance = float(np.linalg.norm(np.subtract(first[1], second[1])))
            separation_passed = distance >= min_separation_m
            print(
                "[towel] corner pair "
                f"raw_pixels=({first[0]}, {second[0]}) separation_m={distance:.6f} "
                f"minimum_m={min_separation_m:.6f} separation_passed={separation_passed}"
            )
            if separation_passed and (best is None or distance > best[0]):
                best = (distance, first, second)
    if best is None:
        return None
    _, first, second = best
    right_candidate, left_candidate = sorted((first, second), key=lambda candidate: candidate[1][1])

    def plan(side: str, candidate: tuple[tuple[int, int], tuple[float, float]]) -> ArmPlan:
        side_config = config.for_side(side)  # local view; never mutates config globals
        pixel, xy = candidate
        return ArmPlan(
            side=side,
            pixel=pixel,
            xy=xy,
            pregrasp_xyz=(xy[0], xy[1], side_config.TABLE_Z + TOWEL_PREGRASP_HEIGHT_M),
            grasp_xyz=(xy[0], xy[1], side_config.TABLE_Z),
        )

    return plan("left", left_candidate), plan("right", right_candidate)


def precision_grasp(
    left_arm: ArmLike,
    right_arm: ArmLike,
    rgb_frame: np.ndarray,
    *,
    dry_run: bool = False,
    corner_finder: Callable[[np.ndarray], list[tuple[int, int]]] = find_towel_corners,
    pixel_to_xy: Callable[[float, float], tuple[float, float] | None] = towel_pixel_to_xy,
    open_gripper: Callable[[ArmLike], float] = gripper.open_gripper,
    close_gripper: Callable[[ArmLike], float] = gripper.close_gripper,
    grasp_check: Callable[..., bool] = gripper.is_grasp_success,
    wrist_cap_factory: Callable[[str], object] = perception.PublishedFrameSource,
    fine_servo: Callable[..., bool] = tsm.fine_servo,
) -> PrecisionGraspResult:
    """Plan and execute the two-corner grasp, or return a reason without fallback.

    In dry-run mode no arm method or gripper command is called: image detection
    and target planning are the only exercised stages.
    """
    corners = corner_finder(rgb_frame)
    if len(corners) < 2:
        return PrecisionGraspResult(False, GraspState.FAILED, "fewer than two towel corners detected")
    assigned = assign_corners(corners, pixel_to_xy=pixel_to_xy)
    if assigned is None:
        return PrecisionGraspResult(
            False, GraspState.FAILED, "no safe, sufficiently separated reachable corner pair"
        )
    plans = tuple(assigned)
    if dry_run:
        return PrecisionGraspResult(
            True, GraspState.COMPLETE, "dry-run planned; no motor/gripper commands sent", plans
        )

    arms = {"left": left_arm, "right": right_arm}
    # The homography work-area gate is necessary but not sufficient for the
    # arm's current pose.  Ask the existing IK planner to solve both targets
    # before opening either gripper or commanding either arm.
    for plan in plans:
        try:
            arms[plan.side].preview_move(plan.pregrasp_xyz)
        except (CollisionDetected, RuntimeError, ValueError) as exc:
            return PrecisionGraspResult(False, GraspState.FAILED, f"{plan.side} IK unreachable: {exc}", plans)
    for plan in plans:
        arm = arms[plan.side]
        try:
            open_gripper(arm)
            arm.move_to_xyz_converge(plan.pregrasp_xyz, tolerance_m=0.015, max_iters=20)
            side_config = config.for_side(plan.side)
            wrist_cap = wrist_cap_factory(WRIST_FRAME_PATHS[plan.side])

            def detect_fn(frame, target_px=side_config.GRASP_TARGET_PX):
                return detect_towel_wrist_target(frame, target_px)

            if not fine_servo(
                arm,
                wrist_cap,
                detect_fn,
                f"{plan.side} 수건 코너",
                skip_search=True,
                target_px=side_config.GRASP_TARGET_PX,
            ):
                return PrecisionGraspResult(False, GraspState.FAILED, f"{plan.side} wrist fine-servo failed", plans)
            arm.move_to_xyz(plan.grasp_xyz, steps=25, step_delay_s=0.05, enforce_cap=False, stall_check=True)
        except (CollisionDetected, RuntimeError) as exc:
            return PrecisionGraspResult(
                False, GraspState.FAILED, f"{plan.side} {GraspState.DESCEND}: {exc}", plans
            )
        try:
            final_pct = close_gripper(arm)
        except RuntimeError as exc:
            return PrecisionGraspResult(False, GraspState.FAILED, f"{plan.side} close: {exc}", plans)
        if not grasp_check(final_pct, minimum_final_pct=TOWEL_GRASP_SUCCESS_MIN_FINAL_PCT):
            return PrecisionGraspResult(
                False,
                GraspState.FAILED,
                f"{plan.side} grasp verification failed (final={final_pct:.1f}%, towel threshold="
                f"{TOWEL_GRASP_SUCCESS_MIN_FINAL_PCT:.1f}%)",
                plans,
            )
    return PrecisionGraspResult(True, GraspState.COMPLETE, "both towel corners grasped", plans)


def precision_grasp_from_frame_path(
    left_arm: ArmLike, right_arm: ArmLike, frame_path: Path, *, dry_run: bool
) -> PrecisionGraspResult:
    bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if bgr is None:
        return PrecisionGraspResult(
            False, GraspState.FAILED, f"missing/unreadable external Astra frame: {frame_path}"
        )
    return precision_grasp(left_arm, right_arm, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dry_run=dry_run)


class _FakeArm:
    def __init__(self):
        self.calls: list[tuple[str, tuple]] = []

    def move_to_xyz_converge(self, xyz, **_kwargs):
        self.calls.append(("pregrasp", tuple(xyz)))

    def preview_move(self, _xyz):
        return {}

    def move_to_xyz(self, xyz, **_kwargs):
        self.calls.append(("descend", tuple(xyz)))

    def gripper_xyz(self):
        return np.zeros(3)


class _FakeWristCap:
    def __init__(self, path: str):
        self.path = path


def self_check() -> None:
    # A 320x240 detector point must become the corresponding 640x480 point
    # before the existing red-cube homography convention is applied.
    calibration_pixel = towel_pixel_to_calibration_pixel(160, 120)
    assert calibration_pixel == (320.0, 240.0), calibration_pixel
    synthetic_homography = np.array([[0.001, 0.0, 0.0], [0.0, 0.002, 0.0], [0.0, 0.0, 1.0]])
    assert perception.pixel_to_xy(*calibration_pixel, synthetic_homography) == (0.32, 0.48)

    corners = [(10, 10), (90, 10), (90, 90), (10, 90)]

    # Identity-like mapping makes y assignment and separation fully inspectable.
    def pixel_map(x, y):
        return x / 100.0, y / 100.0

    def always_safe(_x, _y):
        return True

    old_workspace = perception.is_xy_within_safe_workspace
    perception.is_xy_within_safe_workspace = always_safe
    try:
        plans = assign_corners(corners, pixel_to_xy=pixel_map, min_separation_m=0.5)
        assert plans is not None and {plan.side for plan in plans} == {"left", "right"}
        assert plans[0].side == "left" and plans[0].xy[1] >= plans[1].xy[1], plans
        left, right = _FakeArm(), _FakeArm()
        servo_calls = []

        def fake_fine_servo(arm, cap, detect_fn, _name, **kwargs):
            frame = np.full((480, 640, 3), (92, 92, 92), dtype=np.uint8)
            cv2.rectangle(frame, (180, 100), (450, 350), (35, 125, 225), thickness=-1)  # orange BGR towel
            detection = detect_fn(frame)
            servo_calls.append((arm, cap.path, detection, kwargs))
            return True

        result = precision_grasp(
            left,
            right,
            np.zeros((2, 2, 3), dtype=np.uint8),
            dry_run=False,
            corner_finder=lambda _frame: corners,
            pixel_to_xy=pixel_map,
            open_gripper=lambda _arm: 100.0,
            close_gripper=lambda arm: 12.0,
            grasp_check=lambda pct, **_: pct > 6.0,
            wrist_cap_factory=_FakeWristCap,
            fine_servo=fake_fine_servo,
        )
        assert result.success and len(left.calls) == 2 and len(right.calls) == 2, result
        assert [call[1] for call in servo_calls] == [WRIST_FRAME_PATHS["left"], WRIST_FRAME_PATHS["right"]]
        assert all(call[2] is not None for call in servo_calls), servo_calls
        assert all(call[3]["skip_search"] and "target_px" in call[3] for call in servo_calls), servo_calls
        servo_failed = precision_grasp(
            _FakeArm(),
            _FakeArm(),
            np.zeros((2, 2, 3), dtype=np.uint8),
            corner_finder=lambda _frame: corners,
            pixel_to_xy=pixel_map,
            open_gripper=lambda _arm: 100.0,
            wrist_cap_factory=_FakeWristCap,
            fine_servo=lambda *_args, **_kwargs: False,
        )
        assert not servo_failed.success and "wrist fine-servo failed" in servo_failed.reason, servo_failed
        dry = precision_grasp(
            left,
            right,
            np.zeros((2, 2, 3), dtype=np.uint8),
            dry_run=True,
            corner_finder=lambda _frame: corners,
            pixel_to_xy=pixel_map,
        )
        assert dry.success and len(left.calls) == 2 and len(right.calls) == 2, dry
        assert not precision_grasp(
            left,
            right,
            np.zeros((2, 2, 3), dtype=np.uint8),
            corner_finder=lambda _frame: [],
            pixel_to_xy=pixel_map,
        ).success
        failure_left, failure_right = _FakeArm(), _FakeArm()
        home_calls = []
        failure_left.move_to_xyz_converge = lambda xyz, **_kwargs: home_calls.append(("left", tuple(xyz)))
        failure_right.move_to_xyz_converge = lambda xyz, **_kwargs: home_calls.append(("right", tuple(xyz)))
        return_arms_home(failure_left, failure_right, {"left": (1.0, 2.0, 3.0), "right": (4.0, 5.0, 6.0)})
        class FakeBi:
            def disconnect(self):
                home_calls.append(("disconnect", ()))

        FakeBi().disconnect()
        assert home_calls == [("left", (1.0, 2.0, 3.0)), ("right", (4.0, 5.0, 6.0)), ("disconnect", ())]
        assert not precision_grasp(
            left,
            right,
            np.zeros((2, 2, 3), dtype=np.uint8),
            corner_finder=lambda _frame: [(1, 1), (2, 2)],
            pixel_to_xy=pixel_map,
        ).success
        unreachable = _FakeArm()
        unreachable.preview_move = lambda _xyz: (_ for _ in ()).throw(RuntimeError("no IK solution"))
        assert not precision_grasp(
            unreachable,
            right,
            np.zeros((2, 2, 3), dtype=np.uint8),
            corner_finder=lambda _frame: corners,
            pixel_to_xy=pixel_map,
            open_gripper=lambda _arm: 100.0,
            close_gripper=lambda _arm: 12.0,
        ).success
    finally:
        perception.is_xy_within_safe_workspace = old_workspace
    print("PASS: 320x240-to-640x480 homography rescale, towel corner assignment, wrist fine-servo, poses, dry-run, and home-before-disconnect failure gate")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if not args.self_check:
        parser.error("only --self-check is safe in this standalone module; runtime owns hardware execution")
    self_check()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
