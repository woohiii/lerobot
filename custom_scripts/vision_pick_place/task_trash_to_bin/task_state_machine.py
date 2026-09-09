"""Single-step grasp logic: DESCEND -> GRASP -> VERIFY. No SEARCH/APPROACH/
FINE_SERVO states - see this module's 2026-09-08 trim note below.

Every state transition is logged, same convention as the sibling task.

2026-09-08: trimmed copy of ../task_red_cube_to_bin_new_gripper/task_state_machine.py.
Kept ONLY `descend_and_grasp()` (and its `_log` helper) - `search`, `coarse_center`,
`estimate_jacobian`, `fine_servo`, `get_pixel`, and the color-detect_fn-based `run()`
sequence are all deleted: they exist to align the wrist camera on a KNOWN-COLOR
object (red cube / black bin), and there is no generic detector for an
arbitrary clicked trash object. click_grasp_trash.py's IK move to the clicked
pixel's homography position already IS the "standardized pre-grasp pose" this
task uses - descend_and_grasp(click_px=...) takes it from there via Astra
depth-delta height estimation + contact-detected descent, same mechanism the
sibling task's descend_and_grasp already validated on real hardware.
"""

from __future__ import annotations

import time

import config
import gripper
import perception
import enum
from kinematics import CollisionDetected, SOArm101


class TaskState(enum.Enum):
    HOME_CAPTURED = enum.auto()
    DESCEND = enum.auto()
    GRASP = enum.auto()
    VERIFY = enum.auto()
    LIFT = enum.auto()
    TRANSPORT = enum.auto()
    RELEASE = enum.auto()
    HOME = enum.auto()
    FAILED = enum.auto()
    DONE = enum.auto()


def _log(state: TaskState, msg: str = "") -> None:
    print(f"[{state.name}]{' ' + msg if msg else ''}")


def descend_and_grasp(arm: SOArm101, click_px: tuple[float, float]) -> bool:
    """click_px is required here (unlike the sibling task's version) - this
    task has no color-detector fallback to estimate height without one."""
    cur = arm.gripper_xyz()
    object_height_m = perception.estimate_height_at_px_m(*click_px)
    if object_height_m is not None:
        target_z = min(config.TABLE_Z + object_height_m - config.DESCEND_MARGIN_M, cur[2])
        _log(TaskState.DESCEND, f"Astra 높이 추정 {object_height_m*1000:.1f}mm -> 1차 목표 z={target_z:.4f}")
    else:
        target_z = config.TABLE_Z
        _log(TaskState.DESCEND, "높이 추정 실패 - TABLE_Z로 하강")

    contacted = False
    try:
        arm.move_to_xyz((cur[0], cur[1], target_z), steps=25, step_delay_s=0.05, enforce_cap=False, stall_check=True)
        _log(TaskState.DESCEND, "1차 목표 도달 (접촉 없음)")
    except CollisionDetected:
        _log(TaskState.DESCEND, "접촉 감지 (물체로 판단)")
        contacted = True

    # A depth estimate can undershoot - contact detection, not the estimate,
    # is what actually decides "found it". Keep easing down to TABLE_Z
    # (the measured real table contact point + margin) if phase 1 found
    # nothing, instead of accepting "reached the estimate, nothing there".
    if not contacted and target_z > config.TABLE_Z + 1e-4:
        cur2 = arm.gripper_xyz()
        try:
            arm.move_to_xyz((cur2[0], cur2[1], config.TABLE_Z), steps=25, step_delay_s=0.06, enforce_cap=False, stall_check=True)
            _log(TaskState.DESCEND, "TABLE_Z까지 도달 (접촉 없음)")
        except CollisionDetected:
            _log(TaskState.DESCEND, "2차 하강 중 접촉 감지 (물체로 판단)")
            contacted = True

    time.sleep(0.2)
    _log(TaskState.GRASP, "그리퍼 닫는 중")
    final_pct = gripper.close_gripper(arm)
    time.sleep(0.3)
    grasped = gripper.is_grasp_success(final_pct)
    _log(TaskState.VERIFY, f"최종 그리퍼 위치 {final_pct:.1f}% -> {'집힘' if grasped else '못 집음'}")
    return grasped
