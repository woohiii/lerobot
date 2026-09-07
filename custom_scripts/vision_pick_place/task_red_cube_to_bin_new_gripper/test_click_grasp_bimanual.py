"""Hardware-free dry run for click_grasp_bimanual.handle_click's place/
release step (Phase 1 of the trash->bin hybrid MVP plan): after a successful
grasp+lift, the arm must move to config.BIN_POSE_XYZ and open the gripper
when that's been measured, or hold the object in place (no bin move, stay
closed) while it's still the unmeasured-placeholder None - never guess a
bin position.

Reuses sim_dry_run.py's FakeSOArm101 (implements exactly the SOArm101
surface this code calls) and monkeypatches perception.pixel_to_xy/
is_xy_within_safe_workspace + task_state_machine.descend_and_grasp, since
none of those are what this test is about (homography lookup and the
descend/grasp servo loop are covered elsewhere, by click_pick_place.py's own
usage and task_state_machine's existing tests respectively). Also verifies
the IL strategy delegates to il_grasp_skill.run_grasp_skill without loading a
real policy.

Run: uv run python3 custom_scripts/vision_pick_place/task_red_cube_to_bin_new_gripper/test_click_grasp_bimanual.py
"""

from __future__ import annotations

import sys

import click_grasp_bimanual as cgb
import config
import perception
import task_state_machine as tsm
from sim_dry_run import FakeSOArm101

FIXED_XY = (0.20, 0.01)
HOME_XYZ = (0.10, 0.0, 0.05)
BIN_XYZ = (0.05, 0.20, 0.10)

# Homography lookup and the actual descend/grasp servo loop aren't under
# test here - only what happens after a grasp already succeeded.
perception.pixel_to_xy = lambda x, y: FIXED_XY
perception.is_xy_within_safe_workspace = lambda x, y: True
DESCEND_CALLS = []
tsm.descend_and_grasp = lambda arm, click_px=None: DESCEND_CALLS.append((arm, click_px)) or True


def _run_right(bin_pose) -> FakeSOArm101:
    # handle_click() calls config.apply_side("right") as its first line,
    # which repopulates the module globals FROM config.RIGHT_DEFAULTS - so
    # the override has to land in that dict, not on the live config.* name.
    # Uses "right" (the already-calibrated side, no unverified-guard) rather
    # than "left" purely so this test doesn't need allow_unverified_left=True
    # to reach the code under test - the place/release logic itself doesn't
    # care which side it runs on.
    config.RIGHT_DEFAULTS["BIN_POSE_XYZ"] = bin_pose
    arm_left = FakeSOArm101(HOME_XYZ, object_present=True)
    arm_right = FakeSOArm101(HOME_XYZ, object_present=True)
    cgb.handle_click("right", 0, 0, arm_left, arm_right, False, HOME_XYZ, HOME_XYZ)
    return arm_right


def test_holds_when_bin_pose_unmeasured():
    arm = _run_right(bin_pose=None)
    expected_xyz = (FIXED_XY[0], FIXED_XY[1], config.SEARCH_HOVER_XYZ[2] + config.LIFT_M)
    got = tuple(round(v, 6) for v in arm.gripper_xyz())
    assert got == tuple(round(v, 6) for v in expected_xyz), f"상승 위치에서 벗어남: {got} != {expected_xyz}"
    assert arm.get_joint_deg()[-1] > config.GRIPPER_EMPTY_CLOSED_PCT, "BIN_POSE_XYZ 미측정인데 물체를 놓아버렸다"


def test_places_and_returns_home_when_bin_pose_measured():
    arm = _run_right(bin_pose=BIN_XYZ)
    got = tuple(round(v, 6) for v in arm.gripper_xyz())
    assert got == tuple(round(v, 6) for v in HOME_XYZ), f"놓기 후 홈으로 복귀하지 않음: {got}"
    assert arm.get_joint_deg()[-1] > config.GRIPPER_EMPTY_CLOSED_PCT, "놓기(open_gripper) 후에도 닫힌 상태"


def test_legacy_passes_click_pixels_to_descend():
    DESCEND_CALLS.clear()
    config.RIGHT_DEFAULTS["BIN_POSE_XYZ"] = None
    arm_left = FakeSOArm101(HOME_XYZ, object_present=True)
    arm_right = FakeSOArm101(HOME_XYZ, object_present=True)
    cgb.handle_click("right", 123, 234, arm_left, arm_right, False, HOME_XYZ, HOME_XYZ)
    assert DESCEND_CALLS == [(arm_right, (123, 234))], f"클릭 좌표가 descend에 전달되지 않음: {DESCEND_CALLS}"


def test_il_strategy_runs_grasp_skill():
    import il_grasp_skill

    calls = []
    original_run_grasp_skill = il_grasp_skill.run_grasp_skill
    il_grasp_skill.run_grasp_skill = lambda arm, cap, policy_path: calls.append((arm, cap, policy_path)) or True
    try:
        cap = object()
        config.RIGHT_DEFAULTS["BIN_POSE_XYZ"] = None
        arm_left = FakeSOArm101(HOME_XYZ, object_present=True)
        arm_right = FakeSOArm101(HOME_XYZ, object_present=True)
        cgb.handle_click(
            "right", 0, 0, arm_left, arm_right, False, HOME_XYZ, HOME_XYZ,
            grasp_strategy="il", policy_path="fake-checkpoint", cap=cap,
        )
        assert calls == [(arm_right, cap, "fake-checkpoint")], "IL 파지 스킬이 정확한 인자로 호출되지 않음"
    finally:
        il_grasp_skill.run_grasp_skill = original_run_grasp_skill


if __name__ == "__main__":
    failures = 0
    for test in (test_holds_when_bin_pose_unmeasured, test_places_and_returns_home_when_bin_pose_measured,
                 test_legacy_passes_click_pixels_to_descend, test_il_strategy_runs_grasp_skill):
        try:
            print(f"[{test.__name__}]")
            test()
            print("  PASS")
        except Exception as e:  # noqa: BLE001 - a dry-run harness, report and continue
            failures += 1
            print(f"  FAIL: {type(e).__name__}: {e}")
    print("\n" + ("FAIL" if failures else "PASS") + f" ({failures} failing)")
    sys.exit(1 if failures else 0)
