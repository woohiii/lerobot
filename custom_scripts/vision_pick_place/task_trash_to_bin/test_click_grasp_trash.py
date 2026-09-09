"""Hardware-free dry run for click_grasp_trash.handle_click.

Run: uv run python3 custom_scripts/vision_pick_place/task_trash_to_bin/test_click_grasp_trash.py
"""

from __future__ import annotations

import sys

import numpy as np

import click_grasp_trash as cgt
import config
import il_grasp_skill
import perception
import task_state_machine as tsm

FIXED_XY = (0.20, 0.01)
HOME_XYZ = (0.10, 0.0, 0.05)
BIN_XYZ = (0.05, 0.20, 0.10)


class FakeSOArm101:
    """Minimal SOArm101 surface used by handle_click() and gripper.py."""

    def __init__(self, start_xyz, object_present=True):
        self._xyz = np.array(start_xyz, dtype=float)
        self._joint = np.zeros(len(config.ALL_JOINTS))
        self._joint[-1] = 100.0
        self._object_present = object_present

    def gripper_xyz(self):
        return self._xyz.copy()

    def get_joint_deg(self):
        return self._joint.copy()

    def send_joint_deg(self, joint_deg):
        pct = float(joint_deg[-1])
        if pct <= config.GRIPPER_EMPTY_CLOSED_PCT:
            pct = 24.0 if self._object_present else config.GRIPPER_EMPTY_CLOSED_PCT
        self._joint = np.array(joint_deg, dtype=float)
        self._joint[-1] = pct

    def move_to_xyz(self, xyz, steps=20, step_delay_s=0.05, enforce_cap=True, stall_check=True):
        self._xyz = np.array(xyz, dtype=float)

    def move_to_xyz_converge(self, xyz, tolerance_m=0.005, max_iters=15):
        self.move_to_xyz(xyz)
        return self.gripper_xyz()

    def move_z(self, dz, steps=10, step_delay_s=0.04, stall_check=True):
        self._xyz[2] += dz
        return self.gripper_xyz()

    def release_torque(self):
        pass


perception.pixel_to_xy = lambda x, y: FIXED_XY
perception.is_xy_within_safe_workspace = lambda x, y: True
DESCEND_CALLS = []
tsm.descend_and_grasp = lambda arm, click_px: DESCEND_CALLS.append((arm, click_px)) or True


def _run_legacy(bin_pose) -> FakeSOArm101:
    config.BIN_POSE_XYZ = bin_pose
    arm = FakeSOArm101(HOME_XYZ, object_present=True)
    cgt.handle_click(0, 0, arm, HOME_XYZ)
    return arm


def test_dry_hover_returns_home_without_descend():
    calls = []
    original_descend = tsm.descend_and_grasp
    tsm.descend_and_grasp = lambda *args, **kwargs: calls.append((args, kwargs)) or True
    try:
        arm = FakeSOArm101(HOME_XYZ)
        cgt.handle_click(0, 0, arm, HOME_XYZ, dry_hover=True)
        assert tuple(arm.gripper_xyz()) == HOME_XYZ, "dry-hover 뒤 홈으로 복귀하지 않음"
        assert not calls, "dry-hover에서 descend_and_grasp가 호출됨"
    finally:
        tsm.descend_and_grasp = original_descend


def test_holds_when_bin_pose_unmeasured():
    arm = _run_legacy(bin_pose=None)
    expected_xyz = (FIXED_XY[0], FIXED_XY[1], config.SEARCH_HOVER_XYZ[2] + config.LIFT_M)
    assert tuple(round(v, 6) for v in arm.gripper_xyz()) == tuple(round(v, 6) for v in expected_xyz)
    assert arm.get_joint_deg()[-1] > config.GRIPPER_EMPTY_CLOSED_PCT, "BIN_POSE_XYZ 미측정인데 물체를 놓아버렸다"


def test_places_and_returns_home_when_bin_pose_measured():
    arm = _run_legacy(bin_pose=BIN_XYZ)
    assert tuple(round(v, 6) for v in arm.gripper_xyz()) == HOME_XYZ, "놓기 후 홈으로 복귀하지 않음"
    assert arm.get_joint_deg()[-1] > config.GRIPPER_EMPTY_CLOSED_PCT, "놓기(open_gripper) 후에도 닫힌 상태"


def test_legacy_passes_click_pixels_to_descend():
    DESCEND_CALLS.clear()
    config.BIN_POSE_XYZ = None
    arm = FakeSOArm101(HOME_XYZ, object_present=True)
    cgt.handle_click(123, 234, arm, HOME_XYZ)
    assert DESCEND_CALLS == [(arm, (123, 234))], f"클릭 좌표가 descend에 전달되지 않음: {DESCEND_CALLS}"


def test_il_strategy_runs_grasp_skill():
    calls = []
    original_run_grasp_skill = il_grasp_skill.run_grasp_skill
    il_grasp_skill.run_grasp_skill = lambda arm, cap, policy_path: calls.append((arm, cap, policy_path)) or True
    try:
        cap = object()
        config.BIN_POSE_XYZ = None
        arm = FakeSOArm101(HOME_XYZ, object_present=True)
        cgt.handle_click(0, 0, arm, HOME_XYZ, grasp_strategy="il", policy_path="fake-checkpoint", cap=cap)
        assert calls == [(arm, cap, "fake-checkpoint")], "IL 파지 스킬이 정확한 인자로 호출되지 않음"
    finally:
        il_grasp_skill.run_grasp_skill = original_run_grasp_skill


if __name__ == "__main__":
    failures = 0
    for test in (
        test_dry_hover_returns_home_without_descend,
        test_holds_when_bin_pose_unmeasured,
        test_places_and_returns_home_when_bin_pose_measured,
        test_legacy_passes_click_pixels_to_descend,
        test_il_strategy_runs_grasp_skill,
    ):
        try:
            print(f"[{test.__name__}]")
            test()
            print("  PASS")
        except Exception as e:  # noqa: BLE001 - a dry-run harness, report and continue
            failures += 1
            print(f"  FAIL: {type(e).__name__}: {e}")
    print("\n" + ("FAIL" if failures else "PASS") + f" ({failures} failing)")
    sys.exit(1 if failures else 0)
