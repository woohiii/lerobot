"""Hardware-free dry run for il_grasp_skill.run_grasp_skill.

No robot, no camera, no trained checkpoint - a fake engine stands in for the
whole policy pipeline (that pipeline is lerobot's own SyncInferenceEngine and
is tested upstream; what is NOT tested anywhere else is this file's deadline,
its stall check, and its verification contract, which is exactly what this
covers).

Run: uv run python custom_scripts/vision_pick_place/task_red_cube_to_bin_new_gripper/test_il_grasp_skill.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
import gripper  # noqa: E402
import il_grasp_skill  # noqa: E402
from kinematics import CollisionDetected  # noqa: E402

ACTION_KEYS = [f"{j}.pos" for j in config.ALL_JOINTS]
DATASET_FEATURES = {
    "action": {"dtype": "float32", "shape": (len(ACTION_KEYS),), "names": ACTION_KEYS},
    "observation.state": {"dtype": "float32", "shape": (len(ACTION_KEYS),), "names": ACTION_KEYS},
    "observation.images.wrist": {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channels"]},
}


class FakeRobot:
    name = "fake_so101"

    def __init__(self, arm):
        self.arm = arm

    def get_observation(self):
        return {k: float(v) for k, v in zip(ACTION_KEYS, self.arm.pos)}


class FakeArm:
    """Follows every command exactly - no stall."""

    def __init__(self):
        self.pos = np.zeros(len(config.ALL_JOINTS))
        self.sent = []
        self.robot = FakeRobot(self)

    def get_joint_deg(self):
        return self.pos.copy()

    def send_joint_deg(self, joint_deg):
        self.pos = np.asarray(joint_deg, dtype=float).copy()
        self.sent.append(self.pos)


class StuckArm(FakeArm):
    """Commands are accepted but the arm never moves - a real block."""

    def send_joint_deg(self, joint_deg):
        self.sent.append(np.asarray(joint_deg, dtype=float).copy())


class FakeCap:
    def read(self):
        return True, np.zeros((480, 640, 3), dtype=np.uint8)


class FakeEngine:
    """Minimal surface run_grasp_skill actually uses of SyncInferenceEngine."""

    def __init__(self, value: float):
        self.value = value
        self.resets = 0
        self.calls = 0

    def reset(self):
        self.resets += 1

    def get_action(self, obs_frame):
        assert "observation.images.wrist" in obs_frame, "wrist frame missing from the observation"
        assert obs_frame["observation.images.wrist"].shape == (480, 640, 3)
        assert obs_frame["observation.state"].shape == (len(ACTION_KEYS),)
        self.calls += 1
        return [self.value] * len(ACTION_KEYS)


def make_skill(value: float) -> il_grasp_skill.GraspSkill:
    return il_grasp_skill.GraspSkill(
        engine=FakeEngine(value),
        dataset_features=DATASET_FEATURES,
        ordered_action_keys=ACTION_KEYS,
        camera_keys=["wrist"],
    )


class GripperSpy:
    """Replaces gripper.close_gripper / is_grasp_success for the duration of a test."""

    def __init__(self, final_pct: float):
        self.final_pct = final_pct
        self.close_calls = 0
        self.verify_calls = 0

    def __enter__(self):
        self._orig = (gripper.close_gripper, gripper.is_grasp_success)
        gripper.close_gripper = self._close
        gripper.is_grasp_success = self._verify
        return self

    def __exit__(self, *exc):
        gripper.close_gripper, gripper.is_grasp_success = self._orig

    def _close(self, arm):
        self.close_calls += 1
        return self.final_pct

    def _verify(self, final_pct):
        self.verify_calls += 1
        return final_pct > config.GRIPPER_EMPTY_CLOSED_PCT + config.GRASP_DETECT_MARGIN_PCT


def test_deadline_and_verification():
    arm, skill = FakeArm(), make_skill(0.1)
    # fps far above what a real loop runs at, so only the deadline can stop it
    with GripperSpy(final_pct=0.0) as spy:
        t0 = time.perf_counter()
        result = il_grasp_skill.run_grasp_skill(arm, FakeCap(), max_seconds=0.4, fps=1000.0, skill=skill)
        elapsed = time.perf_counter() - t0

    assert elapsed < 0.4 + 0.5, f"loop overran the deadline: {elapsed:.2f}s"
    assert skill.engine.calls > 5, f"expected many ticks inside 0.4s, got {skill.engine.calls}"
    assert len(arm.sent) == skill.engine.calls, "every action should have been sent to the arm"
    assert skill.engine.resets == 1, "engine must be reset once per attempt"
    assert spy.close_calls == 1, f"close_gripper called {spy.close_calls}x, expected exactly 1"
    assert spy.verify_calls == 1, f"is_grasp_success called {spy.verify_calls}x, expected exactly 1"
    assert isinstance(result, bool), f"must return a bool, got {type(result)}"
    assert result is False, "empty-close reading must verify as 'not grasped'"
    print(f"  deadline honored: {elapsed:.2f}s / {skill.engine.calls} ticks, verified once, returned {result}")


def test_success_returns_true():
    arm, skill = FakeArm(), make_skill(0.1)
    wedged = config.GRIPPER_EMPTY_CLOSED_PCT + config.GRASP_DETECT_MARGIN_PCT + 5.0
    with GripperSpy(final_pct=wedged):
        result = il_grasp_skill.run_grasp_skill(arm, FakeCap(), max_seconds=0.15, fps=1000.0, skill=skill)
    assert result is True, "a gripper wedged open on an object must verify as grasped"
    print(f"  wedged gripper ({wedged:.1f}%) -> returned {result}")


def test_stall_raises_collision():
    arm, skill = StuckArm(), make_skill(50.0)  # 50deg command, arm never moves
    with GripperSpy(final_pct=0.0) as spy:
        try:
            il_grasp_skill.run_grasp_skill(arm, FakeCap(), max_seconds=2.0, fps=1000.0, skill=skill)
        except CollisionDetected as e:
            print(f"  stall raised CollisionDetected as expected: {e}")
        else:
            raise AssertionError("a blocked arm must raise CollisionDetected, not finish quietly")
    assert spy.close_calls == 0, "a stall must abort before the grasp verification"


def test_missing_policy_path():
    try:
        il_grasp_skill.run_grasp_skill(FakeArm(), FakeCap())
    except ValueError as e:
        print(f"  missing policy_path rejected: {e}")
    else:
        raise AssertionError("run_grasp_skill without policy_path or skill must raise")


if __name__ == "__main__":
    failures = 0
    for test in (
        test_deadline_and_verification,
        test_success_returns_true,
        test_stall_raises_collision,
        test_missing_policy_path,
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
