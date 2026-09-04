"""Offline check for gripper.set_pct_converge's stall-bailout (see config.py's
GRIPPER_STALL_EPS_PCT comment) - no real hardware, a fake arm whose position
stops responding to commands past a fixed point (simulating a real jam).
Asserts it bails out in GRIPPER_STALL_CONSECUTIVE iterations, not max_iters.

Run: uv run python3 custom_scripts/vision_pick_place/task_red_cube_to_bin_new_gripper/test_gripper_stall.py
"""

from __future__ import annotations

import numpy as np

import config
import gripper


class FakeArm:
    """Jaw jams at `jam_pct` - send_joint_deg past that point is a no-op on
    the gripper channel, same as a real object physically blocking it."""

    def __init__(self, start_pct: float, jam_pct: float | None = None):
        self.pct = start_pct
        self.jam_pct = jam_pct
        self.send_count = 0

    def get_joint_deg(self) -> np.ndarray:
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, self.pct])

    def send_joint_deg(self, joint_deg: np.ndarray) -> None:
        self.send_count += 1
        target = float(joint_deg[-1])
        if self.jam_pct is not None and target < self.jam_pct:
            self.pct = self.jam_pct  # can't move past the jam
        else:
            self.pct = target


def test_bails_out_on_jam():
    arm = FakeArm(start_pct=100.0, jam_pct=40.0)  # jams well short of fully closed (0%)
    result = gripper.set_pct_converge(arm, pct=0.0, max_iters=15, steps=2, step_delay_s=0.0)
    assert abs(result - 40.0) < 1.0, f"expected to settle near the jam (40%), got {result}"
    # each set_pct_converge iteration issues `steps` send_joint_deg calls (2 here);
    # bailing out after GRIPPER_STALL_CONSECUTIVE+1 iterations means far fewer
    # than max_iters=15 * 2 = 30 sends.
    max_possible_sends = 15 * 2
    assert arm.send_count < max_possible_sends, (
        f"did not bail early: {arm.send_count} sends (max_iters would allow {max_possible_sends})"
    )
    print(f"jam test: settled at {result:.1f}% after {arm.send_count} sends (bailed early, not {max_possible_sends})")


def test_reaches_real_target():
    arm = FakeArm(start_pct=100.0, jam_pct=None)  # nothing in the way
    result = gripper.set_pct_converge(arm, pct=0.0, max_iters=15, steps=2, step_delay_s=0.0)
    assert abs(result - 0.0) <= 3.0, f"expected to reach the real target (0%), got {result}"
    print(f"no-jam test: reached {result:.1f}% (target 0%)")


if __name__ == "__main__":
    test_bails_out_on_jam()
    test_reaches_real_target()
    print("PASS")
