"""Safely re-measure TABLE_Z for the replacement gripper.

The jaw-tip height can change even though the arm and IK are unchanged.  Put
the probe point over clear, empty tabletop and keep all objects away.  It
approaches from 10 cm above the table, descends in 4 mm increments, stops on
the wrapper's joint-lag collision detection, and returns to the safe height.

Run from the repository root::

    uv run python custom_scripts/vision_pick_place/task_red_cube_to_bin_new_gripper/probe_table_height.py 0.214 -0.010
"""

from __future__ import annotations

import sys
import time

import config
from kinematics import CollisionDetected, SOArm101

SAFE_START_Z = 0.10
STEP_M = 0.004
HARD_FLOOR_Z = -0.03


def main() -> None:
    x = float(sys.argv[1]) if len(sys.argv) > 1 else 0.214
    y = float(sys.argv[2]) if len(sys.argv) > 2 else -0.010
    if input(f"비어 있는 테이블 좌표 ({x:.3f}, {y:.3f})를 확인한 뒤 Enter (q=취소)... ").strip().lower() == "q":
        return

    arm = SOArm101()
    arm.connect()
    try:
        arm.move_to_xyz_converge((x, y, SAFE_START_Z), tolerance_m=0.01, max_iters=15)
        while arm.gripper_xyz()[2] > HARD_FLOOR_Z:
            try:
                current = arm.move_z(-STEP_M, steps=8, step_delay_s=0.05)
                print(f"[probe] z={current[2]:.4f}: 접촉 없음")
            except CollisionDetected as exc:
                contact_z = arm.gripper_xyz()[2]
                print(f"[probe] 접촉 감지: z={contact_z:.4f} ({exc})")
                print(f"config.py 권장값: TABLE_Z = {contact_z + 0.003:.4f}")
                return
            time.sleep(0.1)
        print(f"[중단] 안전 하한 z={HARD_FLOOR_Z:.3f}까지 접촉이 감지되지 않았습니다.")
    finally:
        try:
            current = arm.gripper_xyz()
            arm.move_to_xyz((current[0], current[1], SAFE_START_Z), steps=20, step_delay_s=0.05, enforce_cap=False)
        finally:
            arm.disconnect()


if __name__ == "__main__":
    main()
