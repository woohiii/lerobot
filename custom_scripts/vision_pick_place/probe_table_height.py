"""One-off diagnostic: TABLE_Z=0.045 was reached with zero contact detected
across multiple real runs (good xy centering, ~4mm), which either means
TABLE_Z doesn't actually reach the physical table, or contact detection
itself isn't triggering for a light touch. This probes for the real answer:
moves to a safe height above a given xy, then descends in small steps
(move_z, stall_check=True) until CollisionDetected fires or a hard floor is
reached, printing the z at contact (or the fact that none was found).

Run: uv run python3 custom_scripts/vision_pick_place/probe_table_height.py [x] [y]
Defaults to the xy from the most recent real descend attempt.
"""

import sys
import time

from robot_control import CollisionDetected, RobotController

PORT = "/dev/ttyACM0"
SAFE_START_Z = 0.10
STEP_M = 0.004
HARD_FLOOR_Z = -0.03  # don't go lower than this no matter what - safety backstop


def main():
    x = float(sys.argv[1]) if len(sys.argv) > 1 else 0.214
    y = float(sys.argv[2]) if len(sys.argv) > 2 else -0.010

    rc = RobotController(port=PORT)
    rc.connect()
    try:
        print(f"[probe] 시작 위치 ({x}, {y}, {SAFE_START_Z})로 이동합니다.")
        rc.move_to_xyz_converge((x, y, SAFE_START_Z), tolerance_m=0.01, max_iters=15)
        cur = rc.current_gripper_xyz()
        print(f"[probe] 도달: {cur}")

        z = cur[2]
        contact_z = None
        while z > HARD_FLOOR_Z:
            try:
                cur = rc.move_z(-STEP_M, steps=8, step_delay_s=0.05)
                z = cur[2]
                print(f"[probe] z={z:.4f} - 접촉 없음")
            except CollisionDetected as e:
                cur = rc.current_gripper_xyz()
                contact_z = cur[2]
                print(f"[probe] 접촉 감지! z={contact_z:.4f} ({e})")
                break
            time.sleep(0.1)

        if contact_z is None:
            print(f"[probe] HARD_FLOOR_Z={HARD_FLOOR_Z}까지 내려갔는데도 접촉이 전혀 없었습니다.")
        else:
            print(f"\n[결과] 실제 접촉 높이: z={contact_z:.4f}  (현재 TABLE_Z=0.045)")

        print("[probe] 안전 높이로 복귀합니다.")
        cur = rc.current_gripper_xyz()
        rc.move_to_xyz((cur[0], cur[1], SAFE_START_Z), steps=20, step_delay_s=0.05, enforce_cap=False)
    finally:
        rc.disconnect()


if __name__ == "__main__":
    main()
