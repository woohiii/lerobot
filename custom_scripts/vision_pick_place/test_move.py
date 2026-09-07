"""First supervised motion test: lift the gripper straight up 3cm from wherever
it currently is, hold briefly, then return to the exact starting pose. Small,
reversible, and printed in full before anything moves."""

import time

from robot_control import RobotController

LIFT_M = 0.03  # 3cm - small and easy to visually confirm, hard to hurt anything with


def main():
    rc = RobotController()
    rc.connect()
    try:
        start_joints = rc.get_joint_deg()
        start_xyz = rc.current_gripper_xyz()
        print("현재 관절각(deg):", start_joints)
        print("현재 그리퍼 위치(m):", start_xyz)

        target_xyz = tuple(start_xyz + [0, 0, LIFT_M])
        plan = rc.preview_move(target_xyz)
        print(f"\n목표: 현재 위치에서 위로 {LIFT_M*100:.0f}cm ({target_xyz})")
        print("현재 관절각:", plan["current_deg"])
        print("목표 관절각:", plan["target_deg"])
        print("관절별 이동량(deg):", plan["delta_deg"])
        print(f"최대 이동량: {plan['max_abs_delta_deg']:.1f}deg")

        if plan["max_abs_delta_deg"] > 15.0:
            print("\n[중단] 이동량이 예상보다 커서(>15deg) 안전상 중단합니다. IK/캘리브레이션 확인 필요.")
            return

        print("\n3초 후 이동 시작...")
        time.sleep(3)

        print("이동 중 (위로)...")
        rc.move_to_xyz(target_xyz, steps=30, step_delay_s=0.05)
        time.sleep(1.0)

        after_xyz = rc.current_gripper_xyz()
        print("이동 후 그리퍼 위치:", after_xyz)
        print("오차(m):", after_xyz - target_xyz)

        print("\n2초 대기 후 원위치로 복귀...")
        time.sleep(2)
        rc.move_to_xyz(tuple(start_xyz), steps=30, step_delay_s=0.05)
        time.sleep(1.0)

        final_xyz = rc.current_gripper_xyz()
        print("복귀 후 그리퍼 위치:", final_xyz)
        print("시작 위치와의 오차(m):", final_xyz - start_xyz)
        print("\n테스트 완료.")
    finally:
        rc.disconnect()


if __name__ == "__main__":
    main()
