"""3-stage open/close gripper test. Moves toward each target by repeatedly
re-issuing the same target and re-reading the actual position, instead of a
single big interpolated move - lerobot's own max_relative_target (15deg/call)
was capping progress mid-interpolation anyway, so converging by retry is both
simpler and respects that same safety clamp naturally."""

import time

from robot_control import RobotController

WAYPOINTS = [100.0, 0.0, 100.0, 0.0]  # open, close, open, close
TOLERANCE = 3.0
MAX_ITERS = 15


def move_to_pct(rc: RobotController, target: float, label: str) -> None:
    print(f"\n-> {label} (목표 {target}%)")
    for i in range(MAX_ITERS):
        current = rc.get_joint_deg()[-1]
        if abs(current - target) <= TOLERANCE:
            print(f"   도달: {current:.1f}%")
            return
        step_target = current + max(-35.0, min(35.0, target - current))
        rc.set_gripper_pct(step_target, steps=8, step_delay_s=0.04)
        time.sleep(0.15)
        print(f"   iter {i}: {current:.1f}% -> {rc.get_joint_deg()[-1]:.1f}%")
    print(f"   [경고] {MAX_ITERS}회 내 목표 도달 못함, 마지막: {rc.get_joint_deg()[-1]:.1f}%")


def main():
    rc = RobotController()
    rc.connect()
    try:
        print("시작 위치:", rc.get_joint_deg()[-1])
        for i, target in enumerate(WAYPOINTS):
            label = f"{i+1}/{len(WAYPOINTS)}단계"
            move_to_pct(rc, target, label)
            time.sleep(1.0)
        print("\n테스트 완료.")
    finally:
        rc.disconnect()


if __name__ == "__main__":
    main()
