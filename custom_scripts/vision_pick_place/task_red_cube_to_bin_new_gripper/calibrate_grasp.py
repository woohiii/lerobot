"""Calibrate the new gripper's position-only grasp check.

This commands *only the gripper*: keep the arm in a safe, stationary pose.
For each trial, put either nothing or the red cube between the open jaws,
then answer the prompt after it closes.  Take at least three empty and three
cube trials.  The script prints values to copy into ``config.py``.

Run from the repository root::

    uv run python custom_scripts/vision_pick_place/task_red_cube_to_bin_new_gripper/calibrate_grasp.py
"""

from __future__ import annotations

import time

import config
import gripper
from kinematics import SOArm101


def _ask_label() -> bool:
    while True:
        answer = input("큐브를 물고 있나요? (y/n, q=종료): ").strip().lower()
        if answer == "y":
            return True
        if answer == "n":
            return False
        if answer == "q":
            raise KeyboardInterrupt
        print("y, n, q 중 하나를 입력해주세요.")


def main() -> None:
    arm = SOArm101()
    arm.connect()
    results: list[tuple[bool, float]] = []
    try:
        print("[안전] 팔을 움직이지 마세요. 이 도구는 그리퍼만 열고 닫습니다.")
        while True:
            gripper.open_gripper(arm)
            if input("빈 상태 또는 큐브를 턱 사이에 준비한 뒤 Enter (q=종료)... ").strip().lower() == "q":
                break
            has_cube = _ask_label()
            final_pct = gripper.close_gripper(arm)
            print(f"닫힌 뒤 실제 위치: {final_pct:.1f}%")
            results.append((has_cube, final_pct))
            if input("계속하려면 Enter, 종료하려면 q... ").strip().lower() == "q":
                break
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\n캘리브레이션을 종료합니다.")
    finally:
        try:
            gripper.open_gripper(arm)
        finally:
            arm.disconnect()

    empty = [pct for has_cube, pct in results if not has_cube]
    cube = [pct for has_cube, pct in results if has_cube]
    if not empty or not cube:
        print("빈 상태와 큐브 상태를 각각 한 번 이상 측정해야 합니다.")
        return
    empty_max, cube_min = max(empty), min(cube)
    if cube_min <= empty_max:
        print(f"[판정 불가] 빈 상태 최대={empty_max:.1f}%, 큐브 상태 최소={cube_min:.1f}%로 겹칩니다.")
        return
    margin = (cube_min - empty_max) / 2
    print("\nconfig.py에 다음 값을 반영하세요:")
    print(f"GRIPPER_EMPTY_CLOSED_PCT = {empty_max:.1f}")
    print(f"GRASP_DETECT_MARGIN_PCT = {margin:.1f}")
    print(f"# 판정 임계값: {empty_max + margin:.1f}% (빈 최대와 큐브 최소의 중간)")


if __name__ == "__main__":
    main()
