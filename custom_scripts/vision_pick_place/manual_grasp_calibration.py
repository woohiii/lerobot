"""Empirically calibrates GRIPPER_EMPTY_CLOSED_PCT / GRASP_DETECT_MARGIN_PCT
(visual_servo_pick_place.py) from real hand-guided trials, instead of the
placeholder guesses those constants started as.

Flow per trial: cuts torque so the arm can be freely hand-positioned, waits
for Enter once the gripper is placed (by hand) either around the cube or at
an empty spot, re-engages torque *at that exact hand-set pose* (see
RobotController.enable_torque's docstring for why this isn't just flipping
Torque_Enable back on), closes the gripper, and asks whether a cube was
actually there - logging (label, final_pct) pairs.

At the end, prints the actual observed gap between "closed on nothing" and
"closed on the cube" readings and a suggested GRIPPER_EMPTY_CLOSED_PCT /
GRASP_DETECT_MARGIN_PCT, rather than leaving those as guesses.

Run: uv run python3 custom_scripts/vision_pick_place/manual_grasp_calibration.py
(needs the main ~/lerobot venv - same as visual_servo_pick_place.py - for
feetech-servo-sdk/lerobot, not the GUI camera venv. No camera needed at all.)
"""

from __future__ import annotations

from robot_control import RobotController

PORT = "/dev/so101_follower"  # 2026-09-01: reverted - follower's own board is alive again,
# see task_red_cube_to_bin_new_gripper/config.py's FOLLOWER_PORT note


def run_trial(rc: RobotController, trial_num: int) -> tuple[bool, float] | None:
    print(f"\n--- 시도 {trial_num} ---")
    rc.emergency_stop()
    print("토크를 껐습니다 - 팔이 힘없이 움직일 수 있는 상태입니다.")
    print("손으로 그리퍼를 원하는 위치에 놓아주세요 (큐브를 잡을 위치든, 빈 곳이든).")
    try:
        input("위치를 잡으셨으면 Enter를 눌러주세요 (Ctrl+C로 이 시도 취소)... ")
    except KeyboardInterrupt:
        print("\n이 시도를 취소합니다.")
        rc.enable_torque()  # still re-engage holding torque wherever it currently is, for safety
        return None

    rc.enable_torque()
    print("현재 위치에서 토크 재활성화 완료 (팔이 그 자리에서 버팁니다).")
    print("현재 관절각:", rc.get_joint_deg())

    input("그리퍼를 닫습니다 - 준비되면 Enter... ")
    final_pct = rc.set_gripper_pct_converge(0.0)
    print(f"그리퍼 닫음 - 최종 위치: {final_pct:.1f}%")

    while True:
        ans = input("이번 시도에 큐브가 실제로 그리퍼 사이에 있었나요? (y/n): ").strip().lower()
        if ans in ("y", "n"):
            break
        print("y 또는 n으로 답해주세요.")
    has_cube = ans == "y"

    input("다음 시도를 위해 그리퍼를 다시 엽니다 - Enter... ")
    rc.set_gripper_pct_converge(100.0)

    return has_cube, final_pct


def main():
    rc = RobotController(port=PORT)
    rc.connect()
    print("연결 성공. 그립 판정 기준(빈 상태 vs 큐브 문 상태) 캘리브레이션을 시작합니다.")
    print("몇 번 시도하시겠어요? 큐브를 문 경우와 빈 경우를 섞어서 몇 번이든 해보셔도 됩니다.")

    results: list[tuple[bool, float]] = []
    trial_num = 0
    try:
        while True:
            trial_num += 1
            result = run_trial(rc, trial_num)
            if result is not None:
                results.append(result)
            cont = input("\n계속하시겠어요? (Enter=계속, q=종료): ").strip().lower()
            if cont == "q":
                break
    finally:
        rc.disconnect()

    if not results:
        print("\n기록된 시도가 없습니다. 종료합니다.")
        return

    print(f"\n{'=' * 60}\n결과 요약 ({len(results)}개 시도)\n{'=' * 60}")
    empty_pcts = [pct for has_cube, pct in results if not has_cube]
    cube_pcts = [pct for has_cube, pct in results if has_cube]
    for has_cube, pct in results:
        print(f"  {'큐브 있음' if has_cube else '빈 상태  '}: {pct:.1f}%")

    if empty_pcts:
        print(f"\n빈 상태 최종 위치: {empty_pcts} (평균 {sum(empty_pcts)/len(empty_pcts):.1f}%)")
    else:
        print("\n빈 상태 시도가 없어서 기존 기본값(5.0%)을 기준으로 둡니다.")
    if cube_pcts:
        print(f"큐브 있음 최종 위치: {cube_pcts} (평균 {sum(cube_pcts)/len(cube_pcts):.1f}%)")
    else:
        print("큐브 있음 시도가 없어서 마진을 추천할 수 없습니다 - 최소 한 번은 큐브를 물려보세요.")
        return

    empty_baseline = (sum(empty_pcts) / len(empty_pcts)) if empty_pcts else 5.0
    min_cube = min(cube_pcts)
    gap = min_cube - empty_baseline
    if gap <= 0:
        print(
            f"\n[경고] 큐브 있음 최소값({min_cube:.1f}%)이 빈 상태 기준({empty_baseline:.1f}%)보다 "
            f"높지 않습니다 - 이 두 상태가 현재 그리퍼 위치값만으로는 구분되지 않는다는 뜻입니다. "
            f"더 많은 시도로 재확인해보세요."
        )
        return

    suggested_margin = gap * 0.5  # sit halfway between the two observed clusters, not right at the edge
    print(
        f"\n추천 값 (visual_servo_pick_place.py):\n"
        f"  GRIPPER_EMPTY_CLOSED_PCT = {empty_baseline:.1f}\n"
        f"  GRASP_DETECT_MARGIN_PCT = {suggested_margin:.1f}\n"
        f"  (판정 임계값 = {empty_baseline + suggested_margin:.1f}%, 관측된 간격 {empty_baseline:.1f}~{min_cube:.1f}% 중간)"
    )


if __name__ == "__main__":
    main()
