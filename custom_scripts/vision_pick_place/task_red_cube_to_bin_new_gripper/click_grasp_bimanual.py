"""Bimanual click-to-grasp: generalizes click_pick_place.py to both arms of a
bi_so_follower, pick-only (no bin/place step - deliberate scope cut, this is
for grasping whatever the user points at, one arm at a time, and holding it).

Live Astra RGB view stays up the whole session. Left-click anywhere -> queue
a grasp for the LEFT arm at that pixel; right-click -> queue one for the
RIGHT arm. Clicks are handled one at a time, in order; a click during an
in-flight grasp just waits in the queue - video keeps refreshing regardless.
'e' cuts torque on BOTH arms immediately (emergency stop, no scripted motion
after). 'q'/ESC quits normally, both arms return to their recorded home xyz.

The right arm's calibration (config.RIGHT_OVERRIDES: TABLE_Z, wrist-cam grasp
target px, gripper thresholds) is an unverified placeholder copy of the left
arm's until measured - right-side clicks refuse to move by default; pass
--allow-unverified-right once calibrate_grasp.py / probe_table_height_manual.py
/ measure_grasp_target_px.py have been run for the right arm.

Run: uv run python3 custom_scripts/vision_pick_place/task_red_cube_to_bin_new_gripper/click_grasp_bimanual.py \\
    --left-port /dev/so101_follower_left --right-port /dev/so101_follower_right
"""

from __future__ import annotations

import argparse

import cv2

import config
import perception
import task_state_machine as tsm
from kinematics import CollisionDetected, SOArm101
from lerobot.robots.bi_so_follower import BiSOFollower, BiSOFollowerConfig
from lerobot.robots.so_follower import SOFollowerConfig

WINDOW = "left-click=left arm, right-click=right arm - 'e'=e-stop, 'q'=quit"


def handle_click(
    side: str,
    x: int,
    y: int,
    arm_left: SOArm101,
    arm_right: SOArm101,
    allow_unverified_right: bool,
    home_xyz_left,
    home_xyz_right,
) -> None:
    if side == "right" and not allow_unverified_right:
        print(
            "[click_grasp_bimanual] 오른팔 보정값이 아직 미검증 상태입니다 (config.RIGHT_OVERRIDES는 "
            "왼팔 값의 placeholder 복사본 - TABLE_Z, 손목캠 파지 목표 픽셀, 그리퍼 임계값 모두 미측정). "
            "calibrate_grasp.py / probe_table_height_manual.py / measure_grasp_target_px.py로 "
            "오른팔을 먼저 보정한 뒤, --allow-unverified-right 옵션으로 다시 실행하세요."
        )
        return

    config.apply_side(side)
    arm = arm_left if side == "left" else arm_right
    home_xyz = home_xyz_left if side == "left" else home_xyz_right

    xy = perception.pixel_to_xy(x, y)
    if xy is None:
        print("[click_grasp_bimanual] 호모그래피 없음 (homography.json 확인 필요)")
        return
    if not perception.is_xy_within_safe_workspace(*xy):
        print(f"[click_grasp_bimanual] ({xy[0]:.3f}, {xy[1]:.3f})는 보정된 안전 작업영역 밖입니다 - 이동 취소")
        return

    target = (xy[0], xy[1], config.SEARCH_HOVER_XYZ[2])
    print(f"[click_grasp_bimanual] {side} 팔: 클릭 위치로 이동 ({xy[0]:.3f}, {xy[1]:.3f})")
    try:
        arm.move_to_xyz_converge(target, tolerance_m=0.015, max_iters=20)
    except CollisionDetected as e:
        print(f"[click_grasp_bimanual] {side} 팔: 이동 중 충돌 감지 ({e}) - 취소")
        return

    # No generic per-object detector for an arbitrary clicked object, so this
    # skips fine_servo's wrist-cam closed-loop refinement entirely and goes
    # straight to a blind descend - deliberate scope cut, not an oversight.
    grasped = tsm.descend_and_grasp(arm)
    if grasped:
        print(f"[click_grasp_bimanual] {side} 팔: 파지 성공 - 상승 후 제자리 유지")
        arm.move_z(config.LIFT_M, steps=20, step_delay_s=0.05)
        return

    print(f"[click_grasp_bimanual] {side} 팔: 파지 실패 - 홈으로 복귀합니다")
    try:
        arm.move_to_xyz_converge(home_xyz, tolerance_m=0.015, max_iters=20)
    except CollisionDetected as e:
        print(f"[click_grasp_bimanual] {side} 팔: 홈 복귀 중 충돌 감지: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-port", required=True, help="왼쪽 팔 follower 포트")
    parser.add_argument("--right-port", required=True, help="오른쪽 팔 follower 포트")
    parser.add_argument("--allow-unverified-right", action="store_true",
                         help="오른팔 보정값이 미검증 placeholder여도 오른팔 클릭을 허용")
    args = parser.parse_args()

    # BiSOFollower derives each sub-arm's calibration id as f"{id}_left"/f"{id}_right"
    # from THIS id (see bi_so_follower.py) - plain SOFollowerConfig itself has no
    # `id` field. "follower" here reproduces the existing "follower_left"/
    # "follower_right" calibration files already saved under
    # ~/.cache/huggingface/lerobot/calibration/robots/so_follower/.
    bi_config = BiSOFollowerConfig(
        id="follower",
        left_arm_config=SOFollowerConfig(port=args.left_port, max_relative_target=config.MAX_RELATIVE_TARGET_DEG),
        right_arm_config=SOFollowerConfig(port=args.right_port, max_relative_target=config.MAX_RELATIVE_TARGET_DEG),
    )
    bi = BiSOFollower(bi_config)
    bi.connect(calibrate=False)
    print("[click_grasp_bimanual] 양팔 연결 성공")

    arm_left = SOArm101(robot=bi.left_arm)
    arm_right = SOArm101(robot=bi.right_arm)
    arm_left.connect()
    arm_right.connect()

    home_left = arm_left.get_joint_deg()
    home_xyz_left = tuple(arm_left.kin.forward_kinematics(home_left[: len(config.ARM_JOINTS)])[:3, 3])
    home_right = arm_right.get_joint_deg()
    home_xyz_right = tuple(arm_right.kin.forward_kinematics(home_right[: len(config.ARM_JOINTS)])[:3, 3])

    click_queue: list[tuple[str, int, int]] = []

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            click_queue.append(("left", x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            click_queue.append(("right", x, y))

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)
    cap = perception.PublishedFrameSource(config.ASTRA_RGB_FRAME_PATH)

    # ponytail: `return` inside a try still runs `finally` in Python - an
    # emergency stop must NOT be followed by the finally block's home-return
    # move, so track it explicitly and gate the home-return on this flag
    # rather than trying to short-circuit out of the try/finally.
    emergency_stopped = False
    try:
        while True:
            ok, frame = cap.read()
            if ok:
                cv2.imshow(WINDOW, frame)
            # Popping and handling a click here blocks the video feed for the
            # duration of the grasp attempt - same already-blocking design as
            # click_pick_place.py, accepted tradeoff, not fixed here.
            if click_queue:
                side, x, y = click_queue.pop(0)
                handle_click(side, x, y, arm_left, arm_right, args.allow_unverified_right,
                             home_xyz_left, home_xyz_right)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("e"):
                arm_left.release_torque()
                arm_right.release_torque()
                print("[click_grasp_bimanual] 비상 정지 - 양팔 토크 해제됨")
                emergency_stopped = True
                break
            if key in (ord("q"), 27):
                break
    finally:
        cv2.destroyAllWindows()
        if emergency_stopped:
            print("[click_grasp_bimanual] 비상 정지 상태 유지 - 홈 복귀 시도하지 않음 (양팔 토크 해제된 채로 둠)")
        else:
            print(f"[click_grasp_bimanual] 왼팔 홈 복귀: {home_xyz_left}")
            try:
                arm_left.move_to_xyz_converge(home_xyz_left, tolerance_m=0.015, max_iters=20)
            except CollisionDetected as e:
                print(f"[click_grasp_bimanual] 왼팔 홈 복귀 중 충돌 감지: {e}")
            except Exception as e:
                print(f"[click_grasp_bimanual] 왼팔 홈 복귀 실패: {e}")
            print(f"[click_grasp_bimanual] 오른팔 홈 복귀: {home_xyz_right}")
            try:
                arm_right.move_to_xyz_converge(home_xyz_right, tolerance_m=0.015, max_iters=20)
            except CollisionDetected as e:
                print(f"[click_grasp_bimanual] 오른팔 홈 복귀 중 충돌 감지: {e}")
            except Exception as e:
                print(f"[click_grasp_bimanual] 오른팔 홈 복귀 실패: {e}")
        bi.disconnect()


if __name__ == "__main__":
    main()
