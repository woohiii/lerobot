"""Click-driven pick-and-place - per the user's 2026-09-01 request, after a
real run showed search()'s blind SEARCH_OFFSETS grid sweep moving the arm
unpredictably into large joint-lag collisions ("자유분방하게 움직여").
Replaces only the "where do I even start looking" step with the user
pointing at the cube/bin in the Astra RGB view; everything downstream is
unchanged - reuses task_state_machine.py's own fine_servo(skip_search=True)
and descend_and_grasp exactly, same IK move, same wrist-cam closed-loop
refinement, same release logic as the autonomous main.py.

Flow: shows the live Astra RGB view (red-cube/black-bin detections boxed for
reference, if any). Click near the cube -> IK move there via the table
homography, wrist-cam fine-servo, descend, grasp, lift. Then click the bin ->
same, then open the gripper. 'q'/ESC on either click prompt cancels and
returns home.

Needs camera_hub.py AND astra_s_live.py already running and publishing (see
main.py's docstring - camera devices aren't opened here, two processes can't
both hold one).

Run: uv run python3 custom_scripts/vision_pick_place/task_red_cube_to_bin_new_gripper/click_pick_place.py
"""

from __future__ import annotations

import sys
import time

import cv2

import config
import gripper
import perception
import task_state_machine as tsm
from kinematics import CollisionDetected, SOArm101

WINDOW = "click the cube, then the bin - 'q' to cancel"


def _draw_boxes(frame):
    display = frame.copy()
    cube = perception.detect_red_cube(frame)
    if cube is not None:
        x, y, w, h = cube.bbox
        cv2.rectangle(display, (x, y), (x + w, y + h), (0, 0, 255), 2)
    bin_det = perception.detect_black_bin(frame)
    if bin_det is not None:
        x, y, w, h = bin_det.bbox
        cv2.rectangle(display, (x, y), (x + w, y + h), (255, 255, 0), 2)
    return display


def wait_for_click(prompt: str) -> tuple[int, int] | None:
    """Live Astra view, refreshed every tick, until the user clicks a point
    or quits. None on quit."""
    clicked: list[tuple[int, int]] = []

    def on_click(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked.append((x, y))

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_click)
    print(f"[click_pick_place] {prompt}")
    cap = perception.PublishedFrameSource(config.ASTRA_RGB_FRAME_PATH)
    try:
        while True:
            ok, frame = cap.read()
            if ok:
                cv2.imshow(WINDOW, _draw_boxes(frame))
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                return None
            if clicked:
                return clicked[-1]
    finally:
        cv2.destroyWindow(WINDOW)


def pick(arm: SOArm101, wrist_cap) -> bool:
    pt = wait_for_click("빨간 큐브를 클릭하세요 (q=취소)")
    if pt is None:
        return False
    xy = perception.pixel_to_xy(*pt)
    if xy is None:
        print("[click_pick_place] 호모그래피 없음 (homography.json 확인 필요)")
        return False
    target = (xy[0], xy[1], config.SEARCH_HOVER_XYZ[2])
    print(f"[click_pick_place] 큐브 추정 위치로 이동: ({xy[0]:.3f}, {xy[1]:.3f})")
    try:
        arm.move_to_xyz_converge(target, tolerance_m=0.015, max_iters=20)
    except CollisionDetected as e:
        print(f"[click_pick_place] 이동 중 충돌 감지 ({e}) - 취소")
        return False
    if not tsm.fine_servo(arm, wrist_cap, perception.detect_red_cube, "빨간 큐브", skip_search=True):
        print("[click_pick_place] 큐브 정렬 실패")
        return False
    return tsm.descend_and_grasp(arm)


def place(arm: SOArm101, wrist_cap) -> bool:
    pt = wait_for_click("검은 쓰레기통을 클릭하세요 (q=취소)")
    if pt is None:
        return False
    xy = perception.pixel_to_xy(*pt)
    if xy is None:
        print("[click_pick_place] 호모그래피 없음")
        return False
    hover = (xy[0], xy[1], config.SEARCH_HOVER_XYZ[2])
    print(f"[click_pick_place] 쓰레기통 추정 위치로 이동: ({xy[0]:.3f}, {xy[1]:.3f})")
    try:
        arm.move_to_xyz_converge(hover, tolerance_m=0.015, max_iters=20)
    except CollisionDetected as e:
        print(f"[click_pick_place] 이동 중 충돌 감지 ({e}) - 현재 위치에서 계속")
    if not tsm.fine_servo(arm, wrist_cap, perception.detect_black_bin, "검은 쓰레기통", skip_search=True):
        print("[click_pick_place] 쓰레기통 정렬 실패 - 현재 위치에서 내려놓습니다")

    try:
        arm.move_z(-config.BIN_DESCEND_M, steps=15, step_delay_s=0.05)
    except CollisionDetected:
        print("[click_pick_place] 하강 중 접촉 감지 (쓰레기통 벽/바닥) - 현재 위치에서 놓음")
    gripper.open_gripper(arm)
    time.sleep(0.3)
    arm.move_z(config.BIN_DESCEND_M, steps=15, step_delay_s=0.05)
    return True


def main() -> bool:
    wrist_cap = perception.PublishedFrameSource(config.WRIST_FRAME_PATH)
    if not wrist_cap.isOpened():
        print(
            "[click_pick_place] 손목캠 프레임이 없습니다. 먼저 camera_hub.py를 실행해주세요:\n"
            "  source ~/lerobot_song_venv/bin/activate && "
            "python custom_scripts/vision_pick_place/camera_hub.py"
        )
        return False
    astra_cap = perception.PublishedFrameSource(config.ASTRA_RGB_FRAME_PATH)
    if not astra_cap.isOpened():
        print(
            "[click_pick_place] Astra 프레임이 없습니다. 먼저 astra_s_live.py를 실행해주세요:\n"
            "  source ~/lerobot_song_venv/bin/activate && "
            "python custom_scripts/vision_pick_place/astra_s_live.py"
        )
        return False

    arm = SOArm101()
    arm.connect()
    print("[click_pick_place] 연결 성공. 현재 관절각:", arm.get_joint_deg())
    home_pose = arm.get_joint_deg()
    home_xyz = tuple(arm.kin.forward_kinematics(home_pose[: len(config.ARM_JOINTS)])[:3, 3])

    try:
        gripper.open_gripper(arm)
        if not pick(arm, wrist_cap):
            print("[click_pick_place] 픽업 실패/취소 - 홈으로 복귀합니다.")
            return False
        print("[click_pick_place] 파지 성공 - 상승")
        arm.move_z(config.LIFT_M, steps=20, step_delay_s=0.05)
        if not place(arm, wrist_cap):
            print("[click_pick_place] 놓기 실패/취소 - 큐브를 든 채로 홈 복귀합니다.")
            return False
        print("[click_pick_place] 완료")
        return True
    except KeyboardInterrupt:
        print("\n[click_pick_place] 사용자가 중단했습니다.")
        return False
    finally:
        cv2.destroyAllWindows()
        print(f"[click_pick_place] 홈 포즈로 복귀: {home_xyz}")
        try:
            arm.move_to_xyz_converge(home_xyz, tolerance_m=0.015, max_iters=20)
        except CollisionDetected as e:
            print(f"[click_pick_place] 홈 복귀 중 충돌 감지, 안전 위치에서 정지: {e}")
        except Exception as e:
            print(f"[click_pick_place] 홈 복귀 실패: {e}")
        arm.disconnect()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
