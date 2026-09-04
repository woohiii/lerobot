"""Qwen-detected, click-to-select pick-and-place - the user's 2026-09-01 5-step
spec, implemented as click_pick_place.py's already-validated click flow with
two upgrades: (1) Qwen2.5-VL shows what's on the table instead of a fixed
HSV red-cube/black-bin detector, so ANY object can be picked, not just the
red cube; (2) a REAL wrist-cam closed-loop fine_servo for the picked object
(perception.build_color_detect_fn, bootstrapped from Qwen's one-shot color
sample - see that function's docstring) - llm_pick_place.py's fully open-loop
version grasped nothing on its one real run (final gripper position read as
empty), and task_state_machine.fine_servo's closed loop is exactly the fix
already built into this project for that failure mode, just never wired up
for an arbitrary (non-hardcoded-color) object until now.

Flow (the user's own 5 steps):
  1. Start at home (captured live, like every other script here).
  2. perception_qwen.detect_all_qwen() on the current Astra frame - boxes
     drawn on screen (best-effort - see that function's own docstring on
     its real, measured recall gaps; an object it misses can still be
     picked via the manual-description fallback below).
  3. Click near a shown box (or type a description if nothing's boxed
     there) -> IK move to hover (homography) -> REAL wrist-cam closed-loop
     fine_servo using a color detector bootstrapped from THAT box's own
     sampled HSV -> descend_and_grasp.
  4. If grasped: lift, click a destination point (no detection needed -
     an empty spot isn't "an object") -> move there -> descend -> release.
  5. Return home (in a finally: block regardless of outcome, same as
     click_pick_place.py/llm_pick_place.py).

Needs camera_hub.py AND astra_s_live.py already running and publishing (see
click_pick_place.py's docstring).

Run: uv run python3 custom_scripts/vision_pick_place/task_red_cube_to_bin_new_gripper/qwen_click_pick_place.py
"""

from __future__ import annotations

import sys
import time

import cv2

import config
import gripper
import perception
import perception_qwen
import task_state_machine as tsm
from kinematics import CollisionDetected, SOArm101

WINDOW = "Qwen 감지 (박스 근처 클릭, 없으면 텍스트로 설명) - 'q'=취소"
CLICK_MATCH_RADIUS_PX = 60  # 클릭이 어떤 박스와도 이 거리 안에 안 걸리면 "감지 안 됨"으로 취급


def _draw_boxes(frame, detections: list[tuple[perception.Detection, str]]):
    display = frame.copy()
    for det, label in detections:
        x, y, w, h = det.bbox
        cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(display, label, (x, max(0, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return display


def _nearest_detection(detections, pt) -> perception.Detection | None:
    px, py = pt
    best, best_dist = None, CLICK_MATCH_RADIUS_PX
    for det, _label in detections:
        x, y, w, h = det.bbox
        if x <= px <= x + w and y <= py <= y + h:
            return det  # 박스 안 클릭 - 바로 확정
        dist = ((det.cx - px) ** 2 + (det.cy - py) ** 2) ** 0.5
        if dist < best_dist:
            best, best_dist = det, dist
    return best


def wait_for_click(prompt: str, detections: list[tuple[perception.Detection, str]] | None = None) -> tuple[int, int] | None:
    """click_pick_place.py의 wait_for_click과 동일 - detections가 있으면 참고용
    박스를 같이 그려줌."""
    clicked: list[tuple[int, int]] = []

    def on_click(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked.append((x, y))

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_click)
    print(f"[qwen_click_pick_place] {prompt}")
    cap = perception.PublishedFrameSource(config.ASTRA_RGB_FRAME_PATH)
    try:
        while True:
            ok, frame = cap.read()
            if ok:
                shown = _draw_boxes(frame, detections) if detections else frame
                cv2.imshow(WINDOW, shown)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                return None
            if clicked:
                return clicked[-1]
    finally:
        cv2.destroyWindow(WINDOW)


def pick(arm: SOArm101, wrist_cap) -> bool:
    astra_cap = perception.PublishedFrameSource(config.ASTRA_RGB_FRAME_PATH)
    ok, astra_frame = astra_cap.read()
    if not ok or astra_frame is None:
        print("[qwen_click_pick_place] Astra 프레임을 읽지 못했습니다.")
        return False

    print("[qwen_click_pick_place] Qwen으로 테이블 위 물체 탐지 중...")
    detections = perception_qwen.detect_all_qwen(astra_frame)
    for det, label in detections:
        print(f"  - {label}: 픽셀={det.bbox}")
    if not detections:
        print("[qwen_click_pick_place] 감지된 물체 없음 - 그래도 원하는 위치를 클릭하면 직접 지정 가능")

    pt = wait_for_click("집을 물체 근처를 클릭하세요 (q=취소)", detections)
    if pt is None:
        return False

    det = _nearest_detection(detections, pt)
    if det is None:
        # 감지가 놓친 물체 - perception_qwen.detect_all_qwen 자체 문서화된 재현율
        # 한계(모든 물체를 다 찾진 못함) 때문에 남겨둔 수동 대체 경로. 클릭한
        # 자리 주변만 잘라서 묻지 않고(이미 실측으로 정확도가 떨어짐이 확인됨 -
        # perception_qwen.detect_all_qwen 문서 참고) 검증된 방식대로 전체
        # 프레임에 텍스트 설명으로 다시 물어봄.
        desc = input("[qwen_click_pick_place] 감지된 박스와 안 맞음 - 무슨 물체인지 설명 입력: ").strip()
        if not desc:
            print("[qwen_click_pick_place] 설명 없음 - 취소")
            return False
        det = perception_qwen.detect_qwen(astra_frame, desc)
        if det is None:
            print(f"[qwen_click_pick_place] '{desc}'을(를) 찾지 못했습니다 - 취소")
            return False
        print(f"[qwen_click_pick_place] '{desc}' 감지: 픽셀={det.bbox}")

    xy = perception.pixel_to_xy(det.cx, det.cy)
    if xy is None or not perception.is_xy_within_safe_workspace(*xy):
        print("[qwen_click_pick_place] 호모그래피 없음/안전 범위 밖 - 취소")
        return False

    # 폐루프 보정용 색상 감지기를 이 물체의 실측 색상으로 즉석 구성 - 손목캠
    # 프레임마다 Qwen을 다시 부르면 너무 느려서(perception_qwen 문서 참고) 안
    # 됨. 이게 llm_pick_place.py의 open-loop 방식이 실기에서 파지 실패했던
    # 원인을 고치는 부분 - task_state_machine.fine_servo에 실제로 매 프레임
    # 동작하는 detect_fn을 처음으로 넘겨줌.
    hsv_ranges = perception.sample_hsv_ranges(astra_frame, det.bbox)
    color_detect_fn = perception.build_color_detect_fn(hsv_ranges)

    target = (xy[0], xy[1], config.SEARCH_HOVER_XYZ[2])
    print(f"[qwen_click_pick_place] 대상 위치로 이동: ({xy[0]:.3f}, {xy[1]:.3f})")
    try:
        arm.move_to_xyz_converge(target, tolerance_m=0.015, max_iters=20)
    except CollisionDetected as e:
        print(f"[qwen_click_pick_place] 이동 중 충돌 감지 ({e}) - 취소")
        return False

    if not tsm.fine_servo(arm, wrist_cap, color_detect_fn, "선택한 물체", skip_search=True):
        print("[qwen_click_pick_place] 손목캠 정밀 정렬 실패 - 현재 위치에서 그냥 파지 시도")
    return tsm.descend_and_grasp(arm)


def place(arm: SOArm101) -> bool:
    pt = wait_for_click("놓을 위치를 클릭하세요 (q=취소)")
    if pt is None:
        return False
    xy = perception.pixel_to_xy(*pt)
    if xy is None or not perception.is_xy_within_safe_workspace(*xy):
        print("[qwen_click_pick_place] 호모그래피 없음/안전 범위 밖 - 취소")
        return False
    hover = (xy[0], xy[1], config.SEARCH_HOVER_XYZ[2])
    print(f"[qwen_click_pick_place] 놓을 위치로 이동: ({xy[0]:.3f}, {xy[1]:.3f})")
    try:
        arm.move_to_xyz_converge(hover, tolerance_m=0.015, max_iters=20)
    except CollisionDetected as e:
        print(f"[qwen_click_pick_place] 이동 중 충돌 감지 ({e}) - 현재 위치에서 계속")

    try:
        arm.move_z(-config.BIN_DESCEND_M, steps=15, step_delay_s=0.05)
    except CollisionDetected:
        print("[qwen_click_pick_place] 하강 중 접촉 감지 - 현재 위치에서 놓음")
    gripper.open_gripper(arm)
    time.sleep(0.3)
    arm.move_z(config.BIN_DESCEND_M, steps=15, step_delay_s=0.05)
    return True


def main() -> bool:
    wrist_cap = perception.PublishedFrameSource(config.WRIST_FRAME_PATH)
    if not wrist_cap.isOpened():
        print(
            "[qwen_click_pick_place] 손목캠 프레임이 없습니다. 먼저 camera_hub.py를 실행해주세요:\n"
            "  source ~/lerobot_song_venv/bin/activate && "
            "python custom_scripts/vision_pick_place/camera_hub.py"
        )
        return False
    astra_cap = perception.PublishedFrameSource(config.ASTRA_RGB_FRAME_PATH)
    if not astra_cap.isOpened():
        print(
            "[qwen_click_pick_place] Astra 프레임이 없습니다. 먼저 astra_s_live.py를 실행해주세요:\n"
            "  source ~/lerobot_song_venv/bin/activate && "
            "python custom_scripts/vision_pick_place/astra_s_live.py"
        )
        return False

    arm = SOArm101()
    arm.connect()
    print("[qwen_click_pick_place] 연결 성공. 현재 관절각(=홈):", arm.get_joint_deg())
    home_pose = arm.get_joint_deg()
    home_xyz = tuple(arm.kin.forward_kinematics(home_pose[: len(config.ARM_JOINTS)])[:3, 3])

    try:
        gripper.open_gripper(arm)
        if not pick(arm, wrist_cap):
            print("[qwen_click_pick_place] 픽업 실패/취소 - 홈으로 복귀합니다.")
            return False
        print("[qwen_click_pick_place] 파지 성공 - 상승")
        arm.move_z(config.LIFT_M, steps=20, step_delay_s=0.05)
        if not place(arm):
            print("[qwen_click_pick_place] 놓기 실패/취소 - 든 채로 홈 복귀합니다.")
            return False
        print("[qwen_click_pick_place] 완료")
        return True
    except KeyboardInterrupt:
        print("\n[qwen_click_pick_place] 사용자가 중단했습니다.")
        return False
    finally:
        cv2.destroyAllWindows()
        print(f"[qwen_click_pick_place] 홈 포즈로 복귀: {home_xyz}")
        try:
            arm.move_to_xyz_converge(home_xyz, tolerance_m=0.015, max_iters=20)
        except CollisionDetected as e:
            print(f"[qwen_click_pick_place] 홈 복귀 중 충돌 감지, 안전 위치에서 정지: {e}")
        except Exception as e:
            print(f"[qwen_click_pick_place] 홈 복귀 실패: {e}")
        arm.disconnect()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
