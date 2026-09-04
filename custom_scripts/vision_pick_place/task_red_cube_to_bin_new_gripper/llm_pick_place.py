"""LLM-located pick-and-place (기본: 로컬 Qwen2.5-VL-3B, perception_qwen.py 참고 -
--backend gemini로 클라우드 Gemini도 여전히 선택 가능) - click_pick_place.py의 흐름을
그대로 따르되, "어디를 볼지"만 마우스 클릭 대신 LLM의 1회 open-vocabulary 감지로
대체한 버전. 텍스트로 대상을 설명하면("빨간 큐브", "파란 장갑" 등) 그 물체를 찾아
집고, 목적지 설명(기본값 "검은 쓰레기통")에 내려놓는다.

다운스트림(호모그래피 pixel_to_xy -> IK move_to_xyz_converge -> descend_and_grasp/place)은
click_pick_place.py와 완전히 동일 - 이 파일이 새로 하는 일은 "픽셀 좌표를 어떻게
얻는가" 하나뿐이다.

2026-09-01: 처음엔 Qwen2-VL-2B로 시작 - 깨끗한 실제 프레임(검은 큐브/작은 빨간
큐브/주황 클립)으로 테스트해보니 좌표계 표기가 호출마다 달라지고(정규화 0-1 vs
자체 리사이즈 픽셀공간 - perception_qwen.py 참고) 3개 중 1개(큰 검은 큐브)만
정확히 맞음. Qwen2.5-VL-3B로 교체(같은 VRAM 예산) 후 동일 테스트 재실행 -
3/3 정확히 감지, 좌표계도 원본 프레임 픽셀 공간으로 일관됨.

픽셀+뎁스 -> 진짜 3D(카메라/로봇 외부 파라미터) 변환도 시도했으나, Astra의 FOV로
직접 pinhole 공식을 재구현한 게 SDK 자체 변환(convert_depth_to_world)과 최대
555mm까지 어긋나는 걸 확인(depth/color 스트림 FOV가 다른가도 확인했지만 동일 -
원인 아님) - 이 센서의 실제 변환은 광고된 FOV만으로는 못 구하는 자체 보정 데이터가
더 들어간 것으로 보여 여기서 중단(capture_astra_intrinsics.py 참고, ABANDONED로
표시해둠). 사용자 결정에 따라 진짜 3D 백프로젝션 대신, 이미 실측 검증된 호모그래피
(xy) + Astra 뎁스-델타(z) 방식을 그대로 재사용 - perception.detect_red_cube/
detect_black_bin이 하던 감지 자리에 perception_qwen.detect_qwen만 끼워넣은 셈.

알려진 한계 (의도적으로 단순화한 부분):
  - fine_servo의 손목캠 폐루프 보정(task_state_machine.fine_servo)은 HSV 색상
    감지에 의존하므로 임의 물체엔 못 씀 - LLM을 손목캠 프레임마다 다시 부르면
    로컬 Qwen도 호출당 1초 안팎이라 폐루프로 쓰기엔 아직 느림. 따라서 이 버전은
    Astra 고정 시점의 호모그래피 좌표 하나로 이동 후 바로 하강/파지를 시도하는
    open-loop 방식 - 실측해서 부정확하면 그 다음 단계로 보강할 것.
  - perception.estimate_cube_height_m()은 내부적으로 detect_red_cube만 봐서 다른
    물체에선 None을 반환하고, descend_and_grasp는 이미 그 경우 TABLE_Z로 안전하게
    폴백하도록 되어 있음(perception.py 참고) - 높이 추정 없이 TABLE_Z까지 접촉
    감지로 하강하는 셈이라 크래시는 없지만, 테이블에서 많이 뜬 물체는 정확도가 낮음.

Needs camera_hub.py AND astra_s_live.py already running and publishing (see
click_pick_place.py's docstring). --backend gemini를 쓸 경우에만 GEMINI_API_KEY
(또는 ~/.gemini_api_key) 필요 - 기본 backend(qwen)는 로컬 모델이라 API 키 불필요.

2026-09-01 (같은 날, 실기 첫 실행 전): LLM 감지가 잘못됐을 때 모터를 상하게 하는 걸
막기 위한 안전장치 두 개 추가 - (1) perception.is_xy_within_safe_workspace: 호모그래피
결과가 실측 보정점(homography.json, 점 4개뿐) 범위를 크게 벗어나면 이동 자체를 거부
(호모그래피는 그 범위 밖에선 이미 외삽 중이라 LLM 오감지가 겹치면 검증 안 된 자세로
팔을 반복해서 밀어넣을 수 있음 - move_to_xyz의 MAX_MOVE_DELTA_DEG 캡은 "현재 위치에서
얼마나 멀리" 만 보지 목표 자체가 안전한지는 안 봄). (2) _confirm_move: 실제로 움직이기
전에 kinematics.SOArm101.preview_move()(이미 있었지만 어떤 흐름도 안 쓰던 함수)로
목표/이동량을 미리 보여주고 사람이 확인해야 진행 - stall/충돌 감지는 이미 움직이기
시작한 뒤의 마지막 방어선이라, 그 앞에 하나 더 둔 것. --yes로 이후 자동화 시 생략 가능.

Run:
  uv run python3 custom_scripts/vision_pick_place/task_red_cube_to_bin_new_gripper/llm_pick_place.py \\
      --pick "빨간 큐브" --place "검은 쓰레기통"
"""

from __future__ import annotations

import argparse
import sys
import time

import config
import gripper
import perception
import task_state_machine as tsm
from kinematics import CollisionDetected, SOArm101

DEFAULT_PICK_DESC = "red cube"
DEFAULT_PLACE_DESC = "black trash bin"

# 2026-09-01: 로컬 Qwen2-VL로 전환(perception_qwen.py) - Gemini는 --backend gemini로
# 여전히 선택 가능(정확도 비교용). 각 백엔드 모듈은 lazy-load라 실제 쓰는 쪽만 로딩됨.
BACKENDS = {"qwen": "perception_qwen", "gemini": "perception_zeroshot"}


def locate_via_llm(astra_cap: perception.PublishedFrameSource, description: str, backend: str) -> tuple[int, int] | None:
    """click_pick_place.wait_for_click과 같은 역할 - 클릭 좌표 대신 Astra 고정
    시점 프레임 하나에 대해 LLM에게 1회 물어 감지된 물체의 중심 픽셀을 반환.
    감지 실패(물체 없음/응답 파싱 실패)면 None."""
    ok, frame = astra_cap.read()
    if not ok or frame is None:
        print("[llm_pick_place] Astra 프레임을 읽지 못했습니다.")
        return None
    print(f"[llm_pick_place] {backend}에게 '{description}' 위치를 묻는 중...")
    import importlib

    module = importlib.import_module(BACKENDS[backend])
    detect_fn = module.detect_qwen if backend == "qwen" else module.detect_zeroshot
    det = detect_fn(frame, description)
    if det is None:
        print(f"[llm_pick_place] '{description}'을(를) 화면에서 찾지 못했습니다.")
        return None
    print(f"[llm_pick_place] '{description}' 감지: 픽셀=({det.cx:.0f}, {det.cy:.0f}) bbox={det.bbox}")
    return (int(det.cx), int(det.cy))


def _safe_xy_or_none(xy: tuple[float, float] | None, label: str) -> tuple[float, float] | None:
    """호모그래피 결과가 없거나(None) 실측 보정된 적 없는 영역까지 벗어나면
    (perception.is_xy_within_safe_workspace) 이동 자체를 거부 - LLM 오감지가
    호모그래피를 크게 외삽시켜 팔을 검증 안 된 자세로 반복해서 밀어넣는 걸
    막는 안전장치. 이 파일이 다루는 감지 소스(LLM)는 아직 신뢰가 안 쌓였으므로
    click_pick_place.py(이미 실측 검증됨)엔 없는 이 검사를 여기서만 추가함."""
    if xy is None:
        print(f"[llm_pick_place] 호모그래피 없음 (homography.json 확인 필요) - {label} 이동 취소")
        return None
    if not perception.is_xy_within_safe_workspace(*xy):
        print(
            f"[llm_pick_place] {label} 좌표 ({xy[0]:.3f}, {xy[1]:.3f})가 실측 보정된 작업공간 범위를 "
            f"벗어남 - 잘못 감지됐을 가능성이 높아 이동을 거부합니다."
        )
        return None
    return xy


def _confirm_move(arm: SOArm101, xyz: tuple[float, float, float], label: str, auto_yes: bool) -> bool:
    """이동 전 미리보기 + 사람 확인 게이트 - arm.preview_move()는 이미 있었지만
    지금까지 어떤 흐름도 실제로 쓰지 않던 함수. 아직 실기 검증 안 된 LLM 감지
    소스가 처음 이 좌표로 팔을 보내기 전에, 목표/이동량을 사람이 보고 명백히
    이상하면 여기서 멈출 수 있게 함 - stall/충돌 감지는 "이미 움직이기 시작한
    뒤" 막는 마지막 방어선이라, 그 전 단계로 하나 더 둠. --yes로 이후 자동화
    시 건너뛸 수 있음."""
    plan = arm.preview_move(xyz)
    print(f"[llm_pick_place] {label} 이동 미리보기: 목표={tuple(round(v,3) for v in xyz)} "
          f"최대 관절 이동량={plan['max_abs_delta_deg']:.1f}deg")
    if auto_yes:
        return True
    ans = input(f"[llm_pick_place] {label}(으)로 이동할까요? (y/N): ").strip().lower()
    if ans != "y":
        print(f"[llm_pick_place] {label} 이동 취소됨 (사용자 확인 거부)")
        return False
    return True


def pick(arm: SOArm101, wrist_cap, astra_cap, description: str, backend: str, auto_yes: bool) -> bool:
    pt = locate_via_llm(astra_cap, description, backend)
    if pt is None:
        return False
    xy = _safe_xy_or_none(perception.pixel_to_xy(*pt), "픽업 대상")
    if xy is None:
        return False
    target = (xy[0], xy[1], config.SEARCH_HOVER_XYZ[2])
    print(f"[llm_pick_place] 대상 추정 위치로 이동: ({xy[0]:.3f}, {xy[1]:.3f})")
    if not _confirm_move(arm, target, "픽업 대상", auto_yes):
        return False
    try:
        arm.move_to_xyz_converge(target, tolerance_m=0.015, max_iters=20)
    except CollisionDetected as e:
        print(f"[llm_pick_place] 이동 중 충돌 감지 ({e}) - 취소")
        return False
    # 손목캠 폐루프 보정 없음 - 모듈 docstring "알려진 한계" 참고.
    return tsm.descend_and_grasp(arm)


def place(arm: SOArm101, wrist_cap, astra_cap, description: str, backend: str, auto_yes: bool) -> bool:
    pt = locate_via_llm(astra_cap, description, backend)
    if pt is None:
        return False
    xy = _safe_xy_or_none(perception.pixel_to_xy(*pt), "놓기 위치")
    if xy is None:
        return False
    hover = (xy[0], xy[1], config.SEARCH_HOVER_XYZ[2])
    print(f"[llm_pick_place] 목적지 추정 위치로 이동: ({xy[0]:.3f}, {xy[1]:.3f})")
    if not _confirm_move(arm, hover, "놓기 위치", auto_yes):
        return False
    try:
        arm.move_to_xyz_converge(hover, tolerance_m=0.015, max_iters=20)
    except CollisionDetected as e:
        print(f"[llm_pick_place] 이동 중 충돌 감지 ({e}) - 현재 위치에서 계속")

    try:
        arm.move_z(-config.BIN_DESCEND_M, steps=15, step_delay_s=0.05)
    except CollisionDetected:
        print("[llm_pick_place] 하강 중 접촉 감지 - 현재 위치에서 놓음")
    gripper.open_gripper(arm)
    time.sleep(0.3)
    arm.move_z(config.BIN_DESCEND_M, steps=15, step_delay_s=0.05)
    return True


def main() -> bool:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pick", default=DEFAULT_PICK_DESC, help="집을 물체 설명 (기본: %(default)r)")
    parser.add_argument("--place", default=DEFAULT_PLACE_DESC, help="내려놓을 위치 설명 (기본: %(default)r)")
    parser.add_argument("--backend", choices=list(BACKENDS), default="qwen", help="위치 감지에 쓸 LLM (기본: %(default)r)")
    parser.add_argument("--yes", action="store_true",
                         help="이동 전 확인 프롬프트 건너뛰기 (기본: 매 이동마다 미리보기+확인 필요 - 아직 실기 "
                              "검증 안 된 LLM 감지라 처음엔 켜두는 걸 권장, 신뢰 쌓이면 자동화용으로 사용)")
    args = parser.parse_args()

    wrist_cap = perception.PublishedFrameSource(config.WRIST_FRAME_PATH)
    if not wrist_cap.isOpened():
        print(
            "[llm_pick_place] 손목캠 프레임이 없습니다. 먼저 camera_hub.py를 실행해주세요:\n"
            "  source ~/lerobot_song_venv/bin/activate && "
            "python custom_scripts/vision_pick_place/camera_hub.py"
        )
        return False
    astra_cap = perception.PublishedFrameSource(config.ASTRA_RGB_FRAME_PATH)
    if not astra_cap.isOpened():
        print(
            "[llm_pick_place] Astra 프레임이 없습니다. 먼저 astra_s_live.py를 실행해주세요:\n"
            "  source ~/lerobot_song_venv/bin/activate && "
            "python custom_scripts/vision_pick_place/astra_s_live.py"
        )
        return False

    arm = SOArm101()
    arm.connect()
    print("[llm_pick_place] 연결 성공. 현재 관절각:", arm.get_joint_deg())
    home_pose = arm.get_joint_deg()
    home_xyz = tuple(arm.kin.forward_kinematics(home_pose[: len(config.ARM_JOINTS)])[:3, 3])

    try:
        gripper.open_gripper(arm)
        if not pick(arm, wrist_cap, astra_cap, args.pick, args.backend, args.yes):
            print(f"[llm_pick_place] '{args.pick}' 픽업 실패/취소 - 홈으로 복귀합니다.")
            return False
        print("[llm_pick_place] 파지 성공 - 상승")
        arm.move_z(config.LIFT_M, steps=20, step_delay_s=0.05)
        if not place(arm, wrist_cap, astra_cap, args.place, args.backend, args.yes):
            print(f"[llm_pick_place] '{args.place}' 위치에 놓기 실패/취소 - 든 채로 홈 복귀합니다.")
            return False
        print("[llm_pick_place] 완료")
        return True
    except KeyboardInterrupt:
        print("\n[llm_pick_place] 사용자가 중단했습니다.")
        return False
    finally:
        print(f"[llm_pick_place] 홈 포즈로 복귀: {home_xyz}")
        try:
            arm.move_to_xyz_converge(home_xyz, tolerance_m=0.015, max_iters=20)
        except CollisionDetected as e:
            print(f"[llm_pick_place] 홈 복귀 중 충돌 감지, 안전 위치에서 정지: {e}")
        except Exception as e:
            print(f"[llm_pick_place] 홈 복귀 실패: {e}")
        arm.disconnect()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
