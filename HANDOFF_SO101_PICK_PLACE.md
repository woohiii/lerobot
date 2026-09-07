# SO-101 큐브 픽앤플레이스 인수인계

작성일: 2026-09-02 (Asia/Seoul)

## 하드웨어

- 팔로워: `/dev/ttyACM0` (ACM-0), 리더: `/dev/ttyACM1` (ACM-1)
- Astra S: OpenNI2/Orbbec 전용 장치 (`2bc5:0402`), `/dev/video*`로 열지 않는다.
- 손목 카메라 실제 영상 노드: `/dev/video4` (YUYV 640x480). `/dev/video5`는 metadata 노드라 사용 금지.
- Astra와 손목 카메라는 한 프로세스씩만 장치 점유 가능.

## 현재 상태

- 현재 실행 중인 카메라/로봇 프로세스 없음(`pgrep` 확인 기준).
- 마지막 시도에서 정렬 오차 악화 안전 게이트가 토크를 해제했다. 모터에 명령하기 전 반드시 안전 토크 프리플라이트를 다시 실행한다.
- 팔은 손으로 홈 위치에 놓고 `home_pose_readonly.json`에 저장한 적이 있다. 홈 저장은 읽기 전용이다.
- 마지막 성공 동작은 큐브 위 안전 고도(명령 z=0.13 m, 실시간 FK 약 0.11 m) 접근 후 유지였다. 하강/그리퍼 닫기/열기는 아직 성공적으로 수행하지 않았다.

## 반드시 지킬 안전 규칙

1. 물리 이동 전 사용자가 작업공간이 비었는지 확인한다.
2. `safe_torque_enable.py --enable-torque --arm-supported`로 현재 자세를 읽고 목표값을 현재값으로 동기화한 뒤 저토크(35%)로만 고정한다.
3. 모션은 `safe_hover_approach.py` 또는 동일한 안전 루틴을 사용한다. 관절 명령 간격은 최대 3도, 온도 65°C/전압 9.0–14.0V/추종오차 초과 시 전 관절 토크 해제.
4. 하강 전에는 반드시 손목 영상 폐루프 정렬과 별도 테이블/충돌 확인을 통과시킨다. Astra 깊이는 절대 base-Z로 사용하지 않는다.
5. 토크 해제 후 팔이 중력으로 내려갈 수 있으므로, 다시 움직이기 전 항상 FK 높이를 읽는다.

## 중요한 파일

- 안전 토크: `custom_scripts/vision_pick_place/click_pick_place_safe/safe_torque_enable.py`
- 안전 고도 접근: `custom_scripts/vision_pick_place/click_pick_place_safe/safe_hover_approach.py`
- 손목 X/Y 2 mm 프로브: `.../safe_wrist_probe.py`
- 손목 1회 보정(현재는 오차 악화로 중단됨): `.../safe_wrist_align.py`
- 홈 읽기: `.../capture_home_readonly.py`
- 홈 기록: `.../home_pose_readonly.json`
- 손목 jaw 중심: `.../wrist_jaw_center.json` (`[238, 360]`, `/dev/video4`)
- 마지막 리포트: `.../safe_hover_approach_report.json`, `.../safe_wrist_probe_report.json`, `.../safe_wrist_align_report.json`
- Astra IR 단독 창: `custom_scripts/vision_pick_place/astra_s_ir_live.py`
- IR+손목 2분할 창: `custom_scripts/vision_pick_place/astra_s_ir_wrist_live.py` (현재 기본으로 쓰는 조합)
- Depth+손목 2분할 창: `custom_scripts/vision_pick_place/astra_s_depth_wrist_live.py`
- ~~IR+Depth+손목 3분할 창~~: `astra_s_ir_depth_wrist_live.py`는 **사용 금지**. Astra S는 이 OpenNI2 드라이버에서 IR과 Depth 스트림을 동시에 열 수 없다(둘 다 `start()`한 상태면 `ir.read_frame()`이 영원히 블록, 60초 넘게도 리턴 안 함; Depth만 단독이면 즉시 정상). IR을 봐야 하면 Depth/손목 창을 먼저 닫고 `astra_s_ir_live.py`를 별도로 띄운다.
- 기존 RGB+Depth 게시기: `custom_scripts/vision_pick_place/astra_s_live.py`

## 재현 명령

```bash
# 읽기 전용 홈 확인
uv run python custom_scripts/vision_pick_place/click_pick_place_safe/capture_home_readonly.py

# 토크 안전 고정(물리 이동 전 필수)
uv run python custom_scripts/vision_pick_place/click_pick_place_safe/safe_torque_enable.py --enable-torque --arm-supported

# Astra RGB+Depth 게시(손목 정렬/기존 Astra 검출에 필요)
ASTRA_LIVE_HEADLESS=1 /home/youngchan/lerobot_song_venv/bin/python custom_scripts/vision_pick_place/astra_s_live.py

# 안전 고도: 먼저 옵션 없이 미리보기, 확인 후 --execute
uv run python custom_scripts/vision_pick_place/click_pick_place_safe/safe_hover_approach.py
uv run python custom_scripts/vision_pick_place/click_pick_place_safe/safe_hover_approach.py --execute --workspace-clear --keep-clear

# 손목 X/Y 반응 프로브(안전 고도에서만)
uv run python custom_scripts/vision_pick_place/click_pick_place_safe/safe_wrist_probe.py --execute --workspace-clear --axis x
uv run python custom_scripts/vision_pick_place/click_pick_place_safe/safe_wrist_probe.py --execute --workspace-clear --axis y
```

## 현재의 다음 과제

X/Y 프로브의 측정값으로 만든 초기 Jacobian 보정이 실제 1회 정렬에서 오차를 악화시켜 중단됐다. 다음 AI는 각 보정 이동 후 실제 픽셀 변화로 Jacobian을 갱신하는 적응형 루프를 구현해야 한다. 한 번에 큰 보정하지 말고 1–3 mm씩 적용하며, 오차 악화 시 토크 해제·현재 자세 기록·재고도 접근 순서를 지킨다. IR+Depth+손목 3분할은 관찰용이며, RGB 게시기와 동시에 Astra를 열면 장치 충돌이 난다.

데이터 수집/학습은 아직 본격 시작하지 않았다. 텔레옵션은 다음 포트 조합으로 동작이 확인됐다: follower `/dev/ttyACM0`, leader `/dev/ttyACM1`.
