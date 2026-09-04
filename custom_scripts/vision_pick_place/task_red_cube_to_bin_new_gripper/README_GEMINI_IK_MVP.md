# SO-101 Gemini 클릭 파지 MVP 인수인계 — 2026-09-04

## 현재 상태

- `click_grasp_bimanual.py`는 Astra RGB 창에서 왼쪽 클릭은 왼팔, 오른쪽 클릭은 오른팔에 작업을 큐잉한다.
- 픽셀은 `homography.json`으로 table-plane 로봇 XY로 변환되고, 보정 작업영역 밖이면 이동 명령을 보내지 않는다.
- 파지 경로는 hover → open-loop descend/grasp → lift → hold다. 임의 Gemini 객체에는 범용 손목 카메라 closed-loop detector가 아직 없으므로 `fine_servo`는 의도적으로 사용하지 않는다.
- **Gemini 라벨 오버레이가 그려진 `camera_preview_gemini.py` 창의 클릭을 이 스크립트에 연결하는 일은 다음 작업이다.** 지금의 클릭 창은 Gemini 탐지를 표시하지 않는다.

## 안전 제약

- 오른팔 보정값은 왼팔 값의 placeholder다. 기본 오른쪽 클릭은 반드시 거부하며, 오른팔의 `calibrate_grasp.py`, `probe_table_height_manual.py`, `measure_grasp_target_px.py` 완료 뒤에만 `--allow-unverified-right`를 쓴다.
- `e`: 양팔 `release_torque()`를 즉시 호출하고, emergency stop 뒤에는 finally 블록의 홈 복귀 명령도 보내지 않는다.
- `q`/ESC: 정상 종료로 취급하여 각 팔을 기록한 시작 홈 XYZ로 복귀 시도한다.
- `max_relative_target`, 소프트웨어 관절 제한, 펌웨어 전류/토크 보호, 관절 지연 기반 충돌 감지를 `kinematics.py`의 기존 구현으로 재사용한다.

## 실행 전 검증

실물 하드웨어 동작은 자동화하지 않는다. 첫 가동은 반드시 사람의 시야 안에서, 팔 주변을 비우고, `e` 키를 바로 누를 수 있는 상태에서 진행한다.

```bash
cd ~/lerobot/custom_scripts/vision_pick_place/task_red_cube_to_bin_new_gripper
uv run python3 -m py_compile click_grasp_bimanual.py
uv run python3 click_grasp_bimanual.py --help
```

실물 시험 명령:

```bash
uv run python3 click_grasp_bimanual.py \
  --left-port /dev/ttyACM0 \
  --right-port /dev/ttyACM1
```

첫 시험에서는 오른쪽 클릭이 경고만 출력하고 모터 명령을 전혀 보내지 않는지 먼저 확인한다. 이어서 안전 작업영역 안의 물체를 왼쪽 클릭하고, 비정상 움직임이면 `e`를 눌러 양팔 토크가 풀린 뒤 추가 홈 복귀 명령이 없는지 확인한다.

## 알려진 미해결 사항

- 오른팔 실제 보정 전에는 오른팔 pick을 활성화하지 않는다.
- 현재 Astra RGB의 Gemini 포인트 탐지는 별도 `~/so101-bimanual-teleop/camera_preview_gemini.py`에서 동작한다. 카메라/USB 장치 경합을 피하려면 Gemini preview와 이 로봇 제어 스크립트를 동시에 실행하지 않는다.
- Gemini 응답 시간은 약 5–6초다. 영상 렌더링은 논블로킹이지만 라벨 자체는 그 주기로 갱신된다.
