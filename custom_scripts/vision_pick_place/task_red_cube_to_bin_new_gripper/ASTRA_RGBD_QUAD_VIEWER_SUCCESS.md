# Astra RGB+Depth 4분할 뷰어 — 성공 기록 (2026-09-07)

## 결과
Astra S 1대에서 depth + RGB 동시 스트림 + 양쪽 손목캠 2대, 총 4분할 화면 실시간 확인 완료.

## 원인이었던 문제들
1. Astra depth 스트림 정지(stall) — OpenNI2 `read_frame()` 블로킹 + 스레드 캐시가 멈춘 프레임을 새 mtime으로 재발행해 "정상처럼" 보임.
2. `Failed to set USB interface!` 에러 — 실제로는 이전 supervisor 프로세스가 Astra USB를 이미 점유 중인 상태에서 두 번째 인스턴스를 실행한 충돌. USB 권한/venv 문제 아니었음.

## 수정
- `astra_s_stream_supervisor.py`: Astra를 별도 프로세스로 격리, RGB+depth 동시 헬스체크(`--mode rgbd`), 시작 전 USB 점유 검사(점유 중이면 무한 재시도 대신 exit code 2), 실패 시 1s→2s→4s 지수 백오프.
- `astra_s_depth_hub.py`: depth를 메인 스레드에서 직접 발행 (hang이 파일 갱신 중단으로 바로 드러나게).
- `camera_hub_supervisor.py`: 양쪽 손목캠 워치독, 하나라도 멈추면 camera hub 재기동.
- `quad_viewer.py` / `quad_viewer_separate.py`: depth 없으면 IR로 표시 (Astra 구조광 depth와 IR은 동시 사용 불가, 배타적 fallback).
- `config.py`: IR publish 경로만 추가 (BIN_POSE_XYZ 확정값은 미변경).

## 확정값 (건드리지 말 것)
- 오른팔 BIN_POSE_XYZ: `(0.3666, -0.0423, 0.0098)`
- 왼팔 `LEFT_OVERRIDES["BIN_POSE_XYZ"]`: `(0.2334, 0.0495, 0.0188)`

## 실행 방법
```bash
# 터미널 1: Astra RGB+Depth (반드시 한 인스턴스만)
cd ~/lerobot
uv run python custom_scripts/vision_pick_place/astra_s_stream_supervisor.py --mode rgbd

# 터미널 2: 양쪽 손목캠 워치독
CAMERA_HUB_HEADLESS=1 uv run python custom_scripts/vision_pick_place/camera_hub_supervisor.py

# 터미널 3: 4분할 GUI (GUI opencv 되는 venv에서)
cd ~/lerobot/custom_scripts/vision_pick_place/task_red_cube_to_bin_new_gripper
uv run --active python quad_viewer.py
```

IR fallback (depth 완전 불가 시): `--mode ir`.

## 검증
- 실시간 4분할 화면 사용자 확인 완료.
- `test_camera_hub.py`: 2 passed.
- 20초간 depth/RGB/오른손목/왼손목 4개 파일 모두 매초 갱신, restart loop 없음.

## 남은 작업
- 왼팔 그리퍼 열림/닫힘 캘리브레이션 (미착수).
- IR로 클릭 위치 잡을 경우 IR 전용 homography 재보정 필요.
