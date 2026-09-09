# SO-101 2팔 + Astra S 로컬 VLM Pick-and-Place 진행 기록

작성일: 2026-09-08  
저장소: `/home/youngchan/lerobot`  
브랜치: `mvp/gemini-click-grasp`  
참고 문서: [ROBOTIS GraspNet Pick-and-Place 기술 문서](https://docs.robotis.com/docs/systems/omy/resources/technical_story/graspnet_pick_and_place/)

## 목표

- SO-101 follower arm 2대와 Astra S depth camera 사용
- Astra S가 발행한 RGB-D 프레임에서 스크린샷 저장
- 로컬 Ollama VLM으로 화면 속 물체 탐지
- 사용자가 `빨간 블록을 파란 트레이에 가져다줘`처럼 입력하면 source/destination을 해석
- 두 팔의 workspace와 안전 조건을 확인한 뒤 pick-and-place 실행
- 실제 로봇 이동 전에는 항상 dry-run preview를 거치고, 캘리브레이션 확인 전 실행 금지

## 지금까지 구현한 내용

### 1. Astra S RGB-D 입력

- `custom_scripts/vision_pick_place/astra_s_live.py`
  - Astra S 스트림을 headless로 실행
  - `/tmp/vsp_astra_rgb.png`와 `/tmp/vsp_astra_depth_mm.npy`를 갱신
- `custom_scripts/vision_pick_place/astra_s_depth_hub.py`
  - RGB/depth publisher 및 정합 상태 확인 로직
- `custom_scripts/vision_pick_place/camera_utils.py`
  - 카메라 프레임 처리 공통 유틸리티
- `so101_graspnet_pick_and_place/snapshot.py`
  - 발행 중인 RGB/depth 프레임을 하나의 snapshot으로 저장
  - RGB와 depth 해상도가 다르면 depth를 RGB 격자에 nearest-neighbor 방식으로 맞춤
- `so101_graspnet_pick_and_place/frame_io.py`
  - `rgb.png`, `depth_mm.npy`, `frame.json` 저장/로드

실제 확인된 입력 크기:

- RGB: `640 x 480`
- Depth: `320 x 240`, `uint16`, millimeter 단위

### 2. 로컬 VLM 탐지 및 자연어 명령

- `so101_graspnet_pick_and_place/local_vlm.py`
  - Ollama HTTP API(`/api/chat`) 호출
  - 기본 endpoint: `http://127.0.0.1:11434`
  - 기본 모델: `qwen2.5vl:7b`
  - 물체 탐지 결과와 명령 해석 결과를 JSON으로 강제
  - 잘못된 JSON 응답에 대한 재시도 처리
- `so101_graspnet_pick_and_place/models.py`
  - `DetectedObject`, `TaskCommand` 데이터 모델
- `so101_graspnet_pick_and_place/local_cli.py`
  - RGB-D 입력 → VLM 탐지 → 자연어 명령 해석 → pick/place 계획 생성
  - `--dry-run` preview 지원
  - 실제 실행은 `--execute --yes`와 캘리브레이션 승인 조건 필요

### 3. 2팔 계획, 좌표 변환, 안전장치

- `so101_graspnet_pick_and_place/local_planner.py`
  - bbox 내부 depth ROI median으로 3D point 계산
  - camera 좌표를 공통 base 좌표로 변환
  - source/destination을 두 팔 workspace에 라우팅
  - depth 유효성, confidence, 중앙 금지 영역 검사
- `so101_graspnet_pick_and_place/routing.py`
  - 양팔 routing 및 port 정보 관리
- `so101_graspnet_pick_and_place/config.py`
  - 카메라 intrinsic, camera-to-base transform, arm port/workspace 설정 로드
- `so101_graspnet_pick_and_place/adapters.py`
  - SO-101 gripper/wrist 상태 확인 어댑터
- 기존 지원 모듈:
  - `calibration.py`, `geometry.py`, `orchestrator.py`, `pipeline.py`
  - `left_arm.py`, `home_pose.py`, `graspnet_adapter.py`, `graspnet_runtime.py`

### 4. 설정 예시

- `so101_graspnet_pick_and_place/config.example.json`
  - 왼팔 port: `/dev/so101_left`
  - 오른팔 port: `/dev/so101_right`
  - 카메라 intrinsic과 camera-to-base transform은 실제 캘리브레이션 값으로 교체 필요
  - 현재 `calibration_confirmed`는 `false`

## 실행 방법

터미널 1: Astra S publisher

```bash
cd /home/youngchan/lerobot/custom_scripts/vision_pick_place
ASTRA_LIVE_HEADLESS=1 uv run python astra_s_live.py
```

터미널 2: Ollama

```bash
ollama serve
ollama pull qwen2.5vl:7b
```

터미널 3: 자연어 명령 preview

```bash
cd /home/youngchan/lerobot
PYTHONPATH=. uv run python -m \
  custom_scripts.vision_pick_place.so101_graspnet_pick_and_place.local_cli \
  --config custom_scripts/vision_pick_place/so101_graspnet_pick_and_place/config.example.json \
  --rgb-path /tmp/vsp_astra_rgb.png \
  --depth-path /tmp/vsp_astra_depth_mm.npy \
  --command "빨간 블록을 파란 트레이에 가져다줘" \
  --dry-run
```

실제 실행 전 필수 조건:

1. 카메라 intrinsic 및 camera-to-base transform 실측
2. 두 팔 port와 workspace 실측
3. `calibration_confirmed: true`로 바꾸기 전 reprojection/workspace 검증
4. dry-run 결과의 source/destination bbox와 3D 좌표 확인
5. 저속·단일 물체로 각 팔을 개별 검증
6. 승인 후에만 `--execute --yes` 사용

## 설치 및 환경 확인

- Ollama가 처음에는 `command not found`였으나 현재 설치 확인:
  - 경로: `/usr/local/bin/ollama`
  - 버전: `0.33.3`
  - 모델: `qwen2.5vl:7b`
- Astra publisher는 headless 실행 상태까지 확인됨
- USB priority warning은 발생했지만 카메라 프레임은 생성됨

## 검증 결과

- 신규 패키지 테스트는 직전 검증에서 `67 passed`까지 통과
- RGB/depth 크기 불일치 문제를 확인했고 snapshot 단계에서 depth를 RGB 크기로 보정하도록 수정
- 전체 저장소 pytest는 기존 Git LFS placeholder 때문에 실패:
  - `tests/artifacts/cameras/image_128x128.png`가 실제 PNG가 아니라 LFS pointer 텍스트
  - 관찰 결과: `91 passed, 2 skipped, 1 failed`
- 실제 촬영 화면에는 당시 명령의 `빨간 블록`과 `파란 트레이`가 명확히 배치되어 있지 않아 계획 실행까지 진행되지 않음
- 마지막 VLM dry-run은 모델이 유효하지 않은 bbox를 반환하여 중단됨. 실제 물체를 화면에 배치한 후 bbox 검증/재시도 로직을 추가 확인해야 함
- 로봇 팔의 실제 이동은 아직 수행하지 않음

## 현재 남은 작업

1. 실제 물체와 트레이를 화면에 배치하고 VLM 탐지 재검증
2. 빈 객체 응답 및 비정상 bbox를 CLI에서 안전하게 거부/재시도
3. Astra RGB-depth 정합 및 depth scale 실측 검증
4. camera-to-base hand-eye calibration 및 양팔 workspace calibration
5. 두 팔 각각 dry-hover 검증
6. 단일 물체 실제 pick/place 1회 검증
7. 이후에만 양팔 작업과 반복 실행으로 확대

## 안전상 주의

- 현재 설정으로 실제 로봇 실행을 하지 않음
- `--dry-run`은 계획만 출력하며 모터를 움직이지 않음
- 실제 실행은 캘리브레이션 승인, workspace 검사, gripper/wrist 검증을 모두 통과해야 함
- 5-DOF SO-101은 6-DOF GraspNet pose를 그대로 추종하지 않고 위치 우선 방식으로 제한됨
- 이 문서에 기록된 경로와 설정은 현재 로컬 작업공간 기준이며, 두 팔 장치명이 실제 udev 이름과 다를 수 있음
