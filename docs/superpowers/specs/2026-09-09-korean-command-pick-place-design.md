# 한국어 명령 기반 RGB-D Pick-and-Place MVP

## 목표

사용자가 다음처럼 한국어 한 문장으로 물체와 목적지를 지정하면, Astra S RGB-D와 노트북의 로컬 VLM이 해당 대상을 찾아 SO-101 좌측 팔로 안전하게 집어 옮긴다.

```
파란색 펜을 주워서 검은색 박스에다 담아줘
```

물체와 목적지의 이름·색상·위치는 코드에 고정하지 않는다. 매 실행마다 명령과 카메라 장면으로 결정한다.

## 사용자 인터페이스

```bash
uv run python -m so101_graspnet_pick_and_place.korean_pick_place \
  --command "파란색 펜을 주워서 검은색 박스에다 담아줘" \
  --calibration /tmp/handeye_left_v7/hand_eye.json \
  --port /dev/ttyACM3 \
  --preview
```

`--preview`는 RGB overlay, 3D 좌표, IK 계획만 표시하고 모터 명령을 보내지 않는다. 실제 동작은 같은 명령에 `--execute`를 추가하고, 화면에 표시한 source/destination overlay를 확인한 뒤 별도 확인 문자열을 입력할 때만 허용한다.

## 파이프라인

1. 로컬 VLM은 한국어 명령을 JSON 계약으로 구조화한다.
   - `pick_query`: 예: `파란색 펜`
   - `place_query`: 예: `검은색 박스`
2. 동일 RGB 프레임에서 VLM이 source와 destination bbox 및 confidence를 반환한다.
3. 각 bbox의 유효 depth ROI를 robust median으로 역투영하여 camera XYZ를 산출한다.
4. 연속 3프레임에서 source/destination XYZ 분산이 기준 이내일 때만 hand-eye transform으로 base XYZ로 변환한다.
5. source에는 top-grasp, destination에는 컨테이너 내부 또는 빈 상단 중심의 release point를 만들고, home→hover→approach→grasp→lift→destination hover→release→retreat 전체 IK를 사전 검증한다.
6. 모든 게이트 통과 뒤에만 제한된 실행을 한다. 각 motion leg 뒤에는 현재 joint/FK와 카메라를 재확인하며, 실패하면 상승·후퇴하고 중단한다.

## 실패-우선 안전 규칙

- source 또는 destination이 0개·복수개·낮은 confidence이면 움직이지 않는다.
- depth=0, RGB-depth 정합 미확인, 3프레임 위치 불안정, 작업영역 밖, 테이블 침범, IK 실패, 과도한 관절 변화가 있으면 중단한다.
- source와 destination bbox가 겹치거나, destination의 빈 공간을 확인할 수 없으면 중단한다.
- 안전 home, hand-eye RMS, left follower calibration ID와 실제 port를 검증한다.
- 첫 MVP의 실제 실행은 한 번의 pick/place와 하나의 팔만 허용한다. 손목 자세가 수직 top-grasp 범위를 넘는 물체·투명/반사체·집게와 겹친 물체는 거부한다.

## 구성 요소

- `command_parser`: Korean instruction → typed pick/place query JSON. 자유 텍스트를 motor command로 사용하지 않는다.
- `local_vlm`: query별 bbox/confidence 탐지. 현재 Ollama client 계약을 재사용하고 Qwen은 선택 backend로 유지한다.
- `localizer`: Astra RGB-depth aligned frame, valid-depth ROI, three-frame stability, camera→base transform.
- `planner`: source top-grasp와 destination release pose, workspace/table/IK preflight.
- `executor`: left-arm-only hover, approach, gripper, lift, release, physical retreat; 단계별 abort 정책.
- `preview`: RGB overlay, camera/base XYZ, confidence, 안정성, 실행 전 모든 joint-plan을 명시한다.

## 수용 테스트

1. 한국어 명령이 `pick_query`/`place_query`로 정확히 구조화된다.
2. synthetic RGB-D bbox에서 known camera/base XYZ를 재현하며 depth=0·복수 후보·불안정 frame을 거부한다.
3. source/destination 모두에 대해 workspace/table/IK failure가 실행 전에 거부된다.
4. executor의 상태 순서가 hover→approach→grasp→lift→destination→release→retreat이며 어느 단계 실패도 retreat/abort로 끝난다.
5. 실제 하드웨어는 preview에서 overlay 및 3D 좌표를 먼저 검증하고, 알려진 펜과 박스로 low-speed single trial을 수행한다. 성공 기준은 물체를 지정된 박스에 놓고 테이블·팔 충돌 없이 retreat하는 것이다.
