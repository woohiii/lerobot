# 수건 접기 SmolVLA 진행 기록

작성일: 2026-09-11

## 현재까지 확인한 내용

- 정책 계열은 SmolVLA이다.
- `outputs/train/smolvla_towel_public/checkpoints/last`는 `CSI-Agent/foldtowel_merged` 데이터로 50,000 step 학습된 체크포인트다.
- `outputs/train/smolvla_towel_own_cam/checkpoints/last`는 `local/towel_half_fold_bimanual_depth_state`를 사용해 15,000 step 학습됐다.
- `outputs/train/smolvla_towel_half_fold/checkpoints/last`는 같은 로컬 데이터셋을 사용해 50,000 step 학습됐다.
- 실행 결과 로봇이 수건을 바라보고 거의 움직이지 않는 문제가 보고됐다.

## 확인된 로컬 데이터셋

### 기존 학습과 호환되는 데이터셋

경로:

`/home/youngchan/.cache/huggingface/lerobot/local/towel_half_fold_bimanual_depth_state`

- 20 episodes
- 20,804 frames
- 12차원 action
- 28차원 `observation.state`
- `left_wrist`, `right_wrist` 영상
- 현재 `smolvla_towel_own_cam` 및 `smolvla_towel_half_fold` 체크포인트의 입력 구조와 일치

### 새로 발견된 추가 데이터

대표 경로:

`/home/youngchan/.cache/huggingface/lerobot/local/towel_half_fold_bimanual_batch2_20260909_142225`

- 8 episodes
- 8,605 frames
- 12차원 action
- 12차원 `observation.state`
- `left_wrist`, `right_wrist`, `astra_depth` 영상

이 데이터는 기존 체크포인트와 바로 합치면 안 된다. 기존 정책은 28차원 state와 wrist 카메라만 기대하지만, 추가 데이터는 12차원 state와 depth 카메라를 포함한다. 카메라 키와 state 차원을 통일하는 변환 또는 동일한 입력 구조로 다시 녹화하는 작업이 먼저 필요하다.

## SmolVLA/XLeRobot 문서 적용 시 주의점

참고 문서의 예시는 양팔 SO-101과 카메라 3대(`front_cam`, `hand_cam`, `side_cam`) 기준이다. 현재 데이터는 `left_wrist`, `right_wrist` 중심이므로 문서 명령어를 그대로 복사하지 않고 현재 체크포인트의 입력 키와 로봇 action 차원을 유지해야 한다.

## 현재 판단

추론 step 수를 늘리는 것보다 먼저 action 출력과 실행 매핑을 확인해야 한다. 동일한 카메라 입력에서 10~20회 예측했을 때 action이 거의 변하지 않으면 학습/정규화 문제이고, action은 변하지만 로봇이 안 움직이면 관절 순서·action scaling·제어 루프 문제일 가능성이 높다.

추가 데이터로 fine-tuning하려면 다음 두 경로 중 하나를 선택한다.

1. 기존 체크포인트 입력 구조에 맞춰 28차원 state와 wrist 카메라만 사용하는 데이터셋을 새로 정리한다.
2. batch2 데이터를 기준으로 state·카메라 입력 구조를 통일한 뒤 SmolVLA를 다시 학습한다.

기존 체크포인트를 초기 가중치로 사용할 때는 `--resume=true`로 학습 상태를 이어가기보다, `pretrained_model`을 `--policy.path`로 지정하고 새 `output_dir`에서 별도 실험을 만드는 편이 적절하다.

