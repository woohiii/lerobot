# SO-101 Astra S 로컬 VLM Pick-and-Place

이 디렉터리는 기존 vision pick-place 파일을 변경하지 않는 독립 파이프라인이다. 기존 GraspNet 경로와 함께 사용할 수 있으며, 기본 명령 경로는 Astra S RGB-D 스냅샷을 Ollama VLM으로 탐지한다.

## 안전 순서

1. `config.example.json`을 복사해 실측 intrinsic, 카메라→공통 base transform, 두 팔 workspace/port로 채운다.
2. `calibration_confirmed`는 reprojection 및 workspace 검증이 끝난 뒤에만 `true`로 바꾼다.
3. Ollama를 실행하고 VLM 모델을 내려받는다. 기본 모델은 `qwen2.5vl:7b`이며 `--ollama-model`로 변경할 수 있다.
4. 합성 입력 단위 테스트와 `--dry-run`만 통과시킨 뒤, 저속 단일 물체로 각 팔을 개별 검증한다.

## 로컬 VLM 실행

먼저 기존 Astra publisher를 실행해 `/tmp/vsp_astra_rgb.png`와 `/tmp/vsp_astra_depth_mm.npy`를 갱신한다. 그 다음 한 프레임을 저장하고 자연어 명령을 preview한다.

```bash
ollama serve
ollama pull qwen2.5vl:7b

PYTHONPATH=. uv run python -m custom_scripts.vision_pick_place.so101_graspnet_pick_and_place.local_cli \
  --config custom_scripts/vision_pick_place/so101_graspnet_pick_and_place/config.example.json \
  --rgb-path /tmp/vsp_astra_rgb.png \
  --depth-path /tmp/vsp_astra_depth_mm.npy \
  --command "빨간 블록을 파란 트레이에 가져다줘" \
  --dry-run
```

실행 전에는 출력된 source/destination box와 좌표를 확인한다. 실제 실행은 calibration을 확인한 설정 파일을 사용하고 `--execute --yes`를 명시해야 한다.

## 안전 불변식

- Ollama VLM bbox는 depth ROI 용도일 뿐 segmentation으로 간주하지 않는다.
- Astra depth가 RGB에 정합됐음을 별도로 검증한다.
- 중앙 금지 띠 또는 두 workspace가 겹치는 목표는 실행하지 않는다.
- 한 팔이 home/대기 상태인 것을 확인하기 전 다른 팔을 이동하지 않는다.
- 5-DOF SO-101은 GraspNet 6-DOF pose를 완전히 추종하지 않는다. 위치 우선 IK와 보수적 wrist orientation만 사용한다.
- gripper와 wrist Gemini 검증 모두 성공해야 bin place가 허용된다.

## 개발 검증

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest \
  custom_scripts/vision_pick_place/so101_graspnet_pick_and_place -q
uv run ruff check custom_scripts/vision_pick_place/so101_graspnet_pick_and_place
```

기본 pytest는 현재 workspace의 기존 `launch_testing` 플러그인과 pytest의 API 불일치 때문에 collection 전에 실패한다. 위 명령은 해당 전역 플러그인만 비활성화하고 신규 모듈을 검증한다.

## 비용 및 라이선스

Ollama VLM과 GraspNet-baseline은 로컬에서 실행되므로 요청 횟수별 API 요금이 없다. GraspNet-baseline과 선택한 모델의 라이선스는 상업 사용 전에 별도 확인한다.
