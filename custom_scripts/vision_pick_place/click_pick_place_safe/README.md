# SO-101 클릭 기반 안전 계획 단계

`plan_click_pick_place.py`는 RGB와 Astra S 깊이 스냅샷만 읽고, 로봇 포트·리더암·팔로워암을 열지 않습니다. 두 큐브 변의 중간점과 놓을 위치를 클릭하면 다음 순서를 `latest_plan.json`에 만듭니다.

1. 현재 관절값을 홈 시드로 사용(선택 입력)
2. 큐브 위 안전 높이 → 하강 → 그리퍼 닫기
3. 4 cm 상승 및 파지 유지
4. 목표 위 안전 높이 → 하강 → 열기 → 후퇴

```bash
uv run python custom_scripts/vision_pick_place/click_pick_place_safe/plan_click_pick_place.py \
  --home-deg '0,0,0,0,0'
```

`--home-deg`의 값은 나중에 안전하게 읽어 기록한 실제 팔로워 관절값으로 바꿉니다. 이 프로그램은 값을 읽지도, 팔을 움직이지도 않습니다. `--down-roll-deg=-90`은 IK 후보일 뿐 실제로 손목/그리퍼가 작업대를 바라보는지 확인되기 전에는 실행에 사용할 수 없습니다.

출력 JSON의 `grasp_descend_model_downward_cosine`은 URDF 모델에서 그리퍼 축이 하방을 향하는 정도입니다. `1.0`이 완전한 하방이며, 파지 단계는 `0.85` 이상을 계획 통과 기준으로 기록합니다. 이는 모델 검증일 뿐 실제 장비 검증은 아닙니다.

NVIDIA API는 클릭·깊이·IK 계산에는 필요하지 않습니다. 향후 임의 물체의 이름 기반 탐지에만 `NVIDIA_API_KEY` 환경 변수를 사용하며, 키를 파일이나 코드에 넣지 않습니다.
