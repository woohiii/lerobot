# Handoff: trash_to_bin IK+IL 통합 (2026-09-08 세션 인계)

브랜치: `mvp/gemini-click-grasp` (repo `/home/youngchan/lerobot`). 커밋 안 됨 — 아래 파일들은 아직 unstaged.

## 오늘 이미 완료 (검증됨, 그대로 유지)

1. `custom_scripts/vision_pick_place/task_trash_to_bin/calibrate_ir_homography.py` (신규)
   — 핸드가이드 토크-OFF IR 호모그래피 캘리브레이션. `ir_homography_candidate.json`만 출력, 자동 승격 없음.
2. `custom_scripts/vision_pick_place/task_trash_to_bin/test_calibrate_ir_homography.py` (신규)
   — `uv run python3 custom_scripts/vision_pick_place/task_trash_to_bin/test_calibrate_ir_homography.py` → PASS (0 failing)
3. `custom_scripts/vision_pick_place/ir_wrist_hybrid_trash_to_bin.py`
   — `TASK_DIR`를 `task_trash_to_bin`으로 재배선 완료 (1줄). dry-run이 ImportError 없이 "missing published frame"에서만 막힘.
   회귀 확인 완료: `task_red_cube_to_bin_new_gripper/test_il_grasp_skill.py` 4/4 PASS, `task_trash_to_bin/test_click_grasp_trash.py` 5/5 PASS.

## 지금 진행 중이던 작업 (미완료) — 기본 정책 체크포인트 이슈

**문제**: `ir_wrist_hybrid_trash_to_bin.py` 32번째 줄:
```python
DEFAULT_POLICY = ROOT / "outputs/train/diffusion_red_cube_to_bin/checkpoints/030000/pretrained_model"
```
이건 "빨간 큐브→통" 태스크로 학습된 체크포인트인데 스크립트는 지금 trash_to_bin용으로 재배선되어 있어, IL 그랩 스킬이 엉뚱한 태스크 정책을 기본값으로 씀. `validate_policy_contract()`는 입출력 shape만 검증하고 태스크 의미는 검증 못 함.

**조사 결과 (완료)**:
- `outputs/train/`에는 `diffusion_red_cube_to_bin`과 `smolvla_towel_half_fold` 두 개만 존재 — **trash_to_bin 전용 체크포인트는 아직 없음**.
- 참고 정: `task_trash_to_bin/click_grasp_trash.py`는 이미 이 문제를 다른 방식으로 처리하고 있음 — `--grasp-strategy il`일 때 `--policy-path`가 없으면 `parser.error("--grasp-strategy il requires --policy-path")`로 명시적 요구 (109, 115-116번 줄). **같은 패턴을 `ir_wrist_hybrid_trash_to_bin.py`에도 적용하는 게 기존 코드베이스 관례와 일치함.**

**다음 세션에서 할 일 (아직 미적용)**:
- 원래 계획: DEFAULT_POLICY를 그대로 두고 preflight 경고 로그만 추가할지, 아니면 `click_grasp_trash.py`처럼 `--policy-path` 필수화(breaking change)로 갈지 결정 필요 — **사용자에게 확인 후 진행** (breaking change는 직접 적용하지 말고 제안만 하라고 이전에 지시받음).
- 권장(발견한 기존 패턴과 일치): `--policy-path`를 required로 바꾸고 `DEFAULT_POLICY`/`--policy-path` default 제거. 최소 변경.
- 변경 후 검증:
  ```bash
  uv run python3 -m py_compile custom_scripts/vision_pick_place/ir_wrist_hybrid_trash_to_bin.py
  uv run python3 custom_scripts/vision_pick_place/ir_wrist_hybrid_trash_to_bin.py --dry-run \
    --ir-homography custom_scripts/vision_pick_place/task_trash_to_bin/ir_homography_candidate.json
  # (candidate json 없으면 test_calibrate_ir_homography.py로 먼저 생성하거나 스킵)
  ```

## 이번 세션에서 막힌 것 — codex MCP 비정상

`mcp__codex__codex` / `codex-reply` 6/6 연속 동일 오류로 완전 무력화:
```
bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted
```
권한 상승 재시도(`danger-full-access`, `read-only`, `workspace-write` 전부)도 `approval request failed`로 거절. **이건 프롬프트 문제가 아니라 이 환경의 codex 컨테이너 자체가 깨진 상태**로 판단됨. 새 세션에서 codex를 다시 부를 때 이 증상이 재현되면 재시도로 해결 안 될 가능성이 높음 — 바로 직접 구현/서브에이전트 위임으로 전환 권장.

## 범위 밖 (이번에도 다음에도 착수 금지, 사용자 승인 필요)

- 실제 하드웨어에서 `calibrate_ir_homography.py` 실행 + `ir_homography_candidate.json` → `ir_homography.json` 수동 승격 (사람이 물리적으로 로봇 만져야 함).
- 왼팔 캘리브레이션 (미검증 상태로 방치 중, 별도 후속).

## 원래 승인된 계획 원본

`/home/youngchan/.claude/plans/gleaming-stargazing-pixel.md`

---

# 세션 인계 추가 (2026-09-08, RGB/Depth 정렬 + 클릭창 분할 세션)

브랜치 동일. 이 세션은 위 IR 하이브리드 작업과 별개 — click_grasp_trash.py + astra_s_depth_hub.py 쪽 작업.

## 표준 작업 방식 (다음 세션도 유지)
- 구현은 전부 `mcp__codex__codex-reply` (threadId `01a07ea1-1640-74c3-894e-1c18aca44809`)에 위임. 검증만 직접(`git status --short` 범위 확인 + `py_compile`).
- 새 코드 주석/설명은 한국어.
- 절대 건드리지 않는 파일: `task_red_cube_to_bin/`, `task_red_cube_to_bin_new_gripper/` 전체.
- 작업 허용 범위: `task_trash_to_bin/` + `astra_s_depth_hub.py`, `astra_s_stream_supervisor.py` + 공유 `homography.json`.

## 이번 세션 완료 (검증됨)

1. **오른팔 토크 해제** — 일회성, 확인됨.
2. **RGB/depth registration 진단 강화** (`astra_s_depth_hub.py`)
   - `IMAGE_REGISTRATION_DEPTH_TO_COLOR`는 원래부터 켜져 있었음(재확인 완료).
   - `get_image_registration_mode()`로 실제 활성 여부 확인 후 로그 + `/tmp/vsp_astra_depth_registration.json`에 `{"registered": bool}` 기록.
   - `is_image_registration_mode_supported()` 체크 없이 무조건 set 하던 부분 제거, 실패 시 예외 로그 남기게 수정.
   - **결론(코드 주석/로그에 명시됨)**: registration 이미 정상 동작 중. RGB가 depth보다 FOV 넓어 가장자리 안 맞는 건 registration과 무관한 하드웨어 한계 — 나중에 registration 껐다 켰다 헛수고 금지.
3. **카메라 허브 미리보기 창**: 분리했다 다시 병합 — 지금은 "Astra S RGB + Depth" 한 창에 좌우 분할(hstack)로 뜸. 사용자 육안 확인 완료.
4. **FOV 불일치 대응**: depth 없는 RGB 영역은 어둡게 마스킹 + "NO DEPTH" 문구 표시 (`perception.py`의 `depth_valid_mask()`).
5. **click_grasp_trash.py 클릭창 분할**: 왼쪽=RGB(NO DEPTH 마스킹 유지)/오른쪽=depth 컬러맵, 한 창에 hstack. 클릭은 왼쪽 패널만 유효 처리, 오른쪽 클릭은 무시+로그(`depth 패널 클릭은 무시합니다`). py_compile 통과, 범위 확인 완료.
6. **안전 작업영역 재보정 준비**: 기존 `homography.json`의 robot_points 범위(x=0.18-0.28, y=-0.08-0.08)가 실측 가동범위 대비 너무 좁다고 사용자가 직접 진단. 사용자가 손으로 팔 움직여 실측한 xyz:
   - 오른팔: (0.3693, 0.0120, -0.0613), (0.3680, -0.1205, -0.0891), (0.4022, 0.0574, -0.0590), (0.3779, 0.1418, -0.0595)
   - 왼팔(참고용, 왼팔 자체는 이 태스크 범위 밖): (0.4276, -0.0049, -0.0203), (0.3797, -0.1858, -0.0207), (0.4254, 0.0278, -0.0214)
   - 위 실측값 기반으로 `task_trash_to_bin/calibrate_camera.py` (신규, `--manual` 전용) 생성 완료. `CALIB_POINTS_XY`를 x=0.20-0.35, y=-0.10-0.10로 확장(실측 최대치보다 여유 둠). **아직 하드웨어에서 실행 안 함 — homography.json 아직 옛날 좁은 범위 그대로.**

## 다음 세션에서 바로 할 일 (우선순위 순)

1. **카메라 프로세스 중복 실행 주의**: `astra_s_stream_supervisor.py`가 이미 `astra_s_depth_hub.py`를 자식으로 띄움. `astra_s_depth_hub.py` 단독 재실행하면 USB interface 충돌남 (`Failed to set USB interface!`). registration 로그 확인하려면 supervisor 콘솔(또는 `/tmp/vsp_astra_depth_registration.json`)만 보면 됨, 절대 두 번 실행 금지.
2. **재보정 실행** (하드웨어 필요, 사용자와 함께):
   ```bash
   cd custom_scripts/vision_pick_place/task_trash_to_bin
   uv run --active python calibrate_camera.py --manual
   ```
   성공하면 공유 `homography.json` 새 범위로 덮어씀.
3. **WORKSPACE_MARGIN_M 재검토** (보류 중, 아직 미결정): 재보정 후 `perception.py`의 `WORKSPACE_MARGIN_M = 0.08`이 실측 최대 가동범위(x≈0.40, y≈±0.14)를 살짝 넘길 수 있음 — codex 제안은 ~0.04로 낮추는 것. 재보정 + dry-hover 재검증 이후에 사용자와 결정.
4. **dry-hover 재검증**:
   ```bash
   uv run python custom_scripts/vision_pick_place/task_trash_to_bin/click_grasp_trash.py --port /dev/so101_follower --dry-hover
   ```
   이전엔 대부분 클릭이 "안전 작업영역 밖"으로 거부됐음 — 재보정 후 정상 통과하는지 확인. 클릭창 분할(왼쪽 RGB만 유효) 동작도 같이 확인.
5. **Stage 3 (실물 그랩+투입)**: dry-hover 통과 확인 후, `--dry-hover` 없이 실제 그랩 1회 테스트. depth 기반 높이 추정 정확도는 아직 실물로 한 번도 검증 안 됨(원래 사용자 우려사항) — 이 단계에서 같이 확인.

## 범위 밖 (승인 없이 착수 금지)
- 왼팔 캘리브레이션.
- `WORKSPACE_MARGIN_M` 값 변경 (재보정 후 사용자 확인 필요).
