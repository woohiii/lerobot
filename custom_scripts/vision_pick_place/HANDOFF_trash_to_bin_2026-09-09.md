# SO-101 trash-to-bin: 2026-09-09 세션 정리

저장소: `/home/youngchan/lerobot`
브랜치: `mvp/gemini-click-grasp`
전제 문서: [HANDOFF_trash_to_bin_2026-09-08.md](./HANDOFF_trash_to_bin_2026-09-08.md)

## 오늘 해결한 것

### 1. `homography.json` 재보정 (블로커 #1)

기존 `CALIB_POINTS_XY`(x=0.20~0.35)가 오른팔의 실제 가동범위를 넘어서 캘리브레이션 3~4번 점이 계속 "충분히 도달하지 못했습니다"로 실패.

- `probe_reach.py`(스크래치패드, 카메라/클릭 없이 팔만 움직여 도달 오차 스캔)로 실측한 결과, `CALIB_HOVER_Z`(TABLE_Z+0.05 ≈ 0.0458)에서 **반경 ~0.27~0.29m에서 급격히 끊기는 하드 리밋** 확인 (그 너머는 z가 테이블로 sag).
- `CALIB_POINTS_XY`를 실측 안전범위(x:0.20~0.27, y:±0.10, 10점)로 교체.
- `cv2.findHomography`를 `method=0`(전체 최소자승, 이상치에 취약) → `cv2.RANSAC`으로 교체 — 사람이 클릭 잘못한 점(오클릭) 자동 배제.
- `mean_reprojection_error_m`을 RANSAC 인라이어 기준으로만 계산하도록 수정 (이상치 포함 평균은 오해를 부름).
- 결과: 인라이어 5/10점, 평균 재투영오차 2.8mm.

### 2. IR-IK+IL 하이브리드 dry-run (블로커 #2)

- `ir_wrist_hybrid_trash_to_bin.py --dry-run`은 `--policy-path`가 필수인데 trash_to_bin 전용 IL 체크포인트가 없음.
- `outputs/train/diffusion_red_cube_to_bin/checkpoints/last`가 `validate_policy_contract()` 형식(state (6,), wrist (3,480,640), action (6,))을 만족해 dry-run 검증용으로 사용 — preflight 통과 확인.
- `--ir-homography`는 `task_trash_to_bin/ir_homography_candidate.json`을 그대로 사용 (정식 `ir_homography.json` 승격은 여전히 보류, 사용자 승인 필요 항목).

### 3. `WORKSPACE_MARGIN_M` 재검토 (블로커 #3, 사용자 승인받고 진행)

- 재보정 후 좁아진 caliar 박스(x:0.20~0.25) + 기존 마진(0.08)을 더하면 실측 리밋(~0.27~0.29m)을 넘는 영역까지 "안전"으로 통과시키는 문제 발견.
- `perception.py`의 `is_xy_within_safe_workspace`를 taught-box+margin 방식에서 **베이스(0,0) 기준 반경(`config.MAX_REACH_XY_M`) 판정**으로 교체 — 캘리브레이션 점 개수/분포에 안 흔들리는 방식.

### 4. 안전영역 시각 오버레이 (`click_grasp_trash.py`)

- 뎁스 무효 영역: 빨간 대각 해칭으로 표시(요청 반영).
- 도달 불가 영역: 처음엔 픽셀 전수조사(순방향 호모그래피)로 구현 → 캘리브레이션 패치 밖에서 투영변환이 폭주해 원호가 아니라 화면 대부분이 붉게 뒤덮이는 문제 발생.
- **수정**: 로봇좌표계에서 원호(半径=`MAX_REACH_XY_M`, 각도=`homography.json`의 taught 점들로부터 실측된 범위, 현재 ±26.6°)를 만들고 **역호모그래피로 경계점만** 픽셀로 매핑 → `cv2.fillPoly`. 결과적으로 안전영역은 화면의 약 1.6%뿐 — 이건 버그가 아니라 이 팔이 이 호버 높이에서 실제로 갈 수 있는 영역이 카메라 시야보다 훨씬 좁다는 실측 사실.
- 사용자가 반원(180°, TCP 중심)을 요청해서 시도했으나, TCP(홈 위치, x≈0.17)를 중심으로 반경 0.27을 더하면 캘리브레이션 범위(x 최대 0.27) 밖으로 더 나가 역호모그래피가 폭주(-810px 등) — **반원은 이 팔 기구학상 물리적으로 불가능**하다고 판단, 원점(0,0) 기준 실측 부채꼴로 되돌림.

### 5. `SEARCH_HOVER_XYZ` 높이 불일치 발견 (중요)

- 실제 그랩 때 쓰는 탐색 호버 높이가 `z=0.13`이었는데, 이제까지의 모든 도달범위 측정은 `CALIB_HOVER_Z`(z≈0.0458)에서 한 것 — **서로 다른 높이 기준으로 안전영역을 계산하고 있었음**.
- `probe_reach_hover13.py`(스크래치패드)로 z=0.13에서 재측정: y=0 축으로는 x=0.30까지 되지만, **y가 조금만 벌어져도(±0.10 이상) x값 상관없이 전부 실패**(z가 5~6cm sag) — z=0.13은 좌우 스윙 여력이 거의 없는 높이였음.
- `config.SEARCH_HOVER_XYZ`를 `(0.23, 0.0, 0.13)` → `(0.23, 0.0, TABLE_Z + 0.05)`로 낮춰서 CALIB_HOVER_Z와 통일 — 이제 안전영역 계산/오버레이/실제 호버 모션이 전부 같은 높이 기준.

## 미해결 / 다음 세션 할 일

1. **y=±0.15까지 도달범위 확장 검증 중단됨**: `probe_reach_y15.py`(스크래치패드, z=CALIB_HOVER_Z에서 x=0.15~0.23, y=±0.15 스캔) 실행 중 사용자가 "이상하다"며 중단 — **재개 전 실제 팔 동작을 육안으로 재확인 필요** (어떤 이상 동작이었는지 다음 세션에서 사용자에게 먼저 확인).
2. **실물 그랩 시도에서 클릭 중심 대비 우측-하단 편향 확인됨**: 계통오차(고정 방향, 랜덤 아님) — 호모그래피 파라랙스(캘리브레이션은 그리퍼 팁 높이, 실제 물체는 테이블 위 두께+카메라 비스듬한 각도) 원인으로 추정. 보정 벡터 실측 절차 준비됐으나 아직 미실행.
3. **안전영역이 화면의 1.6%뿐이라 실사용엔 좁음**: y=±0.15 확장 검증(위 1번) 재개해서 부채꼴을 넓힐지, 카메라 프레임을 로봇 작업반경에 맞게 다시 잡을지 결정 필요.
4. **왼팔 캘리브레이션**: 여전히 범위 밖(미착수).
5. **`ir_homography_candidate.json` → `ir_homography.json` 정식 승격**: 여전히 사용자 승인 대기.

## 수정된 파일

- `custom_scripts/vision_pick_place/homography.json` (재보정)
- `custom_scripts/vision_pick_place/task_trash_to_bin/config.py` (`MAX_REACH_XY_M` 추가, `SEARCH_HOVER_XYZ` 높이 수정)
- `custom_scripts/vision_pick_place/task_trash_to_bin/perception.py` (`is_xy_within_safe_workspace` 반경 판정으로 교체, `reach_boundary_polygon_px` 신규)
- `custom_scripts/vision_pick_place/task_trash_to_bin/calibrate_camera.py` (`CALIB_POINTS_XY` 갱신, RANSAC, 인라이어 기준 오차 리포트)
- `custom_scripts/vision_pick_place/task_trash_to_bin/click_grasp_trash.py` (뎁스/도달불가 영역 빨간 해칭 오버레이)

## 검증 명령

```bash
# 캘리브레이션 재확인 (하드웨어 필요)
cd custom_scripts/vision_pick_place/task_trash_to_bin
~/lerobot_song_venv/bin/python calibrate_camera.py --manual

# dry-hover로 안전영역 오버레이 확인 (하드웨어 필요)
~/lerobot_song_venv/bin/python click_grasp_trash.py --port /dev/so101_follower --dry-hover
```
