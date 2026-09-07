# 왼팔 캘리브레이션 작업 인계 (2026-09-07)

새 Claude Code 세션에서 이 파일을 읽고 이어서 진행하세요. 사용자 지시: **파일 탐색/서치/토큰
소모 큰 작업은 전부 codex MCP 서브에이전트(mcp__codex__codex-reply)에 위임**. Claude는 relay만.

## 진행 중인 codex 스레드

- threadId: `01a07975-dde0-75e2-8039-9ee9949f3a81` (sandbox: danger-full-access)
- 이 threadId로 `mcp__codex__codex-reply`를 계속 호출해서 이어가면 됨 (새 codex 세션 만들지 말 것 -
  기존 진단 컨텍스트/좌표/파일 경로를 이미 다 알고 있음).
- **백그라운드 작업 진행 중**: task_id `k10fvibz8` (codex-reply 호출, follower_left.json 캘리브레이션
  fix 작업 중, 20분 이상 소요 중). 새 세션에서 `TaskOutput`으로 먼저 상태 확인
  (`block: false`로 non-blocking 체크 후, 필요하면 `block: true`로 대기). 이 task는 세션 종료 시
  살아남지 않는다는 경고가 있었으니, 새 세션에서 먼저 이 task_id가 여전히 유효한지/결과가 이미
  났는지 확인. 없어졌으면 같은 threadId로 codex-reply를 다시 호출해서 상태를 물어보면 됨.

## 하드웨어 상태 (변경 금지 사항)

- 오른팔 = `/dev/so101_follower` (serial 5B3D042390) = 이미 완전 캘리브레이션됨. **절대 건드리지 말 것.**
- 왼팔 = `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14029976-if00` (serial 5B14029976) =
  `config.py`의 `LEFT_OVERRIDES` 딕셔너리 대상. 오늘 처음 캘리브레이션 중.
- `config.py`의 `apply_side("left"|"right")` 컨벤션은 이미 물리적 방향과 일치하도록 통일됨 (변경 금지).

## 지금까지 확정되어 반영된 값

- `LEFT_OVERRIDES["TABLE_Z"] = -0.0042` (반영 완료)
- `LEFT_OVERRIDES["GRASP_TARGET_PX"] = (410.0, 152.0)` (반영 완료)
- `camera_hub.py` 수정 완료: 왼쪽 카메라 탐색을 `find_camera_index("USB 2.0 PC Cam", occurrence=1)`에서
  `find_camera_index("Innomaker-U20CAM-720P")`로 변경 (사용자가 육안으로 왼팔 그리퍼 시점 확인함,
  `/dev/video6`). 오른팔 탐색 로직은 그대로.
  - ⚠️ 참고: 기존 dry-run 테스트 하나가 "동일 모델 카메라 2대"를 전제로 해서 지금 하드웨어 구성에서
    깨질 수 있음 (요청 범위 밖이라 테스트 파일은 그대로 둠) — 나중에 필요하면 처리.

## 현재 막혀 있는 문제 (백그라운드 작업이 고치는 중)

`calibrate_grasp.py --port <왼팔 by-id 경로>`로 그리퍼 임계값을 측정했더니 빈 상태 최대=0.8%,
큐브 상태 최소=0.7%로 겹쳐서 판정 불가가 남. 원인 진단 결과:

- `SOArm101(port=...)`가 포트와 무관하게 항상 `id="follower"` (오른팔용) 캘리브레이션 파일을 사용 중.
- 오른팔용 `follower.json`의 그리퍼 raw 범위: 2006–3259
- 실제 저장되어 있는 왼팔 전용 `follower_left.json`의 그리퍼 raw 범위: 573–2080
- 즉 왼팔에서 open/close 퍼센트가 전부 잘못된 범위로 환산되고 있었음 → 지금까지 나온 0.8%/0.7%는
  전부 무효. 사용자도 육안으로 "그리퍼가 조금밖에 안 움직였다"고 확인해서 이 진단과 일치.
- 사용자가 이 수정을 **승인함** ("코덱스 수정제안 수락할테니 진행해"). codex에게 다음을 지시한 상태:
  - `follower_left.json`이 실제 왼팔 물리 팔과 맞는지 raw 위치 직접 읽어 검증
  - `calibrate_grasp.py`(및 필요한 다른 지점)가 왼팔 포트일 때 `follower_left` 캘리브레이션을
    쓰도록 최소 수정 (오른팔 경로는 절대 안 건드림)
  - diff 제시 + 사용자에게 `calibrate_grasp.py`를 처음부터 다시 실행하라고 안내

**새 세션에서 할 일**: 위 task_id/threadId로 결과 확인 → 사용자에게 diff 보여주고 재실행 안내 →
사용자가 새 결과(GRIPPER_EMPTY_CLOSED_PCT / GRASP_DETECT_MARGIN_PCT) 보내주면 codex에게 반영 지시.

## 남은 전체 절차 (왼팔만, 오른팔은 절대 건드리지 않음)

1. ~~TABLE_Z~~ 완료
2. ~~GRASP_TARGET_PX~~ 완료
3. **그리퍼 임계값 (진행 중, 캘리브레이션 ID 버그 수정 중)** — `calibrate_grasp.py --port <왼팔 포트>`
4. BIN_POSE_XYZ — `measure_bin_pose.py --port <왼팔 포트>`

4단계 모두 끝나면 `LEFT_OVERRIDES` 완성 → `orchestrator_bimanual.py` 또는
`click_grasp_bimanual.py --allow-unverified-left`로 왼팔 테스트 가능하다고 안내하고 마무리.

## 응답 스타일 관련 (세션 중 사용자가 설정함)

사용자가 `/caveman` full 모드를 켰음 — 기술 내용은 그대로, 군더더기 없이 압축된 어투로 응답할 것
(관사/필러 생략, 짧은 문장, 기술 용어·명령어·에러 메시지는 그대로 유지). 채팅에 남은 세션 설정이므로
새 세션에서는 사용자가 다시 지시하지 않는 한 기본 톤으로 시작해도 무방.
