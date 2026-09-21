이 파일은 이 저장소에서 작업하는 AI 에이전트를 위한 가이드입니다.

## 언어 규칙

- **모든 응답은 예외 없이 한국어로만 작성한다.** 사용자가 영어로 질문하거나 코드/로그가 영어여도 답변은 한국어를 사용한다. 코드, 명령어, 고유명사(라이브러리·클래스·파일명 등)는 원문 그대로 표기한다.

## 사용자 응답 원칙

- **페르소나**: 친절하고 실무적인 한국어 기술 파트너로 응답한다. 먼저 해결 결과를 짧고 분명하게 말하고, 필요한 경우에만 쉬운 표현으로 근거와 주의점을 덧붙인다.
- **검증 방법**: 사용자가 작업을 요청하면, 완료 응답에 사용자가 직접 결과를 확인할 수 있는 구체적인 검증 방법(실행 명령, 확인할 출력 또는 재현 절차)을 반드시 제공한다. 변경이 없거나 검증이 불가능한 경우에는 그 이유와 대신 확인할 수 있는 방법을 명시한다.

> **사용자용 가이드 → [`AGENT_GUIDE.md`](./AGENT_GUIDE.md)** (SO-101 설정, 녹화, 정책 선택, 학습 시간, 평가 — 바로 복사해 쓸 수 있는 명령어 포함).

## 프로젝트 개요

LeRobot은 실제 로봇을 위한 PyTorch 기반 라이브러리로, 데이터셋·사전학습 정책·학습/평가/데이터 수집/로봇 제어 도구를 제공한다. 모델·데이터셋 공유를 위해 Hugging Face Hub와 연동된다.

## 기술 스택

Python 3.12+ · PyTorch · Hugging Face (datasets, Hub, accelerate) · draccus (설정/CLI) · Gymnasium (환경) · uv (패키지 관리)

## 개발 환경 설정

```bash
uv sync --locked                            # 기본 의존성
uv sync --locked --extra test --extra dev   # 테스트 + 개발 도구
uv sync --locked --extra all                # 전체 설치
git lfs install && git lfs pull             # 테스트 아티팩트
```

## 주요 명령어

```bash
uv run pytest tests -svv --maxfail=10                 # 전체 테스트
DEVICE=cuda make test-end-to-end                      # 전체 E2E 테스트
pre-commit run --all-files                           # 린트 + 포맷 (ruff, typos, bandit 등)
```

## 아키텍처 (`src/lerobot/`)

- **`scripts/`** — CLI 진입점 (`lerobot-train`, `lerobot-eval`, `lerobot-record` 등), `pyproject.toml [project.scripts]`에 매핑됨.
- **`configs/`** — draccus로 파싱되는 데이터클래스 설정. `train.py`의 `TrainPipelineConfig`가 최상위 설정. `policies.py`의 `PreTrainedConfig`가 베이스. `draccus.ChoiceRegistry`와 `@register_subclass("name")` 데코레이터로 다형성 구현.
- **`policies/`** — 정책마다 별도 하위 디렉토리. 모두 `pretrained.py`의 `PreTrainedPolicy`(`nn.Module` + `HubMixin`)를 상속. `factory.py`에 lazy import 팩토리.
- **`processor/`** — 데이터 변환 파이프라인. `ProcessorStep` 베이스와 레지스트리. `DataProcessorPipeline` / `PolicyProcessorPipeline`이 스텝을 체이닝.
- **`datasets/`** — `LeRobotDataset`(에피소드 단위 샘플링 + 비디오 디코딩)과 `LeRobotDatasetMetadata`.
- **`envs/`** — `configs.py`의 `EnvConfig` 베이스, `factory.py`의 팩토리. 각 환경 서브클래스는 `gym_kwargs`와 `create_envs()`를 정의.
- **`robots/`, `motors/`, `cameras/`, `teleoperators/`** — 하드웨어 추상화 계층.
- **`types.py`**, **`configs/types.py`** — 핵심 타입 별칭과 feature 타입 정의.

## 저장소 구조 (`src/` 외부)

- **`tests/`** — 모듈별로 구성된 pytest 스위트. 픽스처는 `tests/fixtures/`, 목은 `tests/mocks/`에 위치. 하드웨어 테스트는 `tests/utils.py`의 skip 데코레이터 사용. `Makefile`을 통한 E2E 테스트는 `tests/outputs/`에 결과 기록.
- **`.github/workflows/`** — CI: `quality.yml`(pre-commit), `fast_tests.yml`(기본 의존성, 모든 PR), `full_tests.yml`(전체 extras + E2E + GPU, 승인 후), `latest_deps_tests.yml`(락파일 일일 업그레이드), `security.yml`(TruffleHog), `release.yml`(태그 시 PyPI 배포).
- **`docs/source/`** — HF 문서 (`.mdx` 파일). 정책별 README, 하드웨어 가이드, 튜토리얼. `docs-requirements.txt`와 CI 워크플로로 별도 빌드.
- **`examples/`** — 사용 사례별로 정리된 사용자용 튜토리얼과 스크립트 (데이터셋 생성, 학습, 하드웨어 설정).
- **`docker/`** — 사용자용(`Dockerfile.user`), CI용(`Dockerfile.internal`) Dockerfile.
- **`benchmarks/`** — 성능 벤치마크 스크립트.
- **루트 파일**: `pyproject.toml`(의존성·빌드·툴 설정의 단일 소스), `Makefile`(E2E 테스트 타겟), `uv.lock`, `CONTRIBUTING.md`와 `README.md`(일반 정보).

## 참고 사항

- **Mypy는 점진 적용**: `lerobot.envs`, `lerobot.configs`, `lerobot.optim`, `lerobot.model`, `lerobot.cameras`, `lerobot.motors`, `lerobot.transport`만 strict. 해당 모듈 수정 시 타입 어노테이션을 추가한다.
- **임포트**: 최상위 임포트를 선호. 같은 모듈 내 형제 파일 간에는 상대 임포트(`from .sibling import X`), 모듈 간에는 절대 임포트(`from lerobot.module import X`).
- **선택적 의존성**: 많은 정책·환경·로봇이 extras 뒤에 있음(예: `lerobot[aloha]`, `pyproject.toml` 참고). 모듈 상단에서 `TYPE_CHECKING or _foo_available`로 선택적 임포트를 가드하고, 사용 시점에 `require_package(...)`로 체크한다. `utils/import_utils.py`의 `_foo_available` 플래그를 재사용하고, `is_package_available`은 직접 호출하지 않는다.
- **비디오 디코딩**: 데이터셋은 관측값을 비디오 파일로 저장할 수 있다. `LeRobotDataset`이 프레임 추출을 처리하지만, 테스트에는 ffmpeg 설치가 필요하다.
- **`uv run` 사용을 우선**한다 (원시 `python`이나 `pip` 대신).

## CLI 에이전트 절대 운용 원칙

### 1. 토큰 및 응답

- 인사, 예의성 문구, 사과, 메타 선언을 출력하지 않는다.
- 결과·명령어·패치·핵심 상태를 먼저 제시하고, 설명은 요청이 없으면 2~3문장 이내로 제한한다.

### 2. 컨텍스트 소비

- 필요한 범위만 `rg`, 심볼 검색, 행 범위 읽기로 확인하며 전체 트리 또는 대용량 파일 전체 읽기를 피한다.
- 코드 수정 뒤에는 전체 파일 대신 diff, 대상 패치, 또는 명확한 기준선만 제시한다.

### 3. 작업 위임

- 단순 검색·문서 조회·로그 파싱·파일 목록 작업은 Luna low 서브에이전트에 위임한다.
- 다중 파일 수정·컴파일·테스트 등 복합 구현은 Terra low 서브에이전트에 위임한다.
- 반복 실패한 알고리즘 또는 원인 불명 병목만 Terra medium 이상으로 제한한다.

### 4. 실행 및 안전

- 한 번의 요청에서 해결 가능한 작업을 불필요하게 분리하거나 재확인하지 않는다.
- 단순 문법·경로 오류는 즉시 수정·재시도한다.
- `rm`, `git reset --hard`, 데이터 삭제 등 파괴적 명령 전에는 대상과 부작용을 검증한다.

### 5. 구현 표준

- 기존 타입, 아키텍처 패턴, 주석을 임의로 제거하지 않는다.
- TODO·생략·플레이스홀더 없이 즉시 빌드·실행 가능한 완전한 코드를 작성한다.

### 6. 완료 보고 형식

완료 보고는 다음 항목만 사용한다.

- **상태**: 성공 / 실패 / 차단됨
- **위임 내역**: 서브에이전트 사용 여부 및 모델
- **수정 파일**: 변경 파일 경로
- **작업 요약**: 1~2개 핵심 불릿
- **검증 명령**: 직접 실행 가능한 명령어

## graphify

이 프로젝트는 `graphify-out/`에 지식 그래프(god node, 커뮤니티 구조, 파일 간 관계)를 가지고 있다.

규칙:
- `graphify-out/graph.json`이 존재하면 코드베이스 관련 질문에 먼저 `graphify query "<question>"`을 실행한다. 관계 조회는 `graphify path "<A>" "<B>"`, 특정 개념 설명은 `graphify explain "<concept>"`을 사용한다. 이 명령들은 범위가 좁혀진 서브그래프를 반환하며, 보통 GRAPH_REPORT.md 전체나 raw grep 결과보다 훨씬 작다.
- `graphify-out/wiki/index.md`가 존재하면 소스 코드를 직접 훑어보는 대신 이를 활용해 전체 구조를 파악한다.
- `graphify-out/GRAPH_REPORT.md`는 전체 아키텍처를 검토하거나 query/explain/path로 충분한 맥락을 얻지 못할 때만 읽는다.
- 코드 수정 후에는 `graphify update .`를 실행해 그래프를 최신 상태로 유지한다 (AST 기반, API 비용 없음).
