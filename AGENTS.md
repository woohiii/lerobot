This file provides guidance to AI agents when working with code in this repository.

## 사용자 응답 원칙

- **페르소나**: 친절하고 실무적인 한국어 기술 파트너로 응답한다. 먼저 해결 결과를 짧고 분명하게 말하고, 필요한 경우에만 쉬운 표현으로 근거와 주의점을 덧붙인다.
- **검증 방법**: 사용자가 작업을 요청하면, 완료 응답에 사용자가 직접 결과를 확인할 수 있는 구체적인 검증 방법(실행 명령, 확인할 출력 또는 재현 절차)을 반드시 제공한다. 변경이 없거나 검증이 불가능한 경우에는 그 이유와 대신 확인할 수 있는 방법을 명시한다.

> **User-facing help → [`AGENT_GUIDE.md`](./AGENT_GUIDE.md)** (SO-101 setup, recording, picking a policy, training duration, eval — with copy-pasteable commands).

## Project Overview

LeRobot is a PyTorch-based library for real-world robotics, providing datasets, pretrained policies, and tools for training, evaluation, data collection, and robot control. It integrates with Hugging Face Hub for model/dataset sharing.

## Tech Stack

Python 3.12+ · PyTorch · Hugging Face (datasets, Hub, accelerate) · draccus (config/CLI) · Gymnasium (envs) · uv (package management)

## Development Setup

```bash
uv sync --locked                            # Base dependencies
uv sync --locked --extra test --extra dev   # Test + dev tools
uv sync --locked --extra all                # Everything
git lfs install && git lfs pull             # Test artifacts
```

## Key Commands

```bash
uv run pytest tests -svv --maxfail=10                 # All tests
DEVICE=cuda make test-end-to-end                      # All E2E tests
pre-commit run --all-files                           # Lint + format (ruff, typos, bandit, etc.)
```

## Architecture (`src/lerobot/`)

- **`scripts/`** — CLI entry points (`lerobot-train`, `lerobot-eval`, `lerobot-record`, etc.), mapped in `pyproject.toml [project.scripts]`.
- **`configs/`** — Dataclass configs parsed by draccus. `train.py` has `TrainPipelineConfig` (top-level). `policies.py` has `PreTrainedConfig` base. Polymorphism via `draccus.ChoiceRegistry` with `@register_subclass("name")` decorators.
- **`policies/`** — Each policy in its own subdir. All inherit `PreTrainedPolicy` (`nn.Module` + `HubMixin`) from `pretrained.py`. Factory with lazy imports in `factory.py`.
- **`processor/`** — Data transformation pipeline. `ProcessorStep` base with registry. `DataProcessorPipeline` / `PolicyProcessorPipeline` chain steps.
- **`datasets/`** — `LeRobotDataset` (episode-aware sampling + video decoding) and `LeRobotDatasetMetadata`.
- **`envs/`** — `EnvConfig` base in `configs.py`, factory in `factory.py`. Each env subclass defines `gym_kwargs` and `create_envs()`.
- **`robots/`, `motors/`, `cameras/`, `teleoperators/`** — Hardware abstraction layers.
- **`types.py`** and **`configs/types.py`** — Core type aliases and feature type definitions.

## Repository Structure (outside `src/`)

- **`tests/`** — Pytest suite organized by module. Fixtures in `tests/fixtures/`, mocks in `tests/mocks/`. Hardware tests use skip decorators from `tests/utils.py`. E2E tests via `Makefile` write to `tests/outputs/`.
- **`.github/workflows/`** — CI: `quality.yml` (pre-commit), `fast_tests.yml` (base deps, every PR), `full_tests.yml` (all extras + E2E + GPU, post-approval), `latest_deps_tests.yml` (daily lockfile upgrade), `security.yml` (TruffleHog), `release.yml` (PyPI publish on tags).
- **`docs/source/`** — HF documentation (`.mdx` files). Per-policy READMEs, hardware guides, tutorials. Built separately via `docs-requirements.txt` and CI workflows.
- **`examples/`** — End-user tutorials and scripts organized by use case (dataset creation, training, hardware setup).
- **`docker/`** — Dockerfiles for user (`Dockerfile.user`) and CI (`Dockerfile.internal`).
- **`benchmarks/`** — Performance benchmarking scripts.
- **Root files**: `pyproject.toml` (single source of truth for deps, build, tool config), `Makefile` (E2E test targets), `uv.lock`, `CONTRIBUTING.md` & `README.md` (general information).

## Notes

- **Mypy is gradual**: strict only for `lerobot.envs`, `lerobot.configs`, `lerobot.optim`, `lerobot.model`, `lerobot.cameras`, `lerobot.motors`, `lerobot.transport`. Add type annotations when modifying these modules.
- **Imports**: prefer top-level imports; relative (`from .sibling import X`) across sibling files within a module, absolute (`from lerobot.module import X`) across modules.
- **Optional dependencies**: many policies, envs, and robots are behind extras (e.g., `lerobot[aloha]`, see `pyproject.toml`). Guard optional imports with `TYPE_CHECKING or _foo_available` at module top + a `require_package(...)` check at use time. Reuse the `_foo_available` flags in `utils/import_utils.py`; don't call `is_package_available`.
- **Video decoding**: datasets can store observations as video files. `LeRobotDataset` handles frame extraction, but tests need ffmpeg installed.
- **Prioritize use of `uv run`** to execute Python commands (not raw `python` or `pip`).

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
