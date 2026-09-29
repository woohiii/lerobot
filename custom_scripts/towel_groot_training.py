"""Validate the towel dataset and emit reproducible GR00T N1.7 training commands."""

from __future__ import annotations

import argparse
import json
import random
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DATASET_REPO_ID = "Woohi123/towel_fold_v1_balanced150"
DATASET_REVISION = "2b606ccb6b4eb45e1290447a1077504d1d56122a"
SPLIT_SEED = 42
CAMERAS = ("left_wrist", "right_wrist", "astra_rgb", "astra_depth_viz")
TASKS = (
    "Unfold a towel with a folded corner, fold it in half, then fold both sides in like a gate fold",
    "Unfold a towel with one edge folded, fold it in half, then fold both sides in like a gate fold",
    "Unfold a heavily wrinkled towel, fold it in half, then fold both sides in like a gate fold",
)
CONDITIONS = ("corner", "edge", "wrinkled")
SOURCE_RANGES = {"corner": range(0, 25), "edge": range(25, 50), "wrinkled": range(50, 72)}


@dataclass(frozen=True)
class DatasetContract:
    repo_id: str
    revision: str
    tasks: tuple[str, ...]


@dataclass(frozen=True)
class SplitEpisode:
    episode_index: int
    source_index: int
    condition: str


@dataclass(frozen=True)
class DatasetSplit:
    train: tuple[SplitEpisode, ...]
    validation: tuple[SplitEpisode, ...]

    @property
    def train_sources(self) -> frozenset[int]:
        return frozenset(item.source_index for item in self.train)

    @property
    def validation_sources(self) -> frozenset[int]:
        return frozenset(item.source_index for item in self.validation)

    @staticmethod
    def assert_no_leakage(train: tuple[SplitEpisode, ...], validation: tuple[SplitEpisode, ...]) -> None:
        if {item.source_index for item in train} & {item.source_index for item in validation}:
            raise ValueError("source identity leaks between train and validation")


def _metadata_from(value: dict[str, Any] | Path) -> dict[str, Any]:
    if isinstance(value, Path):
        info_path = value / "meta" / "info.json" if value.is_dir() else value
        try:
            metadata = json.loads(info_path.read_text())
        except FileNotFoundError as exc:
            raise ValueError(f"dataset metadata not found: {info_path}") from exc
        tasks_path = info_path.parent / "tasks.parquet"
        if "tasks" not in metadata and tasks_path.is_file():
            import pyarrow.parquet

            metadata["tasks"] = [entry["task"] for entry in pyarrow.parquet.read_table(tasks_path).to_pylist()]
        return metadata
    return value


def validate_dataset_contract(metadata_or_path: dict[str, Any] | Path) -> DatasetContract:
    """Fail closed unless local metadata matches the pinned bimanual dataset contract."""
    metadata = _metadata_from(metadata_or_path)
    expected_scalars = {"total_episodes": 150, "total_frames": 535475, "robot_type": "bi_so_follower"}
    for key, expected in expected_scalars.items():
        if metadata.get(key) != expected:
            raise ValueError(f"expected {key}={expected!r}, got {metadata.get(key)!r}")

    features = metadata.get("features", {})
    for name in ("observation.state", "action"):
        feature = features.get(name, {})
        if feature.get("shape") != [12] or not isinstance(feature.get("names"), list) or len(feature["names"]) != 12:
            raise ValueError(f"{name} requires 12 names and shape [12]")
        if len(set(feature["names"])) != 12:
            raise ValueError(f"{name} names must be unique")
    for camera in CAMERAS:
        if f"observation.images.{camera}" not in features:
            raise ValueError(f"missing required visual feature: {camera}")

    tasks = tuple(metadata.get("tasks", TASKS))
    if tasks != TASKS:
        raise ValueError("dataset tasks do not match the three pinned towel instructions")
    return DatasetContract(DATASET_REPO_ID, DATASET_REVISION, TASKS)


def _replication_plan() -> tuple[SplitEpisode, ...]:
    plan: list[SplitEpisode] = []
    for condition in CONDITIONS:
        sources = tuple(SOURCE_RANGES[condition])
        plan.extend(
            SplitEpisode(len(plan) + offset, source, condition)
            for offset, source in enumerate(sources[index % len(sources)] for index in range(50))
        )
    return tuple(plan)


def build_train_validation_split(seed: int = SPLIT_SEED) -> DatasetSplit:
    """Create 120 train and 15 held-out representatives without source leakage."""
    plan = _replication_plan()
    train: list[SplitEpisode] = []
    validation: list[SplitEpisode] = []
    rng = random.Random(seed)
    for condition in CONDITIONS:
        held_out = set(rng.sample(tuple(SOURCE_RANGES[condition]), 5))
        replicas = [item for item in plan if item.condition == condition]
        validation.extend(next(item for item in replicas if item.source_index == source) for source in sorted(held_out))
        eligible = [item for item in replicas if item.source_index not in held_out]
        # Wrinkled has 22 source identities, so removing five may leave only 38
        # materialized replicas.  Cycle only safe identities to retain the specified
        # forty training samples per condition without leaking a held-out source.
        train.extend(eligible[index % len(eligible)] for index in range(40))
    result = DatasetSplit(tuple(train), tuple(validation))
    result.assert_no_leakage(result.train, result.validation)
    if len(result.train) != 120 or len(result.validation) != 15:
        raise AssertionError("split must contain 120 train and 15 validation episodes")
    return result


def build_training_command(phase: str, output_dir: Path | None = None) -> list[str]:
    """Build a shell-safe, pinned smoke or full GR00T training invocation."""
    if phase not in {"smoke", "full"}:
        raise ValueError("phase must be 'smoke' or 'full'")
    split = build_train_validation_split()
    output_dir = output_dir or Path(f"outputs/towel_groot_{phase}")
    steps = 20 if phase == "smoke" else 60_000
    command = [
        "uv", "run", "--no-sync", "lerobot-train",
        f"--dataset.repo_id={DATASET_REPO_ID}", f"--dataset.revision={DATASET_REVISION}",
        f"--dataset.episodes={json.dumps([item.episode_index for item in (*split.train, *split.validation)], separators=(',', ':'))}",
        "--dataset.eval_split=0.1111111111111111", "--policy.type=groot",
        "--policy.base_model_path=nvidia/GR00T-N1.7-3B", "--policy.embodiment_tag=new_embodiment",
        "--policy.chunk_size=16", "--policy.n_action_steps=16", "--policy.use_relative_actions=true",
        '--policy.relative_exclude_joints=["left_gripper.pos","right_gripper.pos"]',
        "--policy.model_params_fp32=false", "--policy.use_bf16=true", "--batch_size=1",
        "--accelerator.gradient_accumulation.steps=8", "--seed=42", "--steps=" + str(steps),
        "--policy.push_to_hub=true", "--policy.private=true",
        "--policy.repo_id=Woohi123/towel_fold_v1_balanced150_groot_n17", f"--output_dir={output_dir}",
        f"--job_name=towel_groot_{phase}",
    ]
    if phase == "full":
        command.extend(("--save_checkpoint=true", "--save_freq=10000", "--eval_steps=2000"))
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--dataset-root", type=Path, required=True)
    command_parser = subparsers.add_parser("command")
    command_parser.add_argument("--phase", choices=("smoke", "full"), required=True)
    command_parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.subcommand == "validate":
        validate_dataset_contract(args.dataset_root)
        print("dataset contract valid")
    else:
        print(shlex.join(build_training_command(args.phase, args.output_dir)))


if __name__ == "__main__":
    main()
