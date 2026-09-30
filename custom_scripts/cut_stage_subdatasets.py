"""Cut verified stage boundaries into balanced150 replica-group datasets.

The cutter produces four separate LeRobotDatasets while preserving validation splits, so each stage can be trained as
its own short-horizon ACT policy instead of one monolithic policy over a ~120s 3-stage task
(Phase 1 of the towel-fold success-rate plan).

Per-episode stage segments come from manifest.json's split_frame_offsets. Episodes flagged
incomplete_demo (e.g. ep51: only unfolding was ever recorded) only contribute their present
stage(s) and are skipped for the missing ones. Condition-2 episodes (already half-folded at
the start) have no "unfold" stage at all -- stage_names in the manifest reflects this per
episode, so no hardcoded condition ranges are needed here.

Usage:
    uv run python custom_scripts/cut_stage_subdatasets.py                 # all 3 stages
    uv run python custom_scripts/cut_stage_subdatasets.py --stage unfold  # just one
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from duplicate_towel_dataset import (
    DERIVED_FEATURES,
    build_replication_plan,
    condition_for_episode,
    frame_for_writer,
)

from lerobot.datasets.lerobot_dataset import LeRobotDataset

SOURCE_REPO_ID = "Woohi123/towel_fold_v1_balanced150"
REVIEW_DIR = Path("custom_scripts/fold_split_review")
MANIFEST_PATH = REVIEW_DIR / "manifest.json"

STAGE_TASKS = {
    "unfold": "Unfold the towel until it lies flat",
    "reorient_half_fold": "Fix the towel's orientation and fold it in half",
    "half_fold": "Fold the flattened towel in half",
    "gate_fold": "Fold both sides of the towel in like a gate fold",
    # 합성 단계: 펼쳐진 수건 반접기 시작 ~ 대문접기 끝까지 한 에피소드로 연결
    "half_to_gate_fold": "Fold the flat towel in half, then fold both sides in like a gate fold",
}
EVAL_SOURCE_FRACTION = 0.2


@dataclass(frozen=True)
class StageSample:
    """One stage segment and its original, pre-replication episode identity."""

    source_episode: int
    replica_index: int
    start_offset: int
    end_offset: int
    split: str


@dataclass(frozen=True)
class StagePlan:
    """Ordered stage segments; validation segments are always last."""

    stage: str
    samples: tuple[StageSample, ...]
    eval_split: float


def _validation_sources(samples: list[StageSample]) -> set[int]:
    """Hold out a deterministic global 20% of original source episodes."""
    sources_by_condition: dict[str, list[int]] = {}
    for source_episode in sorted({sample.source_episode for sample in samples}):
        sources_by_condition.setdefault(condition_for_episode(source_episode), []).append(source_episode)

    total_sources = sum(len(sources) for sources in sources_by_condition.values())
    target = round(total_sources * EVAL_SOURCE_FRACTION)
    if target < 1 or target >= total_sources:
        raise ValueError("unable to create a non-empty grouped split")
    allocations = {
        condition: math.floor(len(sources) * EVAL_SOURCE_FRACTION)
        for condition, sources in sources_by_condition.items()
    }
    remainder = target - sum(allocations.values())
    ranked = sorted(
        sources_by_condition,
        key=lambda condition: (
            -(len(sources_by_condition[condition]) * EVAL_SOURCE_FRACTION - allocations[condition]),
            condition,
        ),
    )
    for condition in ranked[:remainder]:
        allocations[condition] += 1

    selected: set[int] = set()
    for condition, sources in sources_by_condition.items():
        if allocations[condition]:
            selected.update(sources[-allocations[condition] :])
    return selected


def _load_reviewed_manifest() -> dict:
    manifest = json.loads(MANIFEST_PATH.read_text())
    incomplete = [episode for episode, info in manifest.items() if info.get("needs_review") or not info.get("manual")]
    if incomplete:
        raise ValueError(f"stage manifest has unreviewed episodes: {', '.join(incomplete)}")
    return manifest


def build_balanced_stage_plan(stage: str, *, unique_sources: bool = False) -> StagePlan:
    """Expand balanced150 deterministically, optionally retaining one replica per source."""
    if stage not in STAGE_TASKS:
        raise ValueError(f"unknown stage {stage!r}")

    manifest = _load_reviewed_manifest()
    source_episodes = build_replication_plan(
        [condition_for_episode(index) for index in range(72)], target_per_condition=50
    )
    unassigned = []
    for replica_index, source_episode in enumerate(source_episodes):
        segment = stage_segment(manifest[str(source_episode)], stage)
        if segment is not None:
            unassigned.append(
                StageSample(source_episode, replica_index, segment[0], segment[1], split="")
            )

    if unique_sources:
        first_sample_by_source: dict[int, StageSample] = {}
        for sample in unassigned:
            first_sample_by_source.setdefault(sample.source_episode, sample)
        unassigned = list(first_sample_by_source.values())

    # Assign once across the original collection, before removing absent stages.
    eval_sources = _validation_sources(
        [StageSample(int(index), 0, 0, info["episode_length"], "") for index, info in manifest.items()]
    )
    samples = tuple(
        sorted(
            (
                StageSample(
                    sample.source_episode,
                    sample.replica_index,
                    sample.start_offset,
                    sample.end_offset,
                    "eval" if sample.source_episode in eval_sources else "train",
                )
                for sample in unassigned
            ),
            key=lambda sample: sample.split == "eval",
        )
    )
    return StagePlan(stage=stage, samples=samples, eval_split=sum(s.split == "eval" for s in samples) / len(samples))


def _episode_ranges(dataset: LeRobotDataset) -> dict[int, range]:
    return {
        int(episode["episode_index"]): range(int(episode["dataset_from_index"]), int(episode["dataset_to_index"]))
        for episode in dataset.meta.episodes
    }


def stage_segment(info: dict, stage: str) -> tuple[int, int] | None:
    """Return (start_offset, end_offset) within the episode for `stage`, or None if absent."""
    if stage == "half_to_gate_fold":
        names = info["stage_names"]
        if "half_fold" not in names or "gate_fold" not in names:
            return None
        bounds = [0, *info["split_frame_offsets"], info["episode_length"]]
        return bounds[names.index("half_fold")], bounds[names.index("gate_fold") + 1]
    if stage not in info["stage_names"]:
        return None
    stage_pos = info["stage_names"].index(stage)
    bounds = [0, *info["split_frame_offsets"], info["episode_length"]]
    start, end = bounds[stage_pos], bounds[stage_pos + 1]
    if not 0 <= start < end <= info["episode_length"]:
        raise ValueError(f"invalid {stage!r} boundary: {start}:{end}")
    return start, end


def plan_summary(plan: StagePlan) -> dict:
    """Return JSON-safe provenance and boundary evidence without writing a dataset."""
    split_counts = Counter(sample.split for sample in plan.samples)
    return {
        "stage": plan.stage,
        "episodes": len(plan.samples),
        "source_episodes": len({sample.source_episode for sample in plan.samples}),
        "frames": sum(sample.end_offset - sample.start_offset for sample in plan.samples),
        "split_counts": dict(split_counts),
        "eval_split": plan.eval_split,
        "samples": [asdict(sample) for sample in plan.samples],
    }


def cut_stage(source_root: Path, output_root: Path, stage: str, *, unique_sources: bool = False) -> None:
    """Write one balanced150 stage dataset with whole-source validation groups."""
    plan = build_balanced_stage_plan(stage, unique_sources=unique_sources)
    manifest_path = output_root / "stage_split_manifest.json"
    provenance = {
        "stage": stage,
        "eval_split": plan.eval_split,
        "eval_source_fraction": EVAL_SOURCE_FRACTION,
        "unique_sources": unique_sources,
        "source_root": str(source_root.resolve()),
        "samples": [asdict(sample) for sample in plan.samples],
    }
    if output_root.exists():
        if not manifest_path.is_file():
            raise ValueError("existing output has no stage_split_manifest.json; choose a new output path")
        previous = json.loads(manifest_path.read_text())
        if any(previous.get(key) != value for key, value in provenance.items()):
            raise ValueError("existing stage manifest differs from requested plan; choose a new output path")
    if not (source_root / "meta/info.json").is_file():
        raise ValueError(f"local source metadata is missing: {source_root}")

    source = LeRobotDataset(SOURCE_REPO_ID, root=source_root, return_uint8=True, tolerance_s=1e-3)
    ranges = _episode_ranges(source)
    reviewed = _load_reviewed_manifest()
    if source.meta.total_episodes != 150:
        raise ValueError("this replication plan requires the local balanced150 dataset")
    for sample in plan.samples:
        if len(ranges[sample.replica_index]) != reviewed[str(sample.source_episode)]["episode_length"]:
            raise ValueError(f"source episode length differs from reviewed manifest: {sample.source_episode}")
    features = {key: value for key, value in source.meta.features.items() if key not in DERIVED_FEATURES}
    dataset_suffix = "_unique" if unique_sources else ""
    repo_id = f"Woohi123/towel_fold_v1_balanced150_stage_{stage}{dataset_suffix}"

    if output_root.exists():
        output = LeRobotDataset.resume(repo_id=repo_id, root=output_root, batch_encoding_size=1)
        already_done = output.meta.total_episodes
        print(f"stage {stage!r}: resuming, {already_done}/{len(plan.samples)} episodes already present")
        samples = plan.samples[already_done:]
        start_at = already_done + 1
    else:
        output = LeRobotDataset.create(
            repo_id=repo_id,
            root=output_root,
            fps=source.fps,
            features=features,
            robot_type=source.meta.robot_type,
            use_videos=True,
            batch_encoding_size=1,
        )
        samples = plan.samples
        start_at = 1
    # Persist provenance before any frame writes so a partial dataset cannot be
    # resumed with different source identities or train/validation assignments.
    manifest_path.write_text(json.dumps(provenance | {"complete": False}, indent=2) + "\n")

    task = STAGE_TASKS[stage]
    total = len(samples) + start_at - 1
    try:
        for output_index, sample in enumerate(samples, start=start_at):
            rng = ranges[sample.replica_index]
            for offset in range(sample.start_offset, sample.end_offset):
                source_frame = source[rng.start + offset]
                output.add_frame(frame_for_writer(source_frame, set(features)) | {"task": task})
            output.save_episode()
            print(f"stage {stage!r} [{output_index}/{total}] source episode {sample.source_episode} "
                  f"frames {sample.start_offset}:{sample.end_offset} "
                  f"({sample.end_offset - sample.start_offset} frames, {sample.split})")
    finally:
        output.finalize()
    manifest_path.write_text(json.dumps(provenance | {"complete": True}, indent=2) + "\n")


def main() -> None:
    """Create all balanced150 stage datasets or one selected stage."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-root", type=Path,
                         default=Path("/home/youngchan/.cache/huggingface/lerobot/Woohi123/towel_fold_v1_balanced150"))
    parser.add_argument("--output-root-prefix", type=Path,
                         default=Path("/home/youngchan/.cache/huggingface/lerobot/Woohi123/towel_fold_v1_balanced150_stage_"))
    parser.add_argument("--stage", choices=list(STAGE_TASKS),
                         default=None, help="cut just this stage; default cuts all stages present in the manifest")
    parser.add_argument("--unique-sources", action="store_true",
                        help="write one deterministic segment per original source episode")
    parser.add_argument("--plan-only", action="store_true",
                        help="print verified sample boundaries and provenance as JSON without writing data")
    args = parser.parse_args()

    manifest = _load_reviewed_manifest()
    stages = [args.stage] if args.stage else sorted({s for info in manifest.values() for s in info["stage_names"]})

    for stage in stages:
        if args.plan_only:
            print(json.dumps(plan_summary(build_balanced_stage_plan(stage, unique_sources=args.unique_sources)), indent=2))
            continue
        output_root = Path(f"{args.output_root_prefix}{stage}")
        output_root = Path(f"{output_root}_unique") if args.unique_sources else output_root
        cut_stage(args.source_root, output_root, stage, unique_sources=args.unique_sources)


if __name__ == "__main__":
    main()
