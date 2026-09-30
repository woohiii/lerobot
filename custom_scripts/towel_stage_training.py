"""Validate towel stage datasets and build reproducible ACT training commands.

The command intentionally uses online image augmentation. This preserves the
recorded demonstrations while improving robustness to lighting, focus, and
small camera-placement changes. Geometric transforms are deliberately small;
left/right flips are not valid for a bilateral joint-action dataset.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
from dataclasses import dataclass
from pathlib import Path

REQUIRED_CAMERAS = (
    "left_wrist",
    "right_wrist",
    "astra_rgb",
    "astra_depth_viz",
)
STAGES = ("unfold", "reorient_half_fold", "half_fold", "gate_fold", "half_to_gate_fold")

SAFE_IMAGE_TRANSFORMS = {
    "brightness": {"type": "ColorJitter", "kwargs": {"brightness": [0.85, 1.15]}},
    "contrast": {"type": "ColorJitter", "kwargs": {"contrast": [0.85, 1.15]}},
    "saturation": {"type": "ColorJitter", "kwargs": {"saturation": [0.8, 1.2]}},
    "hue": {"type": "ColorJitter", "kwargs": {"hue": [-0.03, 0.03]}},
    "sharpness": {"type": "SharpnessJitter", "kwargs": {"sharpness": [0.7, 1.3]}},
    "affine": {
        "type": "RandomAffine",
        "kwargs": {"degrees": [-3.0, 3.0], "translate": [0.03, 0.03]},
    },
}


@dataclass(frozen=True)
class DatasetContract:
    """Validated dimensions and camera names required by the stage ACT policy."""

    episodes: int
    state_dim: int
    action_dim: int
    cameras: tuple[str, ...]


def _feature_shape(features: dict, name: str) -> tuple[int, ...]:
    try:
        shape = features[name]["shape"]
    except KeyError as exc:
        raise ValueError(f"missing required feature: {name}") from exc
    return tuple(int(value) for value in shape)


def validate_dataset_contract(root: Path) -> DatasetContract:
    """Validate the ACT input/output contract from a local LeRobot dataset."""
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise ValueError(f"dataset metadata not found: {info_path}")
    info = json.loads(info_path.read_text())
    features = info.get("features", {})

    state_shape = _feature_shape(features, "observation.state")
    action_shape = _feature_shape(features, "action")
    if state_shape != (12,):
        raise ValueError(f"ACT requires a 12-dimensional observation.state, got {state_shape}")
    if action_shape != (12,):
        raise ValueError(f"ACT requires a 12-dimensional action, got {action_shape}")

    available_cameras = tuple(
        key.removeprefix("observation.images.")
        for key in features
        if key.startswith("observation.images.")
    )
    missing = tuple(camera for camera in REQUIRED_CAMERAS if camera not in available_cameras)
    if missing:
        raise ValueError(f"missing required cameras: {', '.join(missing)}")

    episodes = int(info.get("total_episodes", 0))
    if episodes < 1:
        raise ValueError("dataset must contain at least one episode")
    return DatasetContract(episodes, 12, 12, REQUIRED_CAMERAS)


def grouped_eval_split(dataset_root: Path, fallback: float = 0.1) -> float:
    """Read the whole-source validation ratio emitted by the stage cutter."""
    manifest_path = dataset_root / "stage_split_manifest.json"
    if not manifest_path.is_file():
        return fallback
    eval_split = float(json.loads(manifest_path.read_text())["eval_split"])
    if not 0.0 < eval_split < 1.0:
        raise ValueError(f"invalid grouped eval_split in {manifest_path}: {eval_split}")
    return eval_split


def smoke_episode_selection(dataset_root: Path, train_episodes: int) -> tuple[list[int], float]:
    """Keep the first train episodes and every held-out episode in manifest order.

    LeRobot holds out the final ``ceil(n * eval_split)`` episodes per task. The
    returned ratio is checked against that rule so a smoke run cannot silently
    include a held-out source family in training.
    """
    if train_episodes < 1:
        raise ValueError("smoke_train_episodes must be positive")
    manifest_path = dataset_root / "stage_split_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("smoke training requires stage_split_manifest.json")
    samples = json.loads(manifest_path.read_text()).get("samples")
    if not isinstance(samples, list):
        raise ValueError(f"missing samples in {manifest_path}")
    train_indices = [index for index, sample in enumerate(samples) if sample.get("split") == "train"]
    eval_indices = [index for index, sample in enumerate(samples) if sample.get("split") == "eval"]
    if len(train_indices) < train_episodes or not eval_indices:
        raise ValueError("stage split manifest cannot create the requested smoke split")
    selected = [*train_indices[:train_episodes], *eval_indices]
    eval_split = len(eval_indices) / len(selected)
    while math.ceil(len(selected) * eval_split) > len(eval_indices):
        eval_split = math.nextafter(eval_split, 0.0)
    if math.ceil(len(selected) * eval_split) != len(eval_indices):
        raise ValueError("smoke split cannot be represented by LeRobot's ceil eval split rule")
    return selected, eval_split


def build_training_command(
    *,
    stage: str,
    dataset_root: Path,
    output_dir: Path,
    pretrained_path: str | None = None,
    policy: str = "act",
    steps: int = 20_000,
    batch_size: int = 4,
    save_freq: int = 5000,
    eval_split: float | None = None,
    image_augmentation: bool = False,
    smoke_train_episodes: int | None = None,
) -> list[str]:
    """Build a validated, augmentation-enabled ``lerobot-train`` command."""
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    if steps < 1:
        raise ValueError("steps must be positive")
    if policy not in {"act", "smolvla"}:
        raise ValueError("policy must be 'act' or 'smolvla'")
    if policy == "smolvla" and pretrained_path is None:
        pretrained_path = "lerobot/smolvla_base"
    eval_split = grouped_eval_split(dataset_root) if eval_split is None else eval_split
    selected_episodes: list[int] | None = None
    if smoke_train_episodes is not None:
        selected_episodes, eval_split = smoke_episode_selection(dataset_root, smoke_train_episodes)
    if not 0.0 < eval_split < 1.0:
        raise ValueError("eval_split must be between zero and one")
    validate_dataset_contract(dataset_root)

    command = [
        "uv",
        "run",
        "--no-sync",
        "lerobot-train",
        f"--dataset.repo_id=local/towel_fold_v1_balanced150_stage_{stage}",
        f"--dataset.root={dataset_root}",
        f"--dataset.eval_split={eval_split}",
        f"--policy.type={policy}",
        "--policy.push_to_hub=false",
        "--policy.device=cuda",
        "--policy.use_amp=true",
        f"--output_dir={output_dir}",
        f"--job_name={policy}_towel_stage_{stage}",
        f"--batch_size={batch_size}",
        "--num_workers=2",
        f"--steps={steps}",
        "--log_freq=1",
        f"--save_freq={save_freq}",
        "--eval_steps=10000",
    ]
    if image_augmentation:
        command.extend(
            [
                "--dataset.image_transforms.enable=true",
                "--dataset.image_transforms.max_num_transforms=2",
                f"--dataset.image_transforms.tfs={json.dumps(SAFE_IMAGE_TRANSFORMS, separators=(',', ':'))}",
            ]
        )
    if policy == "act":
        command.extend(["--policy.chunk_size=100", "--policy.n_action_steps=100"])
    if pretrained_path is not None:
        command.append(f"--policy.pretrained_path={pretrained_path}")
    if selected_episodes is not None:
        command.append(f"--dataset.episodes={json.dumps(selected_episodes, separators=(',', ':'))}")
    return command


def main() -> None:
    """Print one validated, copy-pasteable training command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pretrained-path")
    parser.add_argument("--policy", choices=("act", "smolvla"), default="act")
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--save-freq", type=int, default=5000)
    parser.add_argument("--eval-split", type=float)
    parser.add_argument("--image-augmentation", action="store_true")
    parser.add_argument("--smoke-train-episodes", type=int)
    args = parser.parse_args()
    print(shlex.join(build_training_command(**vars(args))))


if __name__ == "__main__":
    main()
