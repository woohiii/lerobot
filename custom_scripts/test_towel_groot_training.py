"""Focused contract, split-provenance, and CLI tests for towel training."""

# ruff: noqa: D100, D103

import shlex

import pytest
from towel_groot_training import (
    DATASET_REPO_ID,
    DATASET_REVISION,
    TASKS,
    build_train_validation_split,
    build_training_command,
    validate_dataset_contract,
)


def _metadata():
    names = [f"joint_{index}" for index in range(12)]
    return {
        "total_episodes": 150,
        "total_frames": 535475,
        "robot_type": "bi_so_follower",
        "features": {
            "observation.state": {"shape": [12], "names": names},
            "action": {"shape": [12], "names": names},
            **{
                f"observation.images.{camera}": {"shape": [1]}
                for camera in ("left_wrist", "right_wrist", "astra_rgb", "astra_depth_viz")
            },
        },
        "tasks": list(TASKS),
    }


def test_validate_dataset_contract_accepts_the_pinned_bimanual_dataset():
    contract = validate_dataset_contract(_metadata())

    assert contract.repo_id == DATASET_REPO_ID
    assert contract.revision == DATASET_REVISION
    assert contract.tasks == TASKS


def test_validate_dataset_contract_rejects_wrong_action_names():
    metadata = _metadata()
    metadata["features"]["action"]["names"] = ["joint_0"] * 12

    with pytest.raises(ValueError, match="action names"):
        validate_dataset_contract(metadata)


def test_split_is_deterministic_balanced_and_source_leak_free():
    split = build_train_validation_split()

    assert 115 <= len(split.train) <= 120
    assert len(split.validation) == 15
    assert split == build_train_validation_split()
    assert set(split.train_sources).isdisjoint(split.validation_sources)
    train_indices = [item.episode_index for item in split.train]
    assert len(train_indices) == len(set(train_indices))
    train_counts = [item.condition for item in split.train]
    assert train_counts.count("corner") == 40
    assert train_counts.count("edge") == 40
    assert 35 <= train_counts.count("wrinkled") <= 40
    assert [item.condition for item in split.validation].count("wrinkled") == 5


def test_split_rejects_source_identity_leakage():
    split = build_train_validation_split()

    with pytest.raises(ValueError, match="source identity"):
        split.assert_no_leakage((*split.train, split.validation[0]), split.validation)


@pytest.mark.parametrize(
    ("phase", "steps", "required"),
    [
        ("smoke", "20", ("--output_dir=outputs/towel_groot_smoke",)),
        ("full", "60000", ("--save_freq=10000", "--eval_steps=2000")),
    ],
)
def test_command_has_pinned_groot_flags_and_is_shell_safe(phase, steps, required):
    command = build_training_command(phase=phase)
    rendered = shlex.join(command)

    assert shlex.split(rendered) == command
    assert f"--steps={steps}" in command
    assert f"--dataset.repo_id={DATASET_REPO_ID}" in command
    assert f"--dataset.revision={DATASET_REVISION}" in command
    for flag in (
        "--policy.type=groot",
        "--policy.embodiment_tag=new_embodiment",
        "--policy.chunk_size=16",
        "--policy.n_action_steps=16",
        "--policy.use_relative_actions=true",
        '--policy.relative_exclude_joints=["left_gripper.pos","right_gripper.pos"]',
        "--policy.model_params_fp32=false",
        "--policy.use_bf16=true",
        "--batch_size=1",
        "--accelerator.gradient_accumulation.steps=8",
        "--policy.push_to_hub=true",
        "--policy.private=true",
    ):
        assert flag in command
    for flag in required:
        assert flag in command
