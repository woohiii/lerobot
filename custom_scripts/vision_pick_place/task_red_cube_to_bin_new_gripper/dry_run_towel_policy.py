"""Hardware-free I/O validation for the trained bimanual towel-fold SmolVLA.

This script only loads a local checkpoint and creates in-memory tensors.  It
does not construct a robot, open a serial port, or access a camera device.

Run:
    uv run python3 custom_scripts/vision_pick_place/task_red_cube_to_bin_new_gripper/dry_run_towel_policy.py
"""

from __future__ import annotations

from pathlib import Path

import torch

# Importing the concrete config registers the checkpoint's ``smolvla`` type
# before PreTrainedConfig deserializes config.json.
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: F401
from lerobot.utils.constants import ACTION, OBS_STATE


CHECKPOINT = (
    Path(__file__).resolve().parents[3]
    / "outputs/train/smolvla_towel_half_fold/checkpoints/050000/pretrained_model"
)
IMAGE_KEYS = ("observation.images.left_wrist", "observation.images.right_wrist")
EXPECTED_STATE_DIM = 28
EXPECTED_ACTION_DIM = 12
EXPECTED_IMAGE_SHAPE = (3, 480, 640)


def main() -> None:
    assert CHECKPOINT.is_dir(), f"checkpoint not found: {CHECKPOINT}"
    policy_cfg = PreTrainedConfig.from_pretrained(CHECKPOINT)
    assert policy_cfg.type == "smolvla", f"expected smolvla checkpoint, got {policy_cfg.type!r}"
    assert tuple(policy_cfg.input_features[OBS_STATE].shape) == (EXPECTED_STATE_DIM,)
    assert tuple(policy_cfg.output_features[ACTION].shape) == (EXPECTED_ACTION_DIM,)
    for key in IMAGE_KEYS:
        assert tuple(policy_cfg.input_features[key].shape) == EXPECTED_IMAGE_SHAPE, (
            f"{key} shape differs from checkpoint contract: {policy_cfg.input_features[key].shape}"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy_cfg.pretrained_path = str(CHECKPOINT)
    policy = get_policy_class(policy_cfg.type).from_pretrained(str(CHECKPOINT), config=policy_cfg).to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(CHECKPOINT),
        pretrained_revision=policy_cfg.pretrained_revision,
        preprocessor_overrides={"device_processor": {"device": device}},
    )

    # These unbatched tensors deliberately match the saved CHW feature shapes.
    # The saved to_batch processor adds the batch dimension and the saved
    # normalizer applies the checkpoint's STATE mean/std statistics.
    observation = {
        OBS_STATE: torch.zeros(EXPECTED_STATE_DIM, dtype=torch.float32),
        IMAGE_KEYS[0]: torch.zeros(EXPECTED_IMAGE_SHAPE, dtype=torch.float32),
        IMAGE_KEYS[1]: torch.zeros(EXPECTED_IMAGE_SHAPE, dtype=torch.float32),
        "task": "fold the towel in half",
    }
    assert observation[OBS_STATE].dtype is torch.float32
    for key in IMAGE_KEYS:
        assert observation[key].dtype is torch.float32
        assert tuple(observation[key].shape) == EXPECTED_IMAGE_SHAPE

    processed = preprocessor(observation)
    assert tuple(processed[OBS_STATE].shape) == (1, EXPECTED_STATE_DIM)
    for key in IMAGE_KEYS:
        assert tuple(processed[key].shape) == (1, *EXPECTED_IMAGE_SHAPE)
        assert processed[key].dtype is torch.float32

    with torch.inference_mode():
        normalized_action = policy.select_action(processed)
        action = postprocessor(normalized_action)

    assert tuple(normalized_action.shape) == (1, EXPECTED_ACTION_DIM), normalized_action.shape
    assert tuple(action.shape) == (1, EXPECTED_ACTION_DIM), action.shape
    assert torch.isfinite(action).all(), "policy output contains NaN or infinity"

    print(f"checkpoint: {CHECKPOINT}")
    print(f"device: {device}")
    print(f"processed state shape: {tuple(processed[OBS_STATE].shape)}")
    print(f"processed image shapes: {', '.join(f'{key}={tuple(processed[key].shape)}' for key in IMAGE_KEYS)}")
    print(f"action shape: {tuple(action.shape)}")
    print(f"action sample: {action[0, :5].detach().cpu().tolist()}")
    print("PASS: checkpoint model I/O contract validated without hardware access")


if __name__ == "__main__":
    main()
