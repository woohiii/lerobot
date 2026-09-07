"""Safety-gated bimanual runtime for the local towel-half-fold SmolVLA checkpoint.

The runtime never opens a camera device: RGB frames are read only from the
existing camera_hub published PNG files.  Its 16 depth-state values are also
read from a named JSON mapping because this repository does not contain the
recording-time depth sampling geometry.  It refuses to run if that mapping is
missing or malformed rather than inventing depth features.

Use --self-check for a pure in-memory contract check.  --dry-run still connects
the supplied robots to exercise the existing firmware protection setup, but it
never sends a policy action.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

TASK_DIR = Path(__file__).resolve().parent
REPO_ROOT = TASK_DIR.parents[2]
sys.path.insert(0, str(TASK_DIR))

import config  # noqa: E402
import towel_precision_grasp  # noqa: E402
from kinematics import CollisionDetected, JointStallGuard, SOArm101, clamp_joint_deg  # noqa: E402

from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.policies import get_policy_class, make_pre_post_processors  # noqa: E402
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: F401, E402
from lerobot.robots.bi_so_follower import BiSOFollower, BiSOFollowerConfig  # noqa: E402
from lerobot.robots.so_follower import SOFollowerConfig  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402

CHECKPOINT = REPO_ROOT / "outputs/train/smolvla_towel_half_fold/checkpoints/050000/pretrained_model"
DATASET_INFO = (
    Path.home() / ".cache/huggingface/lerobot/local/towel_half_fold_bimanual_depth_state/meta/info.json"
)
IMAGE_KEYS = ("observation.images.left_wrist", "observation.images.right_wrist")
TASK_TEXT = "Fold the towel in half"
WINDOW = "Towel policy: e=emergency torque-off, q/ESC=stop"


def send_smoothed_policy_targets(
    left_arm: SOArm101,
    right_arm: SOArm101,
    previous_left: np.ndarray,
    previous_right: np.ndarray,
    left_target: np.ndarray,
    right_target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Bridge one policy update with three small goal-position sends.

    SOFollower still applies its existing 15-degree present-position-relative
    cap to every sub-send.  This only removes the discontinuity between two
    consecutive policy targets; it does not relax any safety limit.
    """
    left_target, right_target = clamp_joint_deg(left_target), clamp_joint_deg(right_target)
    for fraction in (1 / 3, 2 / 3, 1.0):
        left_arm.send_joint_deg(previous_left + (left_target - previous_left) * fraction)
        right_arm.send_joint_deg(previous_right + (right_target - previous_right) * fraction)
        time.sleep(0.03)
    return left_target, right_target


def load_feature_names() -> tuple[list[str], list[str]]:
    """Read the recorded dataset schema; never infer vector order from names."""
    assert DATASET_INFO.is_file(), f"missing training dataset metadata: {DATASET_INFO}"
    features = json.loads(DATASET_INFO.read_text())["features"]
    state_names = features[OBS_STATE]["names"]
    action_names = features[ACTION]["names"]
    assert isinstance(state_names, list) and isinstance(action_names, list)
    assert len(state_names) == 28, f"expected 28 state names, got {len(state_names)}"
    assert len(action_names) == 12, f"expected 12 action names, got {len(action_names)}"
    assert state_names[:12] == action_names, "state motor order must exactly equal action order"
    assert state_names[12:] == [f"depth_{i}" for i in range(16)], "unexpected depth-state order"
    return state_names, action_names


def self_check() -> None:
    state_names, action_names = load_feature_names()
    policy_cfg = PreTrainedConfig.from_pretrained(CHECKPOINT)
    assert policy_cfg.type == "smolvla"
    assert tuple(policy_cfg.input_features[OBS_STATE].shape) == (len(state_names),)
    assert tuple(policy_cfg.output_features[ACTION].shape) == (len(action_names),)
    for image_key in IMAGE_KEYS:
        assert tuple(policy_cfg.input_features[image_key].shape) == (3, 480, 640)
    expected_actions = [f"{side}_{joint}.pos" for side in ("left", "right") for joint in config.ALL_JOINTS]
    assert action_names == expected_actions, "dataset motor ordering differs from BiSOFollower ordering"
    state = np.zeros(len(state_names), dtype=np.float32)
    action = np.zeros(len(action_names), dtype=np.float32)
    assert state.shape == (28,) and action.shape == (12,)
    print("PASS: towel state/action schema matches checkpoint and BiSOFollower joint order")
    print(f"state names: {state_names}")
    print(f"action names: {action_names}")


def read_rgb(path: Path, image_key: str) -> torch.Tensor:
    frame_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise RuntimeError(f"missing/unreadable published frame for {image_key}: {path}")
    if frame_bgr.shape[:2] != (480, 640):
        raise RuntimeError(f"{image_key} must be 480x640, got {frame_bgr.shape[:2]}")
    # Training video stats are in [0, 1], RGB, channel-first.
    return torch.from_numpy(frame_bgr[:, :, ::-1].copy()).permute(2, 0, 1).float().div_(255.0)


def read_depth_state(path: Path) -> dict[str, float]:
    """Read the externally produced, explicitly named 16-value depth state.

    No local sampling/reprojection is attempted: its geometry was not saved in
    the dataset metadata, so synthesizing it would silently change the policy
    input distribution.
    """
    try:
        values = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read depth-state JSON {path}: {exc}") from exc
    expected = {f"depth_{i}" for i in range(16)}
    assert set(values) == expected, f"depth-state JSON keys must be exactly {sorted(expected)}"
    depth = {key: float(values[key]) for key in expected}
    assert all(np.isfinite(value) for value in depth.values()), "depth state contains NaN/inf"
    return depth


def make_observation(
    bi: BiSOFollower,
    state_names: list[str],
    left_frame_path: Path,
    right_frame_path: Path,
    depth_state_path: Path,
) -> dict:
    motor_observation = bi.get_observation()
    depth = read_depth_state(depth_state_path)
    state_values = [
        float(motor_observation[name]) if name.endswith(".pos") else depth[name] for name in state_names
    ]
    state = torch.tensor(state_values, dtype=torch.float32)
    assert tuple(state.shape) == (len(state_names),)
    return {
        OBS_STATE: state,
        IMAGE_KEYS[0]: read_rgb(left_frame_path, IMAGE_KEYS[0]),
        IMAGE_KEYS[1]: read_rgb(right_frame_path, IMAGE_KEYS[1]),
        "task": TASK_TEXT,
    }


def run_episode(
    *,
    bi: BiSOFollower,
    left_arm: SOArm101,
    right_arm: SOArm101,
    policy,
    preprocessor,
    postprocessor,
    state_names: list[str],
    action_names: list[str],
    left_frame_path: Path,
    right_frame_path: Path,
    depth_state_path: Path,
    dry_run: bool,
    action_steps: int,
) -> bool:
    """Run exactly one checkpoint action chunk, or log it in --dry-run mode."""
    left_guard = JointStallGuard.start(left_arm)
    right_guard = JointStallGuard.start(right_arm)
    policy.reset()
    previous_left = left_guard.last_good.copy()
    previous_right = right_guard.last_good.copy()
    emergency_stopped = False
    cv2.namedWindow(WINDOW)
    try:
        for tick in range(action_steps):
            observation = make_observation(
                bi, state_names, left_frame_path, right_frame_path, depth_state_path
            )
            processed = preprocessor(observation)
            with torch.inference_mode():
                action = postprocessor(policy.select_action(processed))
            assert tuple(action.shape) == (1, len(action_names)), action.shape
            action_values = action[0].detach().cpu().numpy()
            assert np.isfinite(action_values).all(), "policy produced NaN/inf"
            action_by_name = dict(zip(action_names, action_values, strict=True))
            left_target = np.array([action_by_name[f"left_{joint}.pos"] for joint in config.ALL_JOINTS])
            right_target = np.array([action_by_name[f"right_{joint}.pos"] for joint in config.ALL_JOINTS])

            # Show only file-backed frames. This does not open a camera device.
            cv2.imshow(WINDOW, cv2.imread(str(left_frame_path), cv2.IMREAD_COLOR))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("e"):
                left_arm.release_torque()
                right_arm.release_torque()
                emergency_stopped = True
                print("[towel] EMERGENCY STOP: both arm torques released")
                return False
            if key in (ord("q"), 27):
                print("[towel] operator stopped run; no home-return is issued")
                return False

            if dry_run:
                print(
                    f"[towel] dry-run tick={tick}: left={left_target.tolist()} right={right_target.tolist()}"
                )
                continue

            # Each sub-send retains SOArm101's joint-range clamp and
            # SOFollower's 15-degree relative-target clamp. The shared
            # JointStallGuard retains the existing 10-degree x 3 rule.
            previous_left, previous_right = send_smoothed_policy_targets(
                left_arm, right_arm, previous_left, previous_right, left_target, right_target
            )
            if (tick + 1) % config.STALL_CHECK_EVERY == 0:
                left_guard.check(left_target)
                right_guard.check(right_target)
        return True
    except CollisionDetected as exc:
        # Each guard already retreated its own arm. Hold the other arm at its
        # last good pose too, then stop this episode without a home move.
        left_arm.send_joint_deg(left_guard.last_good)
        right_arm.send_joint_deg(right_guard.last_good)
        print(f"[towel] collision/stall: {exc}; both arms held at last known-good poses")
        return False
    finally:
        cv2.destroyAllWindows()
        if emergency_stopped:
            print("[towel] e-stop state retained; home-return deliberately skipped")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-port")
    parser.add_argument("--right-port")
    parser.add_argument(
        "--max-episodes", type=int, default=1, help="number of 50-action policy chunks (default: 1)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="connect and infer, log policy targets, never send actions"
    )
    parser.add_argument(
        "--precision-grasp",
        action="store_true",
        help="before episode 1, IK-grasp two Astra-detected towel corners; abort on failure",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="validate checkpoint/dataset joint ordering without hardware",
    )
    parser.add_argument("--left-frame-path", type=Path, default=Path("/tmp/vsp_wrist_left.png"))
    parser.add_argument("--right-frame-path", type=Path, default=Path("/tmp/vsp_wrist.png"))
    parser.add_argument(
        "--astra-frame-path",
        type=Path,
        default=Path(config.ASTRA_RGB_FRAME_PATH),
        help="published external Astra table-view RGB frame for --precision-grasp",
    )
    parser.add_argument(
        "--depth-state-path", type=Path, help="JSON object with named depth_0 through depth_15 values"
    )
    args = parser.parse_args()

    if args.self_check:
        self_check()
        return 0
    if not args.left_port or not args.right_port:
        parser.error("--left-port and --right-port are required unless --self-check is used")
    if args.max_episodes < 1:
        parser.error("--max-episodes must be at least 1")
    if args.depth_state_path is None:
        parser.error(
            "--depth-state-path is required; the trained depth_0..depth_15 "
            "sampling geometry is not in this repo"
        )

    state_names, action_names = load_feature_names()
    policy_cfg = PreTrainedConfig.from_pretrained(CHECKPOINT)
    assert tuple(policy_cfg.input_features[OBS_STATE].shape) == (len(state_names),)
    assert tuple(policy_cfg.output_features[ACTION].shape) == (len(action_names),)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy_cfg.pretrained_path = str(CHECKPOINT)
    policy = (
        get_policy_class(policy_cfg.type)
        .from_pretrained(str(CHECKPOINT), config=policy_cfg)
        .to(device)
        .eval()
    )
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(CHECKPOINT),
        pretrained_revision=policy_cfg.pretrained_revision,
        preprocessor_overrides={"device_processor": {"device": device}},
    )

    bi = BiSOFollower(
        BiSOFollowerConfig(
            id="follower",
            left_arm_config=SOFollowerConfig(
                port=args.left_port, max_relative_target=config.MAX_RELATIVE_TARGET_DEG
            ),
            right_arm_config=SOFollowerConfig(
                port=args.right_port, max_relative_target=config.MAX_RELATIVE_TARGET_DEG
            ),
        )
    )
    bi.connect(calibrate=False)
    left_arm, right_arm = SOArm101(robot=bi.left_arm), SOArm101(robot=bi.right_arm)
    left_arm.connect()  # applies existing firmware torque/current protection
    right_arm.connect()
    home_xyz = {
        side: tuple(arm.kin.forward_kinematics(arm.get_joint_deg()[: len(config.ARM_JOINTS)])[:3, 3])
        for side, arm in (("left", left_arm), ("right", right_arm))
    }
    try:
        if args.precision_grasp:
            result = towel_precision_grasp.precision_grasp_from_frame_path(
                left_arm, right_arm, args.astra_frame_path, dry_run=args.dry_run
            )
            print(f"[towel] precision-grasp: {result.state.value}: {result.reason}")
            for plan in result.plans:
                print(f"[towel] precision plan {plan.side}: pixel={plan.pixel}, pregrasp={plan.pregrasp_xyz}")
            if not result.success:
                print("[towel] precision-grasp failed; refusing to fall back to open-loop IL")
                towel_precision_grasp.return_arms_home(left_arm, right_arm, home_xyz)
                return 1
            # A precision phase changes the physical initial state. Discard any
            # cached SmolVLA action chunk before the first policy observation.
            policy.reset()
        for episode in range(args.max_episodes):
            print(f"[towel] episode {episode + 1}/{args.max_episodes}, dry_run={args.dry_run}")
            if not run_episode(
                bi=bi,
                left_arm=left_arm,
                right_arm=right_arm,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                state_names=state_names,
                action_names=action_names,
                left_frame_path=args.left_frame_path,
                right_frame_path=args.right_frame_path,
                depth_state_path=args.depth_state_path,
                dry_run=args.dry_run,
                action_steps=policy_cfg.n_action_steps,
            ):
                return 1
    finally:
        bi.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
