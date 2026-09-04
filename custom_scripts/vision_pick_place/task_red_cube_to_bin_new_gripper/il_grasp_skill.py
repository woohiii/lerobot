"""Short (<=max_seconds) ACT skill that replaces FINE_SERVO+DESCEND+GRASP.

Per the design doc: free-space motion stays with IK/Legacy, and only the
contact-rich grasp is handed to a learned policy, on a HARD wall-clock
deadline. The policy's own idea of "done" is never trusted - after the
deadline this closes the gripper and verifies with the existing
gripper.is_grasp_success(), exactly like descend_and_grasp() does.

What is reused vs. what is glue (deliberately kept as thin as possible -
observation key ordering, normalization and ACT's temporal ensembling are
easy to get subtly wrong, and a wrong action here moves real hardware):

  reused, not reimplemented
    - lerobot.rollout.inference.sync.SyncInferenceEngine - the whole
      preprocessor -> policy.select_action -> postprocessor -> make_robot_action
      per-tick path, including autocast/inference_mode handling.
    - lerobot.policies.make_pre_post_processors / get_policy_class - loads the
      checkpoint's OWN saved normalization stats, so this can never disagree
      with how the policy was trained.
    - lerobot.datasets.aggregate_pipeline_dataset_features / create_initial_features
      + lerobot.utils.feature_utils.{combine_feature_dicts,build_dataset_frame} -
      same feature plumbing lerobot_record_home_reset.py and the rollout CLI use.
    - lerobot.rollout.context._align_state_feature_order / _resolve_action_key_order -
      private, but they are the only implementation of "make the robot's joint
      order match the checkpoint's joint order". Copying them here would be a
      second copy of a safety-relevant ordering rule; importing them means a
      fix upstream reaches this too. If they ever disappear upstream, the
      import error is loud and immediate (not a silent wrong ordering).
    - kinematics.SOArm101.send_joint_deg - so every action still goes through
      joint-limit clamping AND SOFollower's max_relative_target clamp.
    - gripper.close_gripper / gripper.is_grasp_success - unchanged verification.

  glue written here (unavoidable)
    - the wrist frame comes from `cap` (camera_hub.py publishes it; this
      process must not open the device itself - see main.py) instead of from
      robot.cameras, so it is injected into the observation dict by hand
      under the camera key the CHECKPOINT declares, and converted BGR->RGB
      (cap gives OpenCV BGR; recording via OpenCVCamera defaults to
      color_mode=RGB, so feeding BGR here would be an out-of-distribution
      input with no error message).
    - the deadline loop, the stall check, and the joint-name -> ALL_JOINTS
      array mapping for send_joint_deg.

The policy + processors are loaded once and cached module-level, keyed by
(policy_path, id(robot)): run()'s retry loop can call this up to
MAX_GRASP_ATTEMPTS times per task, and re-loading an ACT checkpoint each
attempt would add seconds of dead time between retries. Keyed by robot too
because Phase 3 runs two arms in one process.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from lerobot.configs import FeatureType, PreTrainedConfig
from lerobot.datasets import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.processor import make_default_processors
from lerobot.rollout.context import _align_state_feature_order, _resolve_action_key_order
from lerobot.rollout.inference.sync import SyncInferenceEngine
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts

import config
import gripper
from kinematics import CollisionDetected

if TYPE_CHECKING:
    from kinematics import SOArm101


DEFAULT_TASK = "grasp the object from the standardized pre-grasp pose"

# Unlike move_to_xyz's stall check, a stall here needs BOTH a large
# actual-vs-commanded lag AND the arm having essentially stopped moving.
# Lag alone is not enough: max_relative_target throttles every send_action, so
# a policy output far from the current pose produces a large lag while the arm
# is still moving perfectly freely - that exact mismatch already caused false
# CollisionDetected reports on this hardware (see move_to_xyz_converge's
# 2026-09-01 note). "Not moving at all" is what a real block looks like.
STALL_MOTION_EPS_DEG = 0.5


@dataclass
class GraspSkill:
    """Everything one policy+robot pairing needs per tick. Built by
    load_skill(); the dry-run test builds one by hand with a fake engine."""

    engine: Any  # SyncInferenceEngine (or a fake, in tests)
    dataset_features: dict
    ordered_action_keys: list[str]
    camera_keys: list[str] = field(default_factory=list)


_SKILL_CACHE: dict[tuple[str, int], GraspSkill] = {}


def _policy_camera_shapes(policy_cfg) -> dict[str, tuple[int, int, int]]:
    """Camera keys the CHECKPOINT expects, as hardware-side (H, W, C) shapes.

    Taken from the policy config rather than from robot.cameras because at
    task time this process deliberately owns no camera device (camera_hub.py
    does) - robot.observation_features has no image entry at all, so the
    dataset features have to be told about the wrist camera from the only
    other place that knows about it: the checkpoint."""
    shapes = {}
    for key, ft in policy_cfg.input_features.items():
        if ft.type is not FeatureType.VISUAL:
            continue
        c, h, w = ft.shape  # policy features are channel-first
        shapes[key.removeprefix(f"{OBS_STR}.images.")] = (h, w, c)
    return shapes


def load_skill(arm: SOArm101, policy_path: str, task: str = DEFAULT_TASK, device: str | None = None) -> GraspSkill:
    """Load the ACT checkpoint + its own pre/post-processors, and work out the
    observation/action feature plumbing for this robot. Cached - see module
    docstring."""
    cache_key = (str(policy_path), id(arm.robot))
    if cache_key in _SKILL_CACHE:
        return _SKILL_CACHE[cache_key]

    if device is None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.pretrained_path = policy_path
    policy = get_policy_class(policy_cfg.type).from_pretrained(policy_path, config=policy_cfg)
    policy = policy.to(device)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_path,
        pretrained_revision=policy_cfg.pretrained_revision,
        preprocessor_overrides={"device_processor": {"device": device}},
    )

    robot = arm.robot
    policy_action_names = getattr(policy_cfg, "action_feature_names", None)
    observation_features_hw = {
        k: v
        for k, v in robot.observation_features.items()
        if isinstance(v, tuple) or (v is float and k.endswith((".pos", ".vel")))
    }
    observation_features_hw = _align_state_feature_order(observation_features_hw, policy_action_names)
    camera_shapes = _policy_camera_shapes(policy_cfg)
    for cam_key, shape in camera_shapes.items():
        observation_features_hw.setdefault(cam_key, shape)
    action_features_hw = {k: v for k, v in robot.action_features.items() if k.endswith((".pos", ".vel"))}

    teleop_action_processor, _, robot_observation_processor = make_default_processors()
    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=action_features_hw),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=observation_features_hw),
            use_videos=True,
        ),
    )
    ordered_action_keys = _resolve_action_key_order(
        list(policy_action_names) if policy_action_names else None,
        list(dataset_features[ACTION]["names"]),
    )

    engine = SyncInferenceEngine(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        dataset_features=dataset_features,
        ordered_action_keys=ordered_action_keys,
        task=task,
        device=device,
        robot_type=robot.name,
    )
    engine.start()
    skill = GraspSkill(
        engine=engine,
        dataset_features=dataset_features,
        ordered_action_keys=ordered_action_keys,
        camera_keys=list(camera_shapes),
    )
    _SKILL_CACHE[cache_key] = skill
    return skill


def run_grasp_skill(
    arm: SOArm101,
    cap,
    policy_path: str | None = None,
    max_seconds: float = 5.0,
    fps: float = 30.0,
    skill: GraspSkill | None = None,
) -> bool:
    """Run the ACT grasp skill for at most `max_seconds`, then close the
    gripper and verify the grasp the same way descend_and_grasp() does.

    Returns True only if gripper.is_grasp_success() says so - the policy's own
    termination is never trusted. Raises CollisionDetected on a real stall, so
    task_state_machine.run()'s existing retry loop handles it unchanged.

    `skill` is an escape hatch for the dry-run test (a pre-built bundle with a
    fake engine); normal callers pass policy_path and get the cached one."""
    if skill is None:
        if policy_path is None:
            raise ValueError("run_grasp_skill needs policy_path (or a pre-built skill).")
        skill = load_skill(arm, policy_path)

    # Clear ACT's temporal-ensemble / action-chunk state so a retry starts
    # clean instead of continuing the previous attempt's chunk.
    skill.engine.reset()

    period_s = 1.0 / fps
    deadline = time.perf_counter() + max_seconds
    tick = 0
    stall_count = 0
    last_good = arm.get_joint_deg()
    prev_actual = last_good

    # HARD deadline: whatever the policy outputs, the loop cannot outlive it.
    while time.perf_counter() < deadline:
        loop_start = time.perf_counter()
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.01)
            continue

        obs = arm.robot.get_observation()
        for cam_key in skill.camera_keys:
            obs[cam_key] = frame[:, :, ::-1]  # BGR (cv2) -> RGB (what training saw)
        obs_frame = build_dataset_frame(skill.dataset_features, obs, prefix=OBS_STR)

        action = skill.engine.get_action(obs_frame)
        if action is None:
            continue
        # strict=True: a length mismatch here would silently drop or misalign joint
        # targets on real hardware, so fail loudly on the first tick instead.
        action_dict = dict(zip(skill.ordered_action_keys, [float(v) for v in action], strict=True))

        current = arm.get_joint_deg()
        target = np.array(
            [action_dict.get(f"{j}.pos", current[i]) for i, j in enumerate(config.ALL_JOINTS)], dtype=float
        )
        arm.send_joint_deg(target)  # joint-limit clamp + max_relative_target, both kept
        tick += 1

        if tick % config.STALL_CHECK_EVERY == 0:
            actual = arm.get_joint_deg()
            n_arm = len(config.ARM_JOINTS)
            lag = float(np.max(np.abs(actual[:n_arm] - target[:n_arm])))
            moved = float(np.max(np.abs(actual[:n_arm] - prev_actual[:n_arm])))
            prev_actual = actual
            if lag > config.STALL_THRESHOLD_DEG and moved < STALL_MOTION_EPS_DEG:
                stall_count += 1
            else:
                stall_count, last_good = 0, actual
            if stall_count >= config.STALL_CONSECUTIVE:
                arm.send_joint_deg(last_good)
                time.sleep(0.3)
                raise CollisionDetected(
                    f"il_grasp_skill aborted: joint lag {lag:.1f}deg with no motion for "
                    f"{config.STALL_CONSECUTIVE} consecutive checks - retreated to last known-good pose."
                )

        elapsed = time.perf_counter() - loop_start
        if elapsed < period_s:
            time.sleep(period_s - elapsed)

    # Deadline reached. Verify with the gripper's own resting position, exactly
    # like descend_and_grasp() - close_gripper() is a no-op re-command if the
    # policy already closed on the cube, and the only real check either way.
    print(f"[IL_GRASP] {max_seconds:.1f}s 데드라인 종료 ({tick} ticks) - 그리퍼 닫고 검증")
    final_pct = gripper.close_gripper(arm)
    time.sleep(0.3)
    grasped = gripper.is_grasp_success(final_pct)
    print(f"[IL_GRASP] 최종 그리퍼 위치 {final_pct:.1f}% -> {'집힘' if grasped else '못 집음'}")
    return bool(grasped)
