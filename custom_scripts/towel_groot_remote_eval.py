"""Safety-gated remote inference client for the bimanual towel-fold policy."""

from __future__ import annotations

import argparse
import logging
import threading
import time
from typing import Any

import numpy as np
import torch
from towel_groot_training import TASKS

TASK_BY_CONDITION = dict(zip(("corner", "edge", "wrinkled"), TASKS, strict=True))
CAMERA_SHAPES = {
    "left_wrist": (480, 640, 3),
    "right_wrist": (480, 640, 3),
    "astra_rgb": (240, 320, 3),
    "astra_depth_viz": (240, 320, 3),
}


def validate_action_chunk(actions: torch.Tensor | np.ndarray, max_horizon: int = 16) -> np.ndarray:
    """Require a finite action array with 12 joints and a bounded time horizon."""
    values = actions.detach().cpu().numpy() if isinstance(actions, torch.Tensor) else np.asarray(actions)
    if values.ndim != 2 or values.shape[1] != 12 or not 1 <= values.shape[0] <= max_horizon:
        raise ValueError(f"action chunk must have shape (1..{max_horizon}, 12), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("action chunk contains non-finite values")
    return values


def validate_observation(observation: dict[str, Any]) -> np.ndarray:
    """Validate the four pinned camera feeds and extract finite joint positions."""
    observation = normalize_camera_aliases(observation)
    cameras = {key: value for key, value in observation.items() if key in CAMERA_SHAPES}
    missing = set(CAMERA_SHAPES) - set(cameras)
    if missing:
        raise ValueError(f"missing required camera observation(s): {', '.join(sorted(missing))}")
    for key, shape in CAMERA_SHAPES.items():
        image = np.asarray(cameras[key])
        if image.shape != shape or image.dtype != np.uint8:
            raise ValueError(f"{key} must be uint8 image {shape}, got {image.dtype} {image.shape}")
    joint_keys = [key for key in observation if key.endswith(".pos")]
    if len(joint_keys) != 12:
        raise ValueError(f"observation must contain 12 joint positions, got {len(joint_keys)}")
    state = np.asarray([observation[key] for key in joint_keys], dtype=np.float64)
    if not np.isfinite(state).all():
        raise ValueError("observation contains non-finite joint positions")
    return state


def normalize_camera_aliases(observation: dict[str, Any]) -> dict[str, Any]:
    """Map the Orbbec camera's RGB/depth keys to the dataset's pinned names."""
    normalized = dict(observation)
    for source, target in (("astra", "astra_rgb"), ("astra_depth", "astra_depth_viz")):
        if source in normalized:
            if target in normalized:
                raise ValueError(f"camera alias collision: both {source} and {target} are present")
            normalized[target] = normalized.pop(source)
    return normalized


def validate_latency(generated_at: float, now: float | None = None, max_latency: float = 0.75) -> None:
    """Reject server actions whose wall-clock timestamp is outside the freshness window."""
    age = (time.time() if now is None else now) - generated_at
    if age < -0.25:
        raise ValueError(f"action timestamp is in the future ({age:.3f}s)")
    if age > max_latency:
        raise ValueError(f"action is stale ({age:.3f}s > {max_latency:.3f}s)")


def validate_relative_target(current: np.ndarray, target: np.ndarray, max_delta: float = 5.0) -> None:
    """Bound each action's per-joint displacement from the latest measured state."""
    current_values = np.asarray(current, dtype=np.float64)
    target_values = np.asarray(target, dtype=np.float64)
    if current_values.shape != (12,) or target_values.shape != (12,):
        raise ValueError("current state and target must both have 12 values")
    if not np.isfinite(current_values).all() or not np.isfinite(target_values).all():
        raise ValueError("relative target contains non-finite values")
    largest_delta = float(np.abs(target_values - current_values).max())
    if largest_delta > max_delta:
        raise ValueError(f"unsafe relative target ({largest_delta:.2f} > {max_delta:.2f})")


def send_action_safely(mode: str, send_action: Any, action: dict[str, float]) -> Any:
    """Call the hardware transport only in the explicitly selected execute mode."""
    if mode == "execute":
        return send_action(action)
    if mode != "dry-run":
        raise ValueError(f"unsupported evaluation mode: {mode}")
    return action


def trigger_emergency_stop(client: Any) -> None:
    """Disable torque on both bimanual buses and stop client control loops."""
    client.logger.critical("Keyboard emergency stop requested; disabling both arm torques")
    client.robot.left_arm.bus.disable_torque()
    client.robot.right_arm.bus.disable_torque()
    client.shutdown_event.set()


def _run_client(mode: str, max_actions: int, max_delta: float, argv: list[str]) -> None:
    import draccus

    from lerobot.async_inference.configs import RobotClientConfig
    from lerobot.async_inference.robot_client import RobotClient, visualize_action_queue_size
    from lerobot.cameras.orbbec import OrbbecCameraConfig  # noqa: F401
    from lerobot.utils.import_utils import register_third_party_plugins

    register_third_party_plugins()
    config = draccus.parse(RobotClientConfig, args=argv)

    class GuardedRobotClient(RobotClient):
        action_count = 0
        latest_state: np.ndarray | None = None

        def __init__(self, config: RobotClientConfig):
            super().__init__(config)
            # The stock Orbbec adapter calls streams `<camera>` and `<camera>_depth`;
            # the pinned dataset/model calls them `astra_rgb` and `astra_depth_viz`.
            feature_aliases = {
                "observation.images.astra": "observation.images.astra_rgb",
                "observation.images.astra_depth": "observation.images.astra_depth_viz",
            }
            for source, target in feature_aliases.items():
                if source in self.policy_config.lerobot_features:
                    feature = self.policy_config.lerobot_features.pop(source)
                    feature["shape"] = (*feature["shape"][:2], 3)
                    feature["info"] = {**feature.get("info", {}), "is_depth_map": False}
                    self.policy_config.lerobot_features[target] = feature
            original_get_observation = self.robot.get_observation

            def get_validated_observation():
                observation = normalize_camera_aliases(original_get_observation())
                self.latest_state = validate_observation(observation)
                return observation

            self.robot.get_observation = get_validated_observation

        def control_loop_observation(self, task: str, verbose: bool = False):
            return super().control_loop_observation(task, verbose)

        def control_loop_action(self, verbose: bool = False):
            from queue import Empty

            with self.action_queue_lock:
                timed_action = self.action_queue.get_nowait() if not self.action_queue.empty() else None
            if timed_action is None:
                raise Empty
            validate_latency(timed_action.get_timestamp())
            target = validate_action_chunk(timed_action.get_action().reshape(1, -1))[0]
            if self.latest_state is None:
                raise ValueError("refusing action before a validated observation")
            validate_relative_target(self.latest_state, target, max_delta=max_delta)
            action = self._action_tensor_to_action_dict(timed_action.get_action())
            if mode == "dry-run":
                self.logger.info(
                    "dry-run action #%s validated; hardware send skipped", timed_action.get_timestep()
                )
            result = send_action_safely(mode, self.robot.send_action, action)
            with self.latest_action_lock:
                self.latest_action = timed_action.get_timestep()
            self.action_count += 1
            if self.action_count >= max_actions:
                self.shutdown_event.set()
            return result

    client = GuardedRobotClient(config)

    def emergency_stop_listener() -> None:
        while client.running:
            try:
                if input("긴급 정지: e + Enter > ").strip().lower() == "e":
                    trigger_emergency_stop(client)
                    return
            except (EOFError, OSError):
                return

    try:
        if not client.start():
            raise RuntimeError("policy server handshake failed")
        receiver = threading.Thread(target=client.receive_actions, daemon=True)
        receiver.start()
        emergency_listener = threading.Thread(target=emergency_stop_listener, daemon=True)
        emergency_listener.start()
        client.control_loop(task=config.task, verbose=True)
    finally:
        client.stop()
        if "receiver" in locals():
            receiver.join(timeout=2)
        if config.debug_visualize_queue_size and client.action_queue_size:
            visualize_action_queue_size(client.action_queue_size)


def main() -> None:
    """Select a no-send check/dry-run or explicitly enabled hardware execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--self-check", action="store_true")
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--execute", action="store_true")
    parser.add_argument("--max-actions", type=int, default=1)
    parser.add_argument("--max-delta", type=float, default=5.0)
    args, remaining = parser.parse_known_args()
    if args.self_check:
        validate_action_chunk(np.zeros((1, 12), dtype=np.float32))
        print("remote-eval safety self-check passed; no robot or network access")
        return
    if not args.dry_run and not args.execute:
        parser.error("choose --self-check, --dry-run, or --execute")
    if args.max_actions <= 0 or args.max_delta <= 0:
        parser.error("--max-actions and --max-delta must be positive")
    logging.basicConfig(level=logging.INFO)
    _run_client("execute" if args.execute else "dry-run", args.max_actions, args.max_delta, remaining)


if __name__ == "__main__":
    main()
