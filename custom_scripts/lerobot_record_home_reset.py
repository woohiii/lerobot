# Custom wrapper around `lerobot-record` that adds an auto-return-to-home reset phase.
#
# Standard `lerobot-record` resets the environment between episodes by leaving teleop live
# for `--dataset.reset_time_s` seconds and expecting a human to move the leader arm back by
# hand. This script instead:
#
#   1. Lets you pose the follower via the leader arm as usual, then press the RIGHT ARROW
#      (the same key used everywhere else in lerobot-record) to lock that pose in as "home".
#   2. After every saved episode, drives the follower back to that home pose automatically —
#      a smooth linear interpolation over `--dataset.reset_time_s` seconds — instead of
#      waiting on a human reset.
#
# Everything else (recording, dataset creation/saving, push_to_hub, CLI flags) is the exact
# same code path as `lerobot-record` — this file imports and reuses it rather than
# reimplementing it, so it stays correct as lerobot itself changes.
#
# Usage: identical CLI flags to `lerobot-record`. Run with the lerobot venv, e.g.:
#
#   ~/lerobot/.venv/bin/python ~/lerobot/custom_scripts/lerobot_record_home_reset.py \
#     --robot.type=so101_follower --robot.port=/dev/so101_follower --robot.id=follower \
#     --teleop.type=so101_leader  --teleop.port=/dev/so101_leader  --teleop.id=leader \
#     --robot.cameras='{ cam1: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30}}' \
#     --dataset.repo_id=youngchan/so101_glue_pickup \
#     --dataset.single_task="Pick up the glue container and lift it." \
#     --dataset.num_episodes=10 \
#     --dataset.episode_time_s=20 \
#     --dataset.reset_time_s=6 \
#     --dataset.push_to_hub=false \
#     --display_data=true

import logging
import time
from dataclasses import asdict
from pprint import pformat

from lerobot.common.control_utils import sanity_check_dataset_robot_compatibility
from lerobot.configs import parser
from lerobot.datasets import (
    LeRobotDataset,
    VideoEncodingManager,
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.processor import make_default_processors
from lerobot.robots import make_robot_from_config
from lerobot.scripts.lerobot_record import RecordConfig, record_loop
from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.utils.cycle_timer import CycleTimer
from lerobot.utils.feature_utils import combine_feature_dicts
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.keyboard_input import init_keyboard_listener
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import init_visualization, shutdown_visualization


def move_to_pose(robot, target_pose: dict, duration_s: float, fps: int, events: dict) -> None:
    """Linearly interpolate the follower from its current position to `target_pose`.

    `target_pose` and the follower's observation both use the "{motor}.pos" key format, so
    a plain per-key lerp works. `robot.send_action()` still applies its own
    `max_relative_target` clamp underneath, as an extra safety net against big jumps if the
    interpolation step size is too coarse for a given fps/duration.

    Right-arrow (events["exit_early"]) skips the remaining interpolation and stops early,
    same as it interrupts any other phase of lerobot-record — the arm just stays wherever
    it got to.
    """
    obs = robot.get_observation()
    start_pose = {k: v for k, v in obs.items() if k in target_pose}

    n_steps = max(1, int(duration_s * fps))
    period_s = 1.0 / fps
    for step in range(1, n_steps + 1):
        if events["exit_early"]:
            events["exit_early"] = False
            break
        t = step / n_steps
        action = {k: start_pose[k] + (target_pose[k] - start_pose[k]) * t for k in target_pose}
        loop_start = time.perf_counter()
        robot.send_action(action)
        elapsed = time.perf_counter() - loop_start
        if elapsed < period_s:
            time.sleep(period_s - elapsed)


@parser.wrap()
def record_with_home_reset(cfg: RecordConfig) -> LeRobotDataset:
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_visualization(
            cfg.display_mode, session_name="recording", ip=cfg.display_ip, port=cfg.display_port
        )
    display_compressed_images = (
        True
        if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
        else cfg.display_compressed_images
    )

    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop)

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=cfg.dataset.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=cfg.dataset.video,
        ),
    )

    dataset = None
    listener = None
    timer = CycleTimer(cfg.dataset.fps)
    home_pose = None

    try:
        if cfg.resume:
            num_cameras = len(robot.cameras) if hasattr(robot, "cameras") else 0
            dataset = LeRobotDataset.resume(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                rgb_encoder=cfg.dataset.rgb_encoder,
                depth_encoder=cfg.dataset.depth_encoder,
                encoder_threads=cfg.dataset.encoder_threads,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                image_writer_processes=cfg.dataset.num_image_writer_processes if num_cameras > 0 else 0,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * num_cameras
                if num_cameras > 0
                else 0,
            )
            sanity_check_dataset_robot_compatibility(dataset, robot, cfg.dataset.fps, dataset_features)
        else:
            repo_name = cfg.dataset.repo_id.split("/", 1)[-1]
            if repo_name.startswith("eval_"):
                raise ValueError(
                    "Dataset names starting with 'eval_' are reserved for policy evaluation. "
                    "Use lerobot-rollout for policy deployment."
                )
            cfg.dataset.stamp_repo_id()
            dataset = LeRobotDataset.create(
                cfg.dataset.repo_id,
                cfg.dataset.fps,
                root=cfg.dataset.root,
                robot_type=robot.name,
                features=dataset_features,
                use_videos=cfg.dataset.video,
                image_writer_processes=cfg.dataset.num_image_writer_processes,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                rgb_encoder=cfg.dataset.rgb_encoder,
                depth_encoder=cfg.dataset.depth_encoder,
                encoder_threads=cfg.dataset.encoder_threads,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
            )

        teleop.connect()
        robot.connect()

        listener, events = init_keyboard_listener()

        # --- Capture the home pose ---------------------------------------------------
        # Drive the follower via the leader as usual; press the right arrow to lock in
        # whatever pose it's currently in as "home". Reuses the exact same record_loop /
        # key-handling path as the rest of the session, just with no dataset and a long
        # timeout so it effectively waits for the right-arrow press.
        log_say(
            "Move the follower to the home position with the leader arm, "
            "then press the right arrow key to confirm it.",
            cfg.play_sounds,
            blocking=True,
        )
        record_loop(
            robot=robot,
            events=events,
            fps=cfg.dataset.fps,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            teleop=teleop,
            control_time_s=3600,  # effectively "until right-arrow is pressed"
            display_data=cfg.display_data,
            display_mode=cfg.display_mode,
            display_compressed_images=display_compressed_images,
            timer=timer,
        )
        obs = robot.get_observation()
        home_pose = {k: v for k, v in obs.items() if k.endswith(".pos")}
        log_say("Home position captured.", cfg.play_sounds)
        logging.info(f"Home pose: {home_pose}")

        with VideoEncodingManager(dataset):
            recorded_episodes = 0
            while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
                episode_index = dataset.num_episodes
                log_say(f"Recording episode {episode_index}", cfg.play_sounds)
                record_loop(
                    robot=robot,
                    events=events,
                    fps=cfg.dataset.fps,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=teleop,
                    dataset=dataset,
                    control_time_s=cfg.dataset.episode_time_s,
                    single_task=cfg.dataset.single_task,
                    display_data=cfg.display_data,
                    display_mode=cfg.display_mode,
                    display_compressed_images=display_compressed_images,
                    timer=timer,
                )

                # Auto-return to home instead of a human-driven reset.
                if not events["stop_recording"] and (
                    (recorded_episodes < cfg.dataset.num_episodes - 1) or events["rerecord_episode"]
                ):
                    log_say("Returning to home position", cfg.play_sounds)
                    move_to_pose(
                        robot,
                        home_pose,
                        duration_s=cfg.dataset.reset_time_s,
                        fps=cfg.dataset.fps,
                        events=events,
                    )

                if events["rerecord_episode"]:
                    log_say("Re-record episode", cfg.play_sounds)
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    timer.log_episode_summary("discarded episode")
                    timer.restart()
                    continue

                dataset.save_episode()
                recorded_episodes += 1
                timer.log_episode_summary(f"episode {episode_index}")
                timer.restart()
    finally:
        timer.log_run_summary()

        log_say("Stop recording", cfg.play_sounds, blocking=True)

        if dataset:
            dataset.finalize()

        if robot.is_connected:
            robot.disconnect()
        if teleop and teleop.is_connected:
            teleop.disconnect()

        if listener is not None:
            listener.stop()

        if cfg.display_data:
            shutdown_visualization(cfg.display_mode)

        if cfg.dataset.push_to_hub:
            if dataset and dataset.num_episodes > 0:
                dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)
            else:
                logging.warning("No episodes saved — skipping push to hub")

        log_say("Exiting", cfg.play_sounds)
    return dataset


def main():
    register_third_party_plugins()
    record_with_home_reset()


if __name__ == "__main__":
    main()
