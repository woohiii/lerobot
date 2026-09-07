# Custom wrapper around `lerobot-record` for collecting the SHORT grasp demos the
# hybrid IL skill (il_grasp_skill.py) is trained on.
#
# This is lerobot_record_home_reset.py plus ONE extra phase: before each episode the
# Legacy vision+IK pre-positioning (task_state_machine.search() + coarse_center()) drives
# the follower to the standardized pre-grasp hover pose, and only then does the stock
# `record_loop` run, so the human demonstrates just the 3-5s contact part via the leader
# arm. Every episode therefore starts from the same distribution the skill will see at
# task time - that start-state standardization is the whole reason the hybrid split works.
#
# Everything else (recording, dataset creation/saving, push_to_hub, CLI flags, the
# auto-return-to-home reset between episodes) is the exact same code path as
# `lerobot-record` / lerobot_record_home_reset.py - imported and reused, not reimplemented.
#
# CAMERA OWNERSHIP (the reason for RobotObservationFrameSource below):
# at task time camera_hub.py owns the physical wrist camera and task_state_machine reads
# its published frames via PublishedFrameSource, because two processes cannot both hold the
# same UVC device open. Recording cannot work that way: stock record_loop needs the frames
# to come through robot.cameras so they land in the dataset via the normal capture path.
# So HERE the SOFollower owns the camera for the whole run (pre-positioning included) and
# the adapter feeds robot.get_observation()'s frame to the unmodified search()/
# coarse_center()/get_pixel() functions. camera_hub.py must NOT be running during recording.
#
# SAFETY: after pre-positioning, the follower is wherever IK put it while the leader is
# wherever the human left it. record_loop immediately commands the follower to the leader's
# pose, so a large mismatch is a violent jump. This script blocks on an Enter prompt so the
# human can align the leader first; pass --robot.max_relative_target=15 as well to bound
# that jump in hardware terms (the follower config leaves it unset by default).
#
# Usage: same CLI flags as `lerobot-record`, e.g.:
#
#   env -u PYTHONPATH ~/lerobot/.venv/bin/python ~/lerobot/custom_scripts/lerobot_record_grasp_skill.py \
#     --robot.type=so101_follower --robot.port=/dev/so101_follower --robot.id=follower \
#     --robot.max_relative_target=15 \
#     --teleop.type=so101_leader  --teleop.port=/dev/so101_leader  --teleop.id=leader \
#     --robot.cameras='{ wrist: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30}}' \
#     --dataset.repo_id=youngchan/so101_grasp_skill \
#     --dataset.single_task="grasp the object from the standardized pre-grasp pose" \
#     --dataset.num_episodes=30 \
#     --dataset.episode_time_s=5 \
#     --dataset.reset_time_s=6 \
#     --dataset.push_to_hub=false

import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
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

# The task package uses flat imports (`import config`), same as main.py runs it.
TASK_DIR = Path(__file__).resolve().parent / "vision_pick_place" / "task_red_cube_to_bin_new_gripper"
sys.path.insert(0, str(TASK_DIR))


class RobotObservationFrameSource:
    """Same .read()/.isOpened()/.release() shape as camera_utils.PublishedFrameSource,
    but backed by robot.get_observation() instead of a file camera_hub.py publishes.

    Exists only so the unmodified task_state_machine.search()/coarse_center()/get_pixel()
    can run inside this recording process, where the SOFollower itself owns the wrist
    camera (see the module header on camera ownership). Converts RGB->BGR because
    OpenCVCamera defaults to color_mode=RGB while every perception.* function was written
    against OpenCV BGR frames."""

    def __init__(self, robot, camera_key: str):
        self.robot = robot
        self.camera_key = camera_key

    def isOpened(self) -> bool:
        return self.camera_key in self.robot.cameras

    def read(self):
        frame = self.robot.get_observation().get(self.camera_key)
        if frame is None:
            return False, None
        return True, frame[:, :, ::-1]

    def release(self) -> None:
        pass


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


def move_to_pre_grasp(arm, cap) -> bool:
    """Legacy SEARCH + coarse centering, exactly as task_state_machine runs it at task
    time - no reimplementation, so the recorded start states and the deployed start states
    come from the same code. Returns False if the cube wasn't found; the caller then lets
    the operator decide (re-place the cube, or record anyway).

    Note search() first tries perception.estimate_xy_from_astra(), which reads the Astra
    frames astra_s_live.py publishes. That script CAN run alongside this one (different
    device), but if it isn't running the estimate is simply unavailable and search() falls
    back to its blind grid sweep - no special handling needed here."""
    import perception
    import task_state_machine
    from kinematics import CollisionDetected

    try:
        return task_state_machine.search(
            arm, cap, perception.detect_red_cube, "빨간 큐브"
        ) and task_state_machine.coarse_center(arm, cap, perception.detect_red_cube)
    except CollisionDetected as e:
        logging.warning(f"Pre-grasp positioning hit a collision: {e}")
        return False


@parser.wrap()
def record_grasp_skill(cfg: RecordConfig) -> LeRobotDataset:
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

        if not robot.cameras:
            raise ValueError(
                "This script needs the wrist camera on the robot itself "
                "(--robot.cameras=...) - see the camera-ownership note in the header."
            )
        # One camera expected (the wrist cam). With several, the first is the one the
        # Legacy pre-positioning looks through; there is no flag for it because
        # RecordConfig is reused verbatim.
        camera_key = next(iter(robot.cameras))
        cap = RobotObservationFrameSource(robot, camera_key)

        # Wraps the ALREADY-CONNECTED follower (Phase 1's robot= parameter) rather than
        # opening a second connection to the same port - built once, reused every episode.
        from kinematics import SOArm101

        arm = SOArm101(robot=robot)
        arm.connect()  # no-op for the injected robot; still applies the motor torque caps

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

                # --- NEW PHASE: standardized pre-grasp pose ---------------------------
                log_say("Moving to the pre-grasp pose", cfg.play_sounds)
                if not move_to_pre_grasp(arm, cap):
                    log_say("Pre-grasp positioning failed", cfg.play_sounds)
                    logging.warning("Cube not found / not centered - reposition it before continuing.")
                input(
                    "리더암을 팔로워 자세에 맞춘 뒤 Enter를 누르면 녹화가 시작됩니다 "
                    "(맞추지 않으면 팔로워가 리더 쪽으로 크게 튑니다): "
                )

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
    record_grasp_skill()


if __name__ == "__main__":
    main()
