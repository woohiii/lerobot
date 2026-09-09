"""CLI for local Ollama RGB-D detection and text-directed pick/place preview."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .config import load_pipeline_config
from .frame_io import load_rgbd_frame
from .adapters import SO101ArmAdapter
from .left_pick_executor import build_and_validate_waypoints
from .local_planner import LocalPickPlacePlan, plan_local_task
from .local_vlm import OllamaVlmClient
from .orchestrator import PickAndPlaceExecutor
from .snapshot import capture_published_snapshot


_TABLE_Z_M = -0.04
_HOVER_HEIGHT_M = 0.08
_LIFT_HEIGHT_M = 0.10
_MAX_JOINT_DELTA_DEG = 10.0


def format_preview(plan: LocalPickPlacePlan) -> str:
    return "\n".join(
        [
            "=== Pick-and-place preview (no motor command sent yet) ===",
            f"source: {plan.command.source.label} box={plan.command.source.box_xyxy} "
            f"confidence={plan.command.source.confidence:.2f}",
            f"destination: {plan.command.destination.label} box={plan.command.destination.box_xyxy} "
            f"confidence={plan.command.destination.confidence:.2f}",
            f"arm: {plan.arm.name}",
            f"source base xyz: {tuple(round(value, 4) for value in plan.source_xyz)} m",
            f"destination base xyz: {tuple(round(value, 4) for value in plan.destination_xyz)} m",
            "The next step may send motor commands. Type 'yes' only after checking the plan.",
        ]
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--command", required=True, help="e.g. 'red block를 blue tray에 가져다줘'")
    parser.add_argument("--snapshot", type=Path, help="previously saved RGB-D snapshot directory")
    parser.add_argument("--rgb-path", type=Path, help="published Astra RGB PNG")
    parser.add_argument("--depth-path", type=Path, help="published Astra depth .npy")
    parser.add_argument("--snapshot-root", type=Path, default=Path("captures"))
    parser.add_argument("--ollama-endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--ollama-model", default="qwen2.5vl:7b")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--yes", action="store_true", help="confirm the printed preview")
    return parser


def _load_frame(args, config):
    if args.snapshot is not None:
        return load_rgbd_frame(args.snapshot)
    if args.rgb_path is None or args.depth_path is None:
        raise ValueError("provide --snapshot or both --rgb-path and --depth-path")
    directory = capture_published_snapshot(
        args.rgb_path, args.depth_path, config.camera_intrinsics, args.snapshot_root
    )
    print(f"snapshot: {directory}")
    return load_rgbd_frame(directory)


def _execute(plan: LocalPickPlacePlan) -> None:
    if not plan.arm.port:
        raise ValueError(f"arm {plan.arm.name!r} has no configured port")
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from lerobot.robots.so_follower.so_follower import SOFollower

    from .left_arm import LeftSOArm101

    # Give each physical arm its own LeRobot calibration id. The legacy task
    # wrapper defaults to a single "follower" id, which is unsafe for a pair.
    robot = SOFollower(
        SOFollowerRobotConfig(
            port=plan.arm.port,
            id=plan.arm.name,
            use_degrees=True,
            max_relative_target=15.0,
        )
    )
    robot.connect(calibrate=False)
    arm = LeftSOArm101(port=plan.arm.port, robot=robot)
    arm._protect_arm_motors()
    try:
        _execute_connected_left_arm(plan, arm)
    finally:
        robot.disconnect()


def _execute_connected_left_arm(plan: LocalPickPlacePlan, arm) -> None:
    """Preflight and register retreat before the executor can send a move."""
    waypoints = build_and_validate_waypoints(
        arm,
        np.asarray(plan.source_xyz, dtype=float),
        np.asarray(plan.destination_xyz, dtype=float),
        table_z=_TABLE_Z_M,
        hover_height_m=_HOVER_HEIGHT_M,
        lift_height_m=_LIFT_HEIGHT_M,
        max_joint_delta_deg=_MAX_JOINT_DELTA_DEG,
    )
    home = tuple(float(value) for value in arm.gripper_xyz())
    adapter = SO101ArmAdapter(arm)
    adapter.set_validated_retreat_waypoint(waypoints.retreat)
    executor = PickAndPlaceExecutor(
        arm=adapter,
        verify_gripper=adapter.verify_grasp,
        verify_wrist=adapter.verify_pose,
        home=home,
        bin_pose=plan.destination_xyz,
        approach_height_m=_HOVER_HEIGHT_M,
        lift_height_m=_LIFT_HEIGHT_M,
    )
    executor.execute_validated(waypoints)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.dry_run == args.execute:
        parser.error("choose exactly one of --dry-run or --execute")
    try:
        config = load_pipeline_config(args.config)
        frames = (_load_frame(args, config),)
        if len(frames) != 3:
            raise ValueError(
                "local CLI accepts one RGB-D frame; capture three registered frames before stable planning"
            )
        frame = frames[0]
        client = OllamaVlmClient(endpoint=args.ollama_endpoint, model=args.ollama_model)
        detections = client.detect_objects(
            frame.rgb,
            "Detect every physical object and every destination tray/bin/region visible in the image. "
            "Use kind=object for pickable objects and kind=destination for places. "
            "Ignore robots, cables, furniture, and the whole background. "
            "If no clear pickable object or destination is visible, return an empty objects array.",
        )
        command = client.parse_task(args.command, detections)
        plan = plan_local_task(
            frame,
            command,
            config.camera_to_base,
            config.arms,
            center_exclusion_half_width=config.center_exclusion_half_width_m,
        )
        print(format_preview(plan))
        if args.execute:
            config.require_execution_ready()
            if not args.yes:
                print("execution blocked: rerun with --yes after reviewing the preview")
                return 2
            _execute(plan)
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
