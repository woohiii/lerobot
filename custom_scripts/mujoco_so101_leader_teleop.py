#!/usr/bin/env python3
"""Drive one MuJoCo SO-101 follower with a real, calibrated SO-101 leader.

Only ``Present_Position`` is read from the USB leader; this program never
commands the physical leader or a physical follower.  Moving the leader arm
therefore moves the simulated follower in the MuJoCo window.

Example:
    uv run python custom_scripts/mujoco_so101_leader_teleop.py \
        --leader-port /dev/ttyACM0 --leader-id my_leader

The leader must already have been calibrated with the same ``leader-id``.
Close the MuJoCo window (or press ESC) to disconnect the USB bus cleanly.
"""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig


SCENE_PATH = Path(__file__).resolve().parent / "vision_pick_place/task_red_cube_to_bin/mujoco_sim/scene.xml"
JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def parse_args() -> argparse.Namespace:
    """Read the leader connection details without silently choosing hardware."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leader-port", required=True, help="Leader USB serial port, e.g. /dev/ttyACM0")
    parser.add_argument("--leader-id", required=True, help="Existing LeRobot leader calibration ID")
    parser.add_argument("--fps", type=float, default=120.0, help="Simulation/viewer update rate")
    parser.add_argument("--read-fps", type=float, default=120.0, help="Maximum leader USB read rate")
    parser.add_argument("--camera-fps", type=float, default=15.0, help="Camera overlay update rate")
    return parser.parse_args()


def leader_to_ctrl(
    leader_positions: dict[str, float],
    leader_reference: dict[str, float],
    sim_reference: np.ndarray,
    ctrl_ranges: np.ndarray,
) -> np.ndarray:
    """Map leader-relative motion to the simulation's position actuators."""
    values = np.empty(6)
    for index, name in enumerate(JOINT_NAMES[:5]):
        delta_rad = np.deg2rad(leader_positions[f"{name}.pos"] - leader_reference[f"{name}.pos"])
        values[index] = sim_reference[index] + delta_rad

    # LeRobot exposes the SO-101 gripper as 0..100 percent.  The MJCF gripper
    # uses radians, so map it directly over its declared actuator range.
    gripper_percent = np.clip(leader_positions["gripper.pos"], 0.0, 100.0) / 100.0
    lo, hi = ctrl_ranges[5]
    values[5] = lo + gripper_percent * (hi - lo)
    return np.clip(values, ctrl_ranges[:, 0], ctrl_ranges[:, 1])


def leader_home_to_ctrl(leader_positions: dict[str, float], ctrl_ranges: np.ndarray) -> np.ndarray:
    """Convert the leader's calibrated absolute pose into an SO-101 sim pose."""
    values = np.empty(6)
    for index, name in enumerate(JOINT_NAMES[:5]):
        values[index] = np.deg2rad(leader_positions[f"{name}.pos"])
    gripper_percent = np.clip(leader_positions["gripper.pos"], 0.0, 100.0) / 100.0
    lo, hi = ctrl_ranges[5]
    values[5] = lo + gripper_percent * (hi - lo)
    return np.clip(values, ctrl_ranges[:, 0], ctrl_ranges[:, 1])


def render_camera_overlays(
    renderer: mujoco.Renderer,
    data: mujoco.MjData,
    viewer: mujoco.viewer.Handle,
) -> None:
    """Render wrist RGB plus front depth as two overlays inside the MuJoCo GUI."""
    renderer.disable_depth_rendering()
    renderer.update_scene(data, camera="wrist_cam")
    wrist_rgb = renderer.render()

    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera="astra_cam")
    depth_m = renderer.render()
    renderer.disable_depth_rendering()
    # Near objects are bright.  This depth visualization is deliberately
    # 8-bit RGB so Handle.set_images can draw it without another GUI toolkit.
    normalized = 1.0 - np.clip(np.nan_to_num(depth_m, nan=3.0) / 3.0, 0.0, 1.0)
    depth_rgb = np.repeat((normalized * 255).astype(np.uint8)[..., None], 3, axis=2)

    viewport = viewer.viewport
    width = min(320, viewport.width // 3)
    height = width * 3 // 4
    viewer.set_images(
        [
            (mujoco.MjrRect(10, 10, width, height), wrist_rgb),
            (mujoco.MjrRect(20 + width, 10, width, height), depth_rgb),
        ]
    )
    viewer.set_texts(
        [
            (mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPLEFT, "Wrist RGB", None),
            (mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPRIGHT, "Front depth", None),
        ]
    )


def main() -> None:
    """Open the calibrated leader and run a single-threaded MuJoCo teleop loop."""
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    # `wrist_cam` belongs to the SO-101 MJCF. This front-looking camera is
    # attached to the scene and rendered in depth mode below.
    front_camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "astra_cam")
    if front_camera_id < 0:
        raise RuntimeError("SO-101 scene is missing the front depth camera (astra_cam)")
    model.cam_pos[front_camera_id] = (0.75, -0.55, 0.65)
    model.cam_fovy[front_camera_id] = 65.0
    data = mujoco.MjData(model)
    actuator_ids = np.array(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in JOINT_NAMES]
    )
    joint_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINT_NAMES])
    if np.any(actuator_ids < 0):
        raise RuntimeError(f"SO-101 scene is missing an actuator: {JOINT_NAMES}")
    ctrl_ranges = model.actuator_ctrlrange[actuator_ids].copy()

    leader = SO101Leader(SO101LeaderConfig(port=args.leader_port, id=args.leader_id))
    print("리더암 USB 연결 중입니다. 이 프로그램은 리더암에 목표 위치를 쓰지 않습니다.")
    leader.connect(calibrate=False)
    reader_stop = threading.Event()
    reader_lock = threading.Lock()
    latest_positions: dict[str, float] = {}
    reader_error: list[BaseException] = []
    try:
        if not leader.is_calibrated:
            raise RuntimeError(
                f"리더암 calibration이 없습니다 (id={args.leader_id!r}). 먼저 lerobot-calibrate를 실행하세요."
            )
        leader_reference = leader.get_action()
        sim_reference = leader_home_to_ctrl(leader_reference, ctrl_ranges)
        # Set qpos as well as control targets, so the visual arm begins at
        # the leader pose rather than visibly travelling from a zero pose.
        data.qpos[model.jnt_qposadr[joint_ids]] = sim_reference
        data.ctrl[actuator_ids] = sim_reference
        mujoco.mj_forward(model, data)
        print("리더암의 현재 보정 포즈를 MuJoCo 홈 포즈에 적용했습니다. ESC로 종료합니다.")

        latest_positions = leader_reference.copy()

        def read_leader() -> None:
            """Keep serial I/O off the render loop to minimize control latency."""
            read_interval = 1.0 / args.read_fps
            while not reader_stop.is_set():
                started = time.perf_counter()
                try:
                    positions = leader.get_action()
                except BaseException as exc:  # Stop safely on USB disconnect/errors.
                    reader_error.append(exc)
                    reader_stop.set()
                    return
                with reader_lock:
                    latest_positions.clear()
                    latest_positions.update(positions)
                reader_stop.wait(max(0.0, read_interval - (time.perf_counter() - started)))

        reader = threading.Thread(target=read_leader, name="so101-leader-read", daemon=True)
        reader.start()
        interval = 1.0 / args.fps
        camera_interval = 1.0 / args.camera_fps
        last_camera_render = -float("inf")
        renderer = mujoco.Renderer(model, height=240, width=320)
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                started = time.perf_counter()
                if reader_error:
                    raise RuntimeError("리더암 USB 읽기에 실패했습니다.") from reader_error[0]
                with reader_lock:
                    positions = latest_positions.copy()
                data.ctrl[actuator_ids] = leader_to_ctrl(
                    positions, leader_reference, sim_reference, ctrl_ranges
                )
                mujoco.mj_step(model, data)
                if started - last_camera_render >= camera_interval:
                    render_camera_overlays(renderer, data, viewer)
                    last_camera_render = started
                viewer.sync()
                time.sleep(max(0.0, interval - (time.perf_counter() - started)))
    finally:
        reader_stop.set()
        if "reader" in locals():
            reader.join(timeout=1.0)
        leader.disconnect()
        print("리더암 연결을 해제했습니다.")


if __name__ == "__main__":
    main()
