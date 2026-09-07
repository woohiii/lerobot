#!/usr/bin/env python3
"""Mobile dual-SO-101 MuJoCo demo with keyboard driving, LiDAR SLAM and A*.

This is deliberately a self-contained *simulation* launcher.  On its first
run it obtains the public ``Thakk100/so101_dual_arm_env`` MuJoCo asset through
the Hugging Face cache; no robot hardware, serial port, or model weights are
needed.  The two genuine SO-101 MJCF arms are carried kinematically by a
mobile-base mocap body.  A planar 2-D LiDAR updates an occupancy grid and the
autonomy mode follows an A* path to the green goal.

Run from the LeRobot checkout (desktop/X11/Wayland session required):

    uv run python custom_scripts/mujoco_mobile_bimanual_so101.py

Keys (click the MuJoCo window first):
  W/S/A/D  translate base       Q/E rotate base       P autonomous on/off
  G        choose a new goal in front of the robot    TAB choose left/right arm
  J/L I/K U/O Y/H T/B V/N       selected arm joint -/+ (0 through 5)
  SPACE    stop base            ESC closes the viewer
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from huggingface_hub import snapshot_download

ASSET_REPO = "Thakk100/so101_dual_arm_env"
GRID_RESOLUTION_M = 0.05
GRID_SIZE = 120  # 6 m square map, centred on world origin
BASE_SPEED_M = 0.12
TURN_SPEED_RAD = 0.18
ARM_STEP_RAD = 0.08
LIDAR_MAX_RANGE_M = 2.5


@dataclass
class BaseState:
    x: float = -1.4
    y: float = 0.0
    yaw: float = 0.0
    autonomous: bool = False
    goal: tuple[float, float] = (1.15, 0.0)
    selected_arm: int = 0


def _asset_directory() -> Path:
    """Get the external, dual-arm MJCF only; its revision remains cache-pinned."""
    root = snapshot_download(
        ASSET_REPO,
        repo_type="model",
        allow_patterns=["assets/*", "README.md"],
    )
    return Path(root) / "assets"


def _make_model(asset_dir: Path) -> mujoco.MjModel:
    # The source scene already supplies two individually named SO-101 arms.
    # Add only the mobile carrier and an indoor navigation course at runtime,
    # leaving downloaded third-party files untouched.
    xml = (asset_dir / "scene.xml").read_text()
    xml = xml.replace('meshdir="meshes"', f'meshdir="{asset_dir / "meshes"}"')
    extra = """
    <body name="mobile_base" mocap="true" pos="-1.4 0 0.18">
      <geom name="base_chassis" type="box" size="0.32 0.24 0.12" rgba="0.08 0.20 0.32 1"/>
      <geom type="cylinder" size="0.09 0.05" pos="0.22 0.25 -0.09" euler="90 0 0" rgba="0.04 0.04 0.04 1"/>
      <geom type="cylinder" size="0.09 0.05" pos="0.22 -0.25 -0.09" euler="90 0 0" rgba="0.04 0.04 0.04 1"/>
      <geom type="cylinder" size="0.09 0.05" pos="-0.22 0.25 -0.09" euler="90 0 0" rgba="0.04 0.04 0.04 1"/>
      <geom type="cylinder" size="0.09 0.05" pos="-0.22 -0.25 -0.09" euler="90 0 0" rgba="0.04 0.04 0.04 1"/>
      <geom type="box" size="0.36 0.30 0.035" pos="0 0 0.15" rgba="0.15 0.42 0.65 1"/>
      <site name="lidar" pos="0.22 0 0.22" size="0.01" rgba="0.1 1 0.1 1"/>
    </body>
    <geom name="wall_n" type="box" pos="0 2.85 0.35" size="2.9 0.06 0.35" rgba="0.45 0.45 0.45 1"/>
    <geom name="wall_s" type="box" pos="0 -2.85 0.35" size="2.9 0.06 0.35" rgba="0.45 0.45 0.45 1"/>
    <geom name="wall_e" type="box" pos="2.85 0 0.35" size="0.06 2.9 0.35" rgba="0.45 0.45 0.45 1"/>
    <geom name="wall_w" type="box" pos="-2.85 0 0.35" size="0.06 2.9 0.35" rgba="0.45 0.45 0.45 1"/>
    <geom name="obstacle_1" type="box" pos="-0.20 0.65 0.35" size="0.18 0.70 0.35" rgba="0.75 0.35 0.12 1"/>
    <geom name="obstacle_2" type="box" pos="0.95 -0.80 0.35" size="0.55 0.16 0.35" rgba="0.75 0.35 0.12 1"/>
    <geom name="obstacle_3" type="box" pos="1.55 0.65 0.35" size="0.16 0.55 0.35" rgba="0.75 0.35 0.12 1"/>
    """
    xml = xml.replace("<worldbody>", f"<worldbody>{extra}")
    return mujoco.MjModel.from_xml_string(xml)


class OccupancySLAM:
    """Small lidar-only occupancy map: -1 unknown, 0 free, 1 occupied."""

    def __init__(self) -> None:
        self.grid = np.full((GRID_SIZE, GRID_SIZE), -1, dtype=np.int8)

    @staticmethod
    def _cell(x: float, y: float) -> tuple[int, int] | None:
        col = int(round(x / GRID_RESOLUTION_M + GRID_SIZE / 2))
        row = int(round(y / GRID_RESOLUTION_M + GRID_SIZE / 2))
        if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE:
            return row, col
        return None

    def _ray(self, start: tuple[float, float], end: tuple[float, float], hit: bool) -> None:
        a, b = self._cell(*start), self._cell(*end)
        if a is None or b is None:
            return
        r0, c0, r1, c1 = *a, *b
        steps = max(abs(r1 - r0), abs(c1 - c0), 1)
        for i in range(steps):
            r = round(r0 + (r1 - r0) * i / steps)
            c = round(c0 + (c1 - c0) * i / steps)
            self.grid[r, c] = 0
        self.grid[r1, c1] = 1 if hit else 0

    def update(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        pose: BaseState,
        base_body_id: int,
    ) -> None:
        # MuJoCo raycasts the actual course geometry, so this is sensor data,
        # not a hard-coded map. Ignore the robot's own carrier geometry.
        origin = np.array([pose.x, pose.y, 0.27])
        for rel_angle in np.linspace(-math.pi, math.pi, 72, endpoint=False):
            angle = pose.yaw + float(rel_angle)
            direction = np.array([math.cos(angle), math.sin(angle), 0.0])
            geom_id = np.array([-1], dtype=np.int32)
            distance = mujoco.mj_ray(model, data, origin, direction, None, 1, base_body_id, geom_id)
            hit = 0 < distance < LIDAR_MAX_RANGE_M
            length = min(distance, LIDAR_MAX_RANGE_M) if distance >= 0 else LIDAR_MAX_RANGE_M
            endpoint = origin[:2] + direction[:2] * length
            self._ray((pose.x, pose.y), tuple(endpoint), hit)

    def plan(self, start_xy: tuple[float, float], goal_xy: tuple[float, float]) -> list[tuple[int, int]]:
        start, goal = self._cell(*start_xy), self._cell(*goal_xy)
        if start is None or goal is None:
            return []
        import heapq

        queue: list[tuple[float, tuple[int, int]]] = [(0.0, start)]
        previous = {start: None}
        cost = {start: 0.0}
        neighbors = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]
        while queue:
            _, current = heapq.heappop(queue)
            if current == goal:
                break
            for dr, dc in neighbors:
                nxt = current[0] + dr, current[1] + dc
                if not (0 <= nxt[0] < GRID_SIZE and 0 <= nxt[1] < GRID_SIZE):
                    continue
                if self.grid[nxt] == 1:
                    continue
                # Unknown cells are traversable but more costly: the robot
                # naturally prefers the corridor it has already scanned.
                new_cost = cost[current] + math.hypot(dr, dc) + (2.0 if self.grid[nxt] < 0 else 0.0)
                if new_cost < cost.get(nxt, float("inf")):
                    cost[nxt] = new_cost
                    previous[nxt] = current
                    h = math.hypot(goal[0] - nxt[0], goal[1] - nxt[1])
                    heapq.heappush(queue, (new_cost + h, nxt))
        if goal not in previous:
            return []
        path = []
        current: tuple[int, int] | None = goal
        while current is not None:
            path.append(current)
            current = previous[current]
        return path[::-1]

    @staticmethod
    def world(cell: tuple[int, int]) -> tuple[float, float]:
        return ((cell[1] - GRID_SIZE / 2) * GRID_RESOLUTION_M, (cell[0] - GRID_SIZE / 2) * GRID_RESOLUTION_M)


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def main() -> None:
    asset_dir = _asset_directory()
    model = _make_model(asset_dir)
    data = mujoco.MjData(model)
    state, slam, lock = BaseState(), OccupancySLAM(), threading.Lock()
    mocap_id = model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mobile_base")]
    mobile_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mobile_base")
    # Original bases are world bodies. Updating their poses makes the genuine
    # arm model follow the movable carrier without altering the upstream asset.
    arm_bodies = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_base"),
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_base"),
    ]
    arm_offsets = [(0.02, 0.28), (0.02, -0.28)]
    arm_actuators = [
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{side}_{joint}")
            for joint in (
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
                "gripper",
            )
        ]
        for side in ("left", "right")
    ]
    joint_targets = np.zeros((2, 6))

    def move_base(forward: float = 0.0, lateral: float = 0.0, turn: float = 0.0) -> None:
        with lock:
            state.yaw += turn
            state.x += forward * math.cos(state.yaw) - lateral * math.sin(state.yaw)
            state.y += forward * math.sin(state.yaw) + lateral * math.cos(state.yaw)
            state.autonomous = False

    joint_keys = {
        ord(key): (joint_index, direction)
        for joint_index, pair in enumerate(("jl", "ik", "uo", "yh", "tb", "vn"))
        for key, direction in ((pair[0], -1), (pair[1], 1))
    }

    def on_key(keycode: int) -> None:
        key = chr(keycode).lower() if 0 <= keycode < 256 else ""
        if key == "w":
            move_base(forward=BASE_SPEED_M)
        elif key == "s":
            move_base(forward=-BASE_SPEED_M)
        elif key == "a":
            move_base(lateral=BASE_SPEED_M)
        elif key == "d":
            move_base(lateral=-BASE_SPEED_M)
        elif key == "q":
            move_base(turn=TURN_SPEED_RAD)
        elif key == "e":
            move_base(turn=-TURN_SPEED_RAD)
        elif key == "p":
            with lock:
                state.autonomous = not state.autonomous
        elif key == "g":
            with lock:
                state.goal = (state.x + 1.5 * math.cos(state.yaw), state.y + 1.5 * math.sin(state.yaw))
                state.autonomous = True
        elif key == " ":
            with lock:
                state.autonomous = False
        elif keycode == 258:  # GLFW_KEY_TAB
            with lock:
                state.selected_arm = 1 - state.selected_arm
        elif keycode in joint_keys:
            idx, direction = joint_keys[keycode]
            with lock:
                arm = state.selected_arm
                actuator = arm_actuators[arm][idx]
                lo, hi = model.actuator_ctrlrange[actuator]
                joint_targets[arm, idx] = np.clip(joint_targets[arm, idx] + direction * ARM_STEP_RAD, lo, hi)

    print(__doc__.split("Run from")[0].strip())
    print("\nMuJoCo 창을 클릭한 뒤 W/S/A/D, Q/E, P, G, TAB, J/L…I/K… 키를 사용하세요.")
    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        last_plan = 0.0
        path: list[tuple[int, int]] = []
        while viewer.is_running():
            now = time.monotonic()
            with lock:
                # Update map before planning. Replanning at 5 Hz incorporates
                # newly observed walls as the platform moves.
                slam.update(model, data, state, mobile_body_id)
                if state.autonomous and now - last_plan > 0.2:
                    path = slam.plan((state.x, state.y), state.goal)
                    last_plan = now
                if state.autonomous and path:
                    waypoint = slam.world(path[min(3, len(path) - 1)])
                    dx, dy = waypoint[0] - state.x, waypoint[1] - state.y
                    distance = math.hypot(dx, dy)
                    if (
                        distance < 0.12
                        and math.hypot(state.goal[0] - state.x, state.goal[1] - state.y) < 0.18
                    ):
                        state.autonomous = False
                    elif distance > 1e-5:
                        desired = math.atan2(dy, dx)
                        state.yaw += float(
                            np.clip((desired - state.yaw + math.pi) % (2 * math.pi) - math.pi, -0.08, 0.08)
                        )
                        state.x += BASE_SPEED_M * 0.02 * math.cos(state.yaw)
                        state.y += BASE_SPEED_M * 0.02 * math.sin(state.yaw)

                data.mocap_pos[mocap_id] = (state.x, state.y, 0.18)
                data.mocap_quat[mocap_id] = _yaw_quat(state.yaw)
                for body, (ox, oy) in zip(arm_bodies, arm_offsets, strict=True):
                    model.body_pos[body] = (
                        state.x + ox * math.cos(state.yaw) - oy * math.sin(state.yaw),
                        state.y + ox * math.sin(state.yaw) + oy * math.cos(state.yaw),
                        0.33,
                    )
                    model.body_quat[body] = _yaw_quat(state.yaw)
                for arm in range(2):
                    data.ctrl[arm_actuators[arm]] = joint_targets[arm]
                mujoco.mj_step(model, data)
                viewer.sync()
            time.sleep(0.02)


if __name__ == "__main__":
    main()
