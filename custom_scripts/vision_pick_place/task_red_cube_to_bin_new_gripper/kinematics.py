"""SO-101 IK wrapper + the arm-motion primitives everything else calls into.

# TODO: 실제 SDK 함수로 교체 - this uses lerobot's real SO101Follower driver
(lerobot.robots.so_follower) and its placo-based RobotKinematics, both
already connected to and validated against real hardware in this project all
day (not a stub) - if a different SDK is ever swapped in, SOArm101.connect/
send_joint_deg/get_joint_deg are the methods to replace.

Top-down orientation: see config.py's IK_ORIENTATION_WEIGHT comment. Tested
live today (multiple targets, weight 0.05-1.0, from both a neutral and the
arm's real current pose) - any nonzero weight blew position error up to
16-253mm on this 5-DOF arm. Position-only IK is used deliberately; gripper
orientation is left to float and is compensated for by FINE_SERVO's
per-approach self-calibration instead of an IK constraint.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from lerobot.model import RobotKinematics
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SOFollower

import config


class CollisionDetected(RuntimeError):
    """The arm's actual joint position stopped tracking the commanded
    trajectory for several consecutive checks - it hit something. The arm is
    sent back to its last known-good pose before this is raised."""


@dataclass
class JointStallGuard:
    """Reusable live joint-stall detector for direct policy action loops.

    ``move_to_xyz`` owns the same safety rule for interpolated IK moves. This
    guard is for policies that issue a fresh joint target each tick: it uses
    the configured lag/count thresholds, retreats to the last observed good
    pose, then raises ``CollisionDetected``. ``min_motion_deg`` avoids a
    false collision while the per-send relative-target clamp is still moving
    the arm toward a far policy target.
    """

    arm: "SOArm101"
    last_good: np.ndarray
    prev_actual: np.ndarray
    min_motion_deg: float = 0.5
    stall_count: int = 0

    @classmethod
    def start(cls, arm: "SOArm101", min_motion_deg: float = 0.5) -> "JointStallGuard":
        current = arm.get_joint_deg()
        return cls(arm=arm, last_good=current.copy(), prev_actual=current, min_motion_deg=min_motion_deg)

    def check(self, target: np.ndarray) -> None:
        actual = self.arm.get_joint_deg()
        n_arm = len(config.ARM_JOINTS)
        lag = float(np.max(np.abs(actual[:n_arm] - target[:n_arm])))
        moved = float(np.max(np.abs(actual[:n_arm] - self.prev_actual[:n_arm])))
        self.prev_actual = actual
        if lag > config.STALL_THRESHOLD_DEG and moved < self.min_motion_deg:
            self.stall_count += 1
        else:
            self.stall_count, self.last_good = 0, actual
        if self.stall_count >= config.STALL_CONSECUTIVE:
            self.arm.send_joint_deg(self.last_good)
            time.sleep(0.3)
            raise CollisionDetected(
                f"policy action aborted: joint lag {lag:.1f}deg with no motion for "
                f"{config.STALL_CONSECUTIVE} consecutive checks - retreated to last known-good pose."
            )


def clamp_joint_deg(joint_deg: np.ndarray) -> np.ndarray:
    clamped = joint_deg.copy()
    for i, name in enumerate(config.ALL_JOINTS):
        lo, hi = config.JOINT_LIMITS_DEG[name]
        clamped[i] = np.clip(clamped[i], lo, hi)
    return clamped


def build_kinematics() -> RobotKinematics:
    return RobotKinematics(
        urdf_path=str(config.URDF_PATH), target_frame_name=config.IK_TARGET_FRAME, joint_names=config.ARM_JOINTS
    )


def _euler_xyz_to_matrix(rpy_deg: tuple[float, float, float]) -> np.ndarray:
    rx, ry, rz = np.deg2rad(rpy_deg)
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def make_pose(xyz: tuple[float, float, float], rpy_deg=(180.0, 0.0, 0.0)) -> np.ndarray:
    """rpy_deg's default (top-down) is a soft preference only -
    IK_ORIENTATION_WEIGHT=0.0 means it isn't actually enforced, see module
    docstring."""
    pose = np.eye(4)
    pose[:3, 3] = xyz
    pose[:3, :3] = _euler_xyz_to_matrix(rpy_deg)
    return pose


def solve_ik(kin: RobotKinematics, current_joint_deg: np.ndarray, target_xyz: tuple[float, float, float]) -> np.ndarray:
    """Returns joint degrees (5,) for ARM_JOINTS placing the gripper at
    target_xyz. current_joint_deg seeds the iterative solver - pass the
    arm's real current pose for a solution close to where it already is."""
    target_pose = make_pose(target_xyz)
    joints = current_joint_deg[: len(config.ARM_JOINTS)]
    for _ in range(config.IK_ITERATIONS):
        joints = kin.inverse_kinematics(joints, target_pose, orientation_weight=config.IK_ORIENTATION_WEIGHT)
    return joints


def gripper_position(kin: RobotKinematics, joint_deg: np.ndarray) -> np.ndarray:
    return kin.forward_kinematics(joint_deg[: len(config.ARM_JOINTS)])[:3, 3]


class SOArm101:
    """Owns the live connection + every safety layer a move goes through:
    joint-limit clamping, a per-move outright-refusal delta cap, interpolated
    stepping, and CollisionDetected (stall/collision detection)."""

    def __init__(self, port: str = config.FOLLOWER_PORT, robot: SOFollower | None = None):
        """robot: an already-built SOFollower to wrap instead of constructing
        a new one - e.g. a bimanual BiSOFollower's .left_arm/.right_arm. When
        given, connect() skips connecting it (the caller is expected to have
        already connected the parent, which connects both sub-arms)."""
        self._owns_robot = robot is None
        robot_id = "follower_left" if port == config.LEFT_OVERRIDES["FOLLOWER_PORT"] else "follower"
        self.robot = robot if robot is not None else SOFollower(
            SOFollowerRobotConfig(port=port, id=robot_id, use_degrees=True,
                                   max_relative_target=config.MAX_RELATIVE_TARGET_DEG))
        self.kin = build_kinematics()

    def connect(self) -> None:
        """Retries the whole connect(): the same transient "no status packet"
        comm hiccup this session has already hit more than once (always a
        different specific register write, always transient) can also land
        inside lerobot's own configure() - e.g. re-enabling torque on the
        way out of its torque_disabled() context, which uses num_retry=0 and
        has no retry of its own. disconnect() between attempts because
        connect() refuses (@check_if_already_connected) once the bus itself
        is open, which it already is by the time configure() can fail.

        Skipped entirely when self.robot was injected (bimanual case) - the
        caller already connected the parent BiSOFollower, which connects
        both sub-arms."""
        if self._owns_robot:
            last_exc: ConnectionError | None = None
            for attempt in range(3):
                try:
                    self.robot.connect(calibrate=False)
                    break
                except ConnectionError as exc:
                    last_exc = exc
                    print(f"[kinematics] connect() 실패 (시도 {attempt + 1}/3) - 재시도: {exc}")
                    try:
                        self.robot.disconnect()
                    except Exception:
                        pass
                    time.sleep(0.3)
            else:
                raise ConnectionError(f"SOArm101.connect() failed after 3 attempts: {last_exc}")
        self._protect_arm_motors()

    def _protect_arm_motors(self) -> None:
        """so_follower.py's own configure() already caps Max_Torque_Limit/
        Protection_Current/Overload_Torque at 50%/50%/25% to avoid burnout -
        but only for the gripper motor. Mirrors the same values onto the 5
        arm joints: a real stall went undetected on this hardware once
        already (probe_table_height.py's docstring, wrist_roll/shoulder_lift
        both needed large corrections afterward) - firmware-level torque/
        current caps protect the motor even if this session's own
        stall-detection software misses it again.

        Retries a few times: torque_disabled()'s own disable/enable_torque
        calls use num_retry=0, and a "no status packet" comm hiccup right
        after connect()'s own configure() pass (same bus, back to back) has
        been seen for real on this hardware - not worth failing the whole
        connect over when this is an extra safety layer on top of what
        configure() already applied to the gripper."""
        bus = self.robot.bus
        last_exc: ConnectionError | None = None
        for _ in range(3):
            try:
                with bus.torque_disabled():
                    for motor in config.ARM_JOINTS:
                        bus.write("Max_Torque_Limit", motor, 500)
                        bus.write("Protection_Current", motor, 250)
                        bus.write("Overload_Torque", motor, 25)
                return
            except ConnectionError as exc:
                last_exc = exc
                time.sleep(0.2)
        print(f"[kinematics] 팔 관절 모터 보호 설정 실패 (3회 재시도) - 소프트웨어 스톨 감지만으로 진행합니다: {last_exc}")

    def disconnect(self) -> None:
        self.robot.disconnect()

    def release_torque(self) -> None:
        """Cuts torque on every joint - the arm goes limp and can be moved by
        hand (e.g. to physically check/adjust a height before recalibrating).
        Not a controlled stop, just an immediate one - same as robot_control.
        RobotController.emergency_stop()."""
        for motor in config.ALL_JOINTS:
            self.robot.bus.write("Torque_Enable", motor, 0)

    def enable_torque(self) -> None:
        """Re-engages holding torque at wherever the arm currently is after
        release_torque() + hand-positioning it - reads the actual position
        first and re-sends it as the goal (a no-op move, safe with torque
        still off) so torque re-engages exactly where the arm already is,
        instead of yanking it back toward a stale pre-release Goal_Position.
        Same as robot_control.RobotController.enable_torque()."""
        current = self.get_joint_deg()
        self.send_joint_deg(current)
        for motor in config.ALL_JOINTS:
            self.robot.bus.write("Torque_Enable", motor, 1)

    def get_joint_deg(self) -> np.ndarray:
        obs = self.robot.get_observation()
        return np.array([obs[f"{j}.pos"] for j in config.ALL_JOINTS])

    def send_joint_deg(self, joint_deg: np.ndarray) -> None:
        action = {f"{j}.pos": float(v) for j, v in zip(config.ALL_JOINTS, clamp_joint_deg(joint_deg))}
        self.robot.send_action(action)

    def gripper_xyz(self) -> np.ndarray:
        return gripper_position(self.kin, self.get_joint_deg())

    def preview_move(self, xyz: tuple[float, float, float]) -> dict:
        current = self.get_joint_deg()
        target_arm = solve_ik(self.kin, current, xyz)
        target = clamp_joint_deg(np.concatenate([target_arm, current[len(config.ARM_JOINTS) :]]))
        delta = target - current
        return {"current_deg": current, "target_deg": target, "delta_deg": delta,
                "max_abs_delta_deg": float(np.max(np.abs(delta)))}

    def move_to_xyz(self, xyz, steps=20, step_delay_s=0.05, enforce_cap=True, stall_check=True) -> None:
        """enforce_cap=True (default) refuses the whole move outright if the
        total delta exceeds MAX_MOVE_DELTA_DEG - move_to_xyz_converge calls
        with enforce_cap=False, covering large distances via several small
        retried steps instead. stall_check watches actual-vs-commanded
        position and raises CollisionDetected (after retreating) on a real
        block."""
        plan = self.preview_move(xyz)
        if enforce_cap and plan["max_abs_delta_deg"] > config.MAX_MOVE_DELTA_DEG:
            raise RuntimeError(
                f"move_to_xyz refused: largest single-joint delta {plan['max_abs_delta_deg']:.1f}deg "
                f"exceeds the {config.MAX_MOVE_DELTA_DEG}deg cap."
            )
        current, target = plan["current_deg"], plan["target_deg"]
        last_good = current.copy()
        stall_count = 0
        for i in range(1, steps + 1):
            interp = current + (target - current) * (i / steps)
            self.send_joint_deg(interp)
            time.sleep(step_delay_s)
            if not stall_check or i % config.STALL_CHECK_EVERY != 0:
                continue
            actual = self.get_joint_deg()
            lag = float(np.max(np.abs(actual[: len(config.ARM_JOINTS)] - interp[: len(config.ARM_JOINTS)])))
            if lag > config.STALL_THRESHOLD_DEG:
                stall_count += 1
            else:
                stall_count, last_good = 0, actual
            if stall_count >= config.STALL_CONSECUTIVE:
                self.send_joint_deg(last_good)
                time.sleep(0.3)
                raise CollisionDetected(
                    f"move_to_xyz aborted: joint lag {lag:.1f}deg exceeded {config.STALL_THRESHOLD_DEG}deg "
                    f"for {config.STALL_CONSECUTIVE} consecutive checks - retreated to last known-good pose."
                )

    def move_to_xyz_converge(self, xyz, tolerance_m=0.005, max_iters=15) -> np.ndarray:
        """Retries with a fresh delta each time instead of trusting one
        interpolated move to land exactly on target - lerobot's own
        max_relative_target throttles any single send_action regardless of
        what this requests, so one far-away call routinely stalls partway.

        2026-09-01: that throttling is also what was falsely tripping
        CollisionDetected on real hardware for large moves (home return,
        hover approach) with nothing actually in the way - confirmed live,
        arm stopped in open air. move_to_xyz's interpolation schedule keeps
        marching from this call's ORIGINAL position regardless of how much
        lerobot's own per-send clamp is holding actual back, so on a long
        move actual can drift past STALL_THRESHOLD_DEG from the schedule
        with no real block. One stall per outer attempt is now absorbed:
        move_to_xyz already retreats to a safe pose before raising, so the
        next iteration re-plans from there with a much smaller remaining
        delta, clearing a schedule-vs-clamp mismatch on its own. A SECOND
        stall right after that retreat is treated as a real block."""
        consecutive_stalls = 0
        for _ in range(max_iters):
            cur = self.gripper_xyz()
            if np.linalg.norm(np.array(xyz) - cur) <= tolerance_m:
                return cur
            try:
                self.move_to_xyz(xyz, steps=18, step_delay_s=0.06, enforce_cap=False)
                consecutive_stalls = 0
            except CollisionDetected:
                consecutive_stalls += 1
                if consecutive_stalls >= 2:
                    raise
            time.sleep(0.15)
        return self.gripper_xyz()

    def nudge_xy(self, dx: float, dy: float, steps=6, step_delay_s=0.03, stall_check=True) -> np.ndarray:
        cur = self.gripper_xyz()
        self.move_to_xyz((cur[0] + dx, cur[1] + dy, cur[2]), steps=steps, step_delay_s=step_delay_s,
                          enforce_cap=False, stall_check=stall_check)
        return self.gripper_xyz()

    def move_z(self, dz: float, steps=10, step_delay_s=0.04, stall_check=True) -> np.ndarray:
        cur = self.gripper_xyz()
        self.move_to_xyz((cur[0], cur[1], cur[2] + dz), steps=steps, step_delay_s=step_delay_s,
                          enforce_cap=False, stall_check=stall_check)
        return self.gripper_xyz()
