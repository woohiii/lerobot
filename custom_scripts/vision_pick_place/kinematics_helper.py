"""Thin wrapper around lerobot's placo-based RobotKinematics for the SO-101 follower.

Uses the SO-ARM101 URDF (borrowed from the isaac_so_arm101 project, which already
has the `gripper_frame_link` frame lerobot's kinematics module expects as its
default target) so we get real forward/inverse kinematics without hand-deriving
the arm's geometry.
"""

from pathlib import Path

import numpy as np

from lerobot.model import RobotKinematics

URDF_PATH = Path(__file__).parent / "so101_urdf" / "so_arm101.urdf"

# Order matters: must match the joint_names passed to RobotKinematics, and is the
# order forward/inverse_kinematics expect their joint-position arrays in. Excludes
# "gripper" (the jaw open/close DOF) - only the 5 arm joints affect the gripper's
# pose in space.
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def build_kinematics() -> RobotKinematics:
    return RobotKinematics(
        urdf_path=str(URDF_PATH),
        target_frame_name="gripper_frame_link",
        joint_names=ARM_JOINTS,
    )


def _euler_xyz_to_matrix(rpy_deg: tuple[float, float, float]) -> np.ndarray:
    """Intrinsic X-Y-Z Euler angles (degrees) -> 3x3 rotation matrix. lerobot's own
    Rotation helper doesn't expose from_euler and scipy isn't installed in this venv,
    so this is hand-rolled instead of adding a new dependency for one conversion."""
    rx, ry, rz = np.deg2rad(rpy_deg)
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def make_pose(xyz: tuple[float, float, float], rpy_deg: tuple[float, float, float] = (180.0, 0.0, 0.0)) -> np.ndarray:
    """Builds a 4x4 target pose for inverse_kinematics from a position (meters, robot
    base frame) and an orientation given as roll/pitch/yaw in degrees.

    Default orientation (180, 0, 0) points the gripper straight down at the table -
    the only orientation this pick-and-place task actually needs. IK is called with
    a low orientation_weight, so this is a soft preference, not a hard constraint;
    exact roll/pitch matters less than getting position right for a top-down grasp.
    """
    pose = np.eye(4)
    pose[:3, 3] = xyz
    pose[:3, :3] = _euler_xyz_to_matrix(rpy_deg)
    return pose


def solve_ik(
    kin: RobotKinematics,
    current_joint_deg: np.ndarray,
    target_xyz: tuple[float, float, float],
    iterations: int = 6,
) -> np.ndarray:
    """Returns joint degrees (5,) for ARM_JOINTS that put the gripper at target_xyz,
    gripper pointing down. current_joint_deg is the IK solver's initial guess - pass
    the arm's actual current pose for a solution close to where it already is.

    placo's solver is iterative and one call rarely lands exactly on target -
    measured ~9cm of position error after a single pass on a target ~15cm from the
    initial guess, converging to <1mm by the 4th-5th pass when each pass's output is
    fed back in as the next guess. orientation_weight=0.0 (position-only): the task
    only needs a top-down approach, and letting orientation float gives the solver
    more room to actually hit the requested position.
    """
    target_pose = make_pose(target_xyz)
    joints = current_joint_deg[: len(ARM_JOINTS)]
    for _ in range(iterations):
        joints = kin.inverse_kinematics(joints, target_pose, orientation_weight=0.0)
    return joints


def gripper_position(kin: RobotKinematics, joint_deg: np.ndarray) -> np.ndarray:
    """Current (x, y, z) of gripper_frame_link in the robot base frame, in meters."""
    return kin.forward_kinematics(joint_deg[: len(ARM_JOINTS)])[:3, 3]
