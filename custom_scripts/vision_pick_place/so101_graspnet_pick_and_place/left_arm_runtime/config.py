"""Left SO-101-only kinematic and motor safety configuration.

This module is intentionally named ``config`` because the shared SOArm101
implementation imports that name directly.  It must only be loaded through
``left_arm.py``; do not add this directory globally to PYTHONPATH.
"""

from pathlib import Path

VISION_DIR = Path(__file__).resolve().parents[2]
FOLLOWER_PORT = "/dev/ttyACM1"
CALIBRATION_ID = "so101_left_follower"
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
ALL_JOINTS = ARM_JOINTS + ["gripper"]
JOINT_LIMITS_DEG = {
    "shoulder_pan": (-118.0, 118.0),
    "shoulder_lift": (-105.0, 105.0),
    "elbow_flex": (-98.0, 98.0),
    "wrist_flex": (-102.0, 102.0),
    "wrist_roll": (-179.0, 179.0),
    "gripper": (-8.0, 99.0),
}
MAX_RELATIVE_TARGET_DEG = 10.0
MAX_MOVE_DELTA_DEG = 10.0
STALL_THRESHOLD_DEG = 10.0
STALL_CHECK_EVERY = 3
STALL_CONSECUTIVE = 3
URDF_PATH = VISION_DIR / "so101_urdf" / "so_arm101.urdf"
IK_TARGET_FRAME = "gripper_frame_link"
IK_ITERATIONS = 6
