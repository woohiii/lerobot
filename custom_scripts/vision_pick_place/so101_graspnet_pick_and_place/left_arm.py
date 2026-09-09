"""Load SOArm101 with the left-arm-only configuration namespace."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np


def _load_left_arm_module():
    root = Path(__file__).resolve().parent
    vision_dir = root.parent
    runtime_dir = root / "left_arm_runtime"
    task_dir = vision_dir / "task_trash_to_bin"
    for path in reversed((str(runtime_dir), str(vision_dir), str(task_dir))):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)
    # The shared module uses ``import config``.  Fail closed if another arm's
    # config was imported first in this Python process.
    imported = sys.modules.get("config")
    if imported is not None and Path(imported.__file__).resolve().parent != runtime_dir:
        raise RuntimeError("right-arm config is already loaded; start a fresh process for the left arm")
    from task_trash_to_bin import kinematics

    return kinematics


_kinematics = _load_left_arm_module()
_SharedSOArm101 = _kinematics.SOArm101


class LeftSOArm101(_SharedSOArm101):
    """SOArm101 bound to the left arm's independent LeRobot calibration ID."""

    def __init__(self, port: str = "/dev/ttyACM1", robot=None):
        super().__init__(
            port=port,
            robot=robot,
            robot_id="so101_left_follower",
            disable_torque_on_disconnect=False,
        )

    def send_arm_joint_deg(self, arm_joint_deg: np.ndarray) -> None:
        """Send an approved five-axis target while retaining the live gripper percent."""
        target = np.asarray(arm_joint_deg, dtype=float)
        if target.shape != (5,) or not np.isfinite(target).all():
            raise ValueError("arm joint target must contain five finite values")
        current = self.get_joint_deg()
        self.send_joint_deg(np.concatenate((target, current[5:])))

    def move_validated_arm_joint_deg(
        self, arm_joint_deg: np.ndarray, *, tolerance_deg: float = 1.0, timeout_s: float = 3.0
    ) -> None:
        """Pace one preflight-approved edge and reject clipping, stalls, or missed arrival."""
        target = np.asarray(arm_joint_deg, dtype=float)
        current = self.get_joint_deg()
        if target.shape != (5,) or current.shape != (6,) or not np.isfinite(current).all() or not np.isfinite(target).all():
            raise RuntimeError("validated motion requires finite five-axis target and six-axis feedback")
        limits = [_kinematics.config.JOINT_LIMITS_DEG[name] for name in _kinematics.config.ARM_JOINTS]
        if any(not low <= value <= high for value, (low, high) in zip(target, limits, strict=True)):
            raise RuntimeError("validated motion target exceeds joint limits")
        if float(np.max(np.abs(target - current[:5]))) > _kinematics.config.MAX_RELATIVE_TARGET_DEG:
            raise RuntimeError("live joint state no longer matches the prevalidated edge")
        steps = max(1, int(np.ceil(float(np.max(np.abs(target - current[:5]))) / 2.0)))
        for step in range(1, steps + 1):
            intermediate = current[:5] + (target - current[:5]) * (step / steps)
            self.send_arm_joint_deg(intermediate)
            self._wait_for_validated_arrival(intermediate, tolerance_deg=tolerance_deg, timeout_s=timeout_s)

    def _wait_for_validated_arrival(
        self, target: np.ndarray, *, tolerance_deg: float, timeout_s: float
    ) -> None:
        deadline = time.monotonic() + timeout_s
        previous = self.get_joint_deg()[:5]
        stalled = 0
        while time.monotonic() < deadline:
            time.sleep(0.05)
            actual = self.get_joint_deg()
            if actual.shape != (6,) or not np.isfinite(actual).all():
                raise RuntimeError("non-finite joint feedback during validated motion")
            error = float(np.max(np.abs(target - actual[:5])))
            if error <= tolerance_deg:
                return
            if float(np.max(np.abs(actual[:5] - previous))) < 0.1:
                stalled += 1
                if stalled >= 3:
                    raise RuntimeError("validated motion stalled before arrival")
            else:
                stalled = 0
            previous = actual[:5]
        raise RuntimeError("validated motion timed out before arrival")

    def hold_current_pose(self) -> None:
        """Hold finite current feedback; never reverse toward a historical endpoint on abort."""
        current = self.get_joint_deg()
        if current.shape != (6,) or not np.isfinite(current).all():
            raise RuntimeError("cannot hold without finite current joint feedback")
        self.send_joint_deg(current)


def build_left_kinematics():
    """Construct FK/IK using the isolated left-arm configuration."""
    return _kinematics.build_kinematics()
