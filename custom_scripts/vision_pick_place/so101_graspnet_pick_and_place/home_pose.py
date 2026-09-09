"""Shared, measured safe-start pose for the left SO-101 follower arm."""

import numpy as np

# Measured after the 2026-09-09 left-follower motor recalibration.
HOME_JOINTS_DEG = np.array([-4.66, 3.25, 6.24, 86.51, 0.31, 99.16], dtype=float)


def home_pose_error_deg(joints_deg: np.ndarray) -> float:
    """Return the largest absolute joint error from the recorded safe pose."""
    joints = np.asarray(joints_deg, dtype=float)
    if joints.shape != HOME_JOINTS_DEG.shape or not np.isfinite(joints).all():
        raise ValueError("expected six finite joint positions")
    return float(np.max(np.abs(joints - HOME_JOINTS_DEG)))


def is_at_home(joints_deg: np.ndarray, tolerance_deg: float) -> bool:
    if tolerance_deg <= 0:
        raise ValueError("home tolerance must be positive")
    return home_pose_error_deg(joints_deg) <= tolerance_deg
