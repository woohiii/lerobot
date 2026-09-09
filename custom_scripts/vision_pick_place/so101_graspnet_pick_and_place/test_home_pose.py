import numpy as np
import pytest

from .home_pose import HOME_JOINTS_DEG, home_pose_error_deg, is_at_home


def test_home_pose_matches_itself() -> None:
    assert home_pose_error_deg(HOME_JOINTS_DEG) == 0.0
    assert is_at_home(HOME_JOINTS_DEG, 5.0)


def test_home_pose_rejects_large_error() -> None:
    joints = HOME_JOINTS_DEG.copy()
    joints[2] += 5.1
    assert not is_at_home(joints, 5.0)


def test_home_pose_requires_six_finite_joints() -> None:
    with pytest.raises(ValueError, match="six finite"):
        home_pose_error_deg(np.zeros(5))
