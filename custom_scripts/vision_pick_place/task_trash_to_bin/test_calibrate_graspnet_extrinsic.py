"""GraspNet 외부 파라미터 순수 수학 함수 테스트."""

from __future__ import annotations

import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate_graspnet_extrinsic import estimate_rigid_transform


def test_estimate_rigid_transform_recovers_known_rotation_and_translation() -> None:
    """합성된 비평면 점에서 원래 강체변환을 정확히 복원한다."""
    angle = np.deg2rad(27.0)
    rotation_expected = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translation_expected = np.array([0.31, -0.12, 0.44])
    camera_points = np.array(
        [
            [-0.20, -0.10, 0.35],
            [0.18, -0.13, 0.40],
            [-0.15, 0.22, 0.55],
            [0.25, 0.19, 0.62],
            [0.02, -0.03, 0.85],
            [-0.28, 0.16, 0.72],
            [0.13, 0.27, 0.48],
        ]
    )
    robot_points = (rotation_expected @ camera_points.T).T + translation_expected

    rotation, translation, mean_error = estimate_rigid_transform(camera_points, robot_points)

    np.testing.assert_allclose(rotation, rotation_expected, atol=1e-12)
    np.testing.assert_allclose(translation, translation_expected, atol=1e-12)
    assert mean_error < 1e-12
