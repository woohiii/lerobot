"""Tests for rigid hand-eye calibration from point correspondences."""

from __future__ import annotations

import numpy as np
import pytest

from custom_scripts.vision_pick_place.so101_graspnet_pick_and_place.calibration import (
    CalibrationError,
    solve_joint_tcp_transform,
    solve_rigid_transform,
)


def test_solve_rigid_transform_recovers_rotation_and_translation() -> None:
    camera = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    translation = np.array([0.4, -0.2, 0.7])
    base = camera @ rotation.T + translation

    result = solve_rigid_transform(camera, base)

    assert result.transform[:3, :3] == pytest.approx(rotation)
    assert result.transform[:3, 3] == pytest.approx(translation)
    assert result.rms_error_m == pytest.approx(0.0)


def test_solve_rigid_transform_rejects_degenerate_or_mismatched_points() -> None:
    with pytest.raises(CalibrationError, match="at least three"):
        solve_rigid_transform(np.zeros((2, 3)), np.zeros((2, 3)))
    with pytest.raises(CalibrationError, match="same shape"):
        solve_rigid_transform(np.zeros((3, 3)), np.zeros((4, 3)))


class _SyntheticKinematics:
    def forward_kinematics(self, joints: np.ndarray) -> np.ndarray:
        pose = np.eye(4)
        pose[:3, 3] = joints[:3]
        angle, tilt = joints[3], joints[4]
        rz = np.array(
            [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]
        )
        ry = np.array(
            [[np.cos(tilt), 0.0, np.sin(tilt)], [0.0, 1.0, 0.0], [-np.sin(tilt), 0.0, np.cos(tilt)]]
        )
        pose[:3, :3] = rz @ ry
        return pose


def test_joint_tcp_transform_recovers_camera_transform_and_offset() -> None:
    camera = np.array(
        [
            [-0.2, -0.1, 0.5],
            [0.1, -0.15, 0.55],
            [0.2, 0.1, 0.6],
            [-0.1, 0.2, 0.52],
            [0.3, 0.25, 0.58],
            [-0.25, 0.15, 0.48],
        ]
    )
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    translation = np.array([0.4, -0.2, 0.3])
    offset = np.array([0.02, -0.01, 0.04])
    tcp_base = (rotation @ camera.T).T + translation
    angles = np.linspace(-0.8, 0.9, len(camera))
    tilts = np.linspace(-0.35, 0.4, len(camera))
    joints = np.column_stack([np.zeros((len(camera), 3)), angles, tilts])
    flange = np.empty_like(tcp_base)
    for i, angle in enumerate(angles):
        rotation_z = np.array(
            [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]
        )
        tilt = tilts[i]
        rotation_y = np.array(
            [[np.cos(tilt), 0.0, np.sin(tilt)], [0.0, 1.0, 0.0], [-np.sin(tilt), 0.0, np.cos(tilt)]]
        )
        flange[i] = tcp_base[i] - rotation_z @ rotation_y @ offset
    joints[:, :3] = flange

    result = solve_joint_tcp_transform(camera, joints, _SyntheticKinematics())

    assert result.rms_error_m < 1e-8
    assert result.tcp_offset_m == pytest.approx(offset, abs=1e-6)
    assert result.transform[:3, :3] == pytest.approx(rotation, abs=1e-6)
    assert result.transform[:3, 3] == pytest.approx(translation, abs=1e-6)


def test_joint_tcp_transform_requires_six_points() -> None:
    with pytest.raises(CalibrationError, match="six"):
        solve_joint_tcp_transform(np.zeros((5, 3)), np.zeros((5, 5)), _SyntheticKinematics())
