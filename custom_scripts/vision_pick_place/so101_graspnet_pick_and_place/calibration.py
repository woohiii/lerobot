"""Rigid camera-to-base calibration from corresponding 3-D points."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


class CalibrationError(ValueError):
    """Raised when point correspondences cannot define a rigid transform."""


@dataclass(frozen=True)
class CalibrationResult:
    transform: np.ndarray
    rms_error_m: float


@dataclass(frozen=True)
class JointTcpCalibrationResult:
    transform: np.ndarray
    tcp_offset_m: np.ndarray
    rms_error_m: float


def solve_rigid_transform(camera_points: np.ndarray, base_points: np.ndarray) -> CalibrationResult:
    """Solve ``base = R @ camera + t`` using the Kabsch algorithm."""
    camera = np.asarray(camera_points, dtype=float)
    base = np.asarray(base_points, dtype=float)
    if camera.shape != base.shape:
        raise CalibrationError("camera and base points must have the same shape")
    if camera.ndim != 2 or camera.shape[1] != 3 or len(camera) < 3:
        raise CalibrationError("at least three 3-D points are required")
    if not np.isfinite(camera).all() or not np.isfinite(base).all():
        raise CalibrationError("calibration points must be finite")
    centered_camera = camera - camera.mean(axis=0)
    centered_base = base - base.mean(axis=0)
    if np.linalg.matrix_rank(centered_camera) < 2 or np.linalg.matrix_rank(centered_base) < 2:
        raise CalibrationError("calibration points are degenerate")
    _, _, vh = np.linalg.svd(centered_camera.T @ centered_base)
    u, _, _ = np.linalg.svd(centered_camera.T @ centered_base)
    rotation = vh.T @ u.T
    if np.linalg.det(rotation) < 0:
        vh[-1] *= -1
        rotation = vh.T @ u.T
    translation = base.mean(axis=0) - rotation @ camera.mean(axis=0)
    transformed = camera @ rotation.T + translation
    rms = float(np.sqrt(np.mean(np.sum((transformed - base) ** 2, axis=1))))
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return CalibrationResult(matrix, rms)


def solve_joint_tcp_transform(
    camera_points: np.ndarray, joint_points: np.ndarray, kin
) -> JointTcpCalibrationResult:
    """Jointly solve camera-to-base pose and a fixed flange-to-TCP offset."""
    camera = np.asarray(camera_points, dtype=float)
    joints = np.asarray(joint_points, dtype=float)
    if camera.ndim != 2 or camera.shape[1] != 3:
        raise CalibrationError("camera points must be an N x 3 matrix")
    if joints.ndim != 2 or joints.shape[0] != len(camera) or joints.shape[1] < 5:
        raise CalibrationError("joint points must match camera points as an N x 5+ matrix")
    if len(camera) < 6:
        raise CalibrationError("at least six joint TCP points are required")
    if not np.isfinite(camera).all() or not np.isfinite(joints).all():
        raise CalibrationError("calibration points must be finite")

    flange_poses = np.asarray([kin.forward_kinematics(row[:5]) for row in joints])
    if flange_poses.shape != (len(camera), 4, 4) or not np.isfinite(flange_poses).all():
        raise CalibrationError("kinematics returned invalid flange poses")
    flange_positions = flange_poses[:, :3, 3]
    if np.linalg.matrix_rank(camera - camera.mean(axis=0)) < 2:
        raise CalibrationError("camera points are degenerate")
    if np.linalg.matrix_rank(flange_positions - flange_positions.mean(axis=0)) < 2:
        raise CalibrationError("flange points are degenerate")

    initial = solve_rigid_transform(camera, flange_positions).transform
    initial_rotation = Rotation.from_matrix(initial[:3, :3]).as_rotvec()
    initial_parameters = np.concatenate([initial_rotation, initial[:3, 3], np.zeros(3)])

    def residual(parameters: np.ndarray) -> np.ndarray:
        camera_rotation = Rotation.from_rotvec(parameters[:3]).as_matrix()
        translation = parameters[3:6]
        offset = parameters[6:9]
        predicted_tcp = flange_poses[:, :3, 3] + np.einsum("nij,j->ni", flange_poses[:, :3, :3], offset)
        predicted_camera = (camera_rotation @ camera.T).T + translation
        return (predicted_tcp - predicted_camera).ravel()

    result = least_squares(residual, initial_parameters, method="lm" if len(camera) >= 6 else "trf")
    if not result.success or not np.isfinite(result.x).all():
        raise CalibrationError(f"joint TCP optimization failed: {result.message}")
    rotation = Rotation.from_rotvec(result.x[:3]).as_matrix()
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = result.x[3:6]
    point_residuals = residual(result.x).reshape(-1, 3)
    rms = float(np.sqrt(np.mean(np.sum(point_residuals**2, axis=1))))
    return JointTcpCalibrationResult(transform, result.x[6:9].copy(), rms)
