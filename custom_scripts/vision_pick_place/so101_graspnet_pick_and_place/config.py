"""Explicit, fail-closed calibration and workspace configuration loading."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .models import CameraIntrinsics
from .routing import ArmRoute, WorkspaceBounds


class ConfigError(ValueError):
    """Raised for malformed or unsafe pipeline configuration."""


@dataclass(frozen=True)
class PipelineConfig:
    calibration_confirmed: bool
    camera_intrinsics: CameraIntrinsics
    camera_to_base: np.ndarray
    center_exclusion_half_width_m: float
    arms: tuple[ArmRoute, ...]

    def require_execution_ready(self) -> None:
        if not self.calibration_confirmed:
            raise ConfigError("execution requires confirmed hand-eye calibration")


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    """Load and validate the JSON file that gates physical execution."""

    try:
        document = json.loads(Path(path).read_text())
        intrinsics = CameraIntrinsics(**document["camera_intrinsics"])
        transform = np.asarray(document["camera_to_base"], dtype=float)
        if (
            transform.shape != (4, 4)
            or not np.isfinite(transform).all()
            or not np.allclose(transform[3], (0, 0, 0, 1))
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(transform[:3, :3]), 1.0, atol=1e-5)
        ):
            raise ConfigError("camera_to_base must be a 4x4 homogeneous transform")
        calibration_confirmed = document["calibration_confirmed"]
        if not isinstance(calibration_confirmed, bool):
            raise ConfigError("calibration_confirmed must be a boolean")
        arms = tuple(_arm_route(item) for item in document["arms"])
        if not arms:
            raise ConfigError("at least one arm is required")
        center_width = float(document["center_exclusion_half_width_m"])
        if not np.isfinite(center_width) or center_width < 0:
            raise ConfigError("center exclusion half-width must be non-negative")
        return PipelineConfig(
            calibration_confirmed=calibration_confirmed,
            camera_intrinsics=intrinsics,
            camera_to_base=transform,
            center_exclusion_half_width_m=center_width,
            arms=arms,
        )
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ConfigError(f"invalid pipeline config: {exc}") from exc


def _arm_route(item: dict[str, Any]) -> ArmRoute:
    workspace = item["workspace"]
    if len(workspace) != 6:
        raise ConfigError("workspace must contain six bounds")
    bin_pose = tuple(float(value) for value in item["bin_pose"])
    if len(bin_pose) != 3:
        raise ConfigError("bin_pose must contain three coordinates")
    return ArmRoute(
        name=str(item["name"]),
        workspace=WorkspaceBounds(*(float(value) for value in workspace)),
        bin_pose=bin_pose,
        port=str(item["port"]) if item.get("port") is not None else None,
    )
