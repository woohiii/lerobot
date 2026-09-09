"""Validation tests for pipeline data contracts."""

from __future__ import annotations

import math

import pytest

from custom_scripts.vision_pick_place.so101_graspnet_pick_and_place.models import CameraIntrinsics


def test_camera_intrinsics_rejects_non_finite_values() -> None:
    """Non-finite calibration values cannot reach motion planning."""
    with pytest.raises(ValueError, match="finite"):
        CameraIntrinsics(math.nan, 1.0, 0.0, 0.0)
