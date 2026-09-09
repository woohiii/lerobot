"""Tests for the low-speed hover-only validation command."""

from __future__ import annotations

import numpy as np
import pytest

from custom_scripts.vision_pick_place.so101_graspnet_pick_and_place.hover_test import (
    compute_hover_target,
    validate_calibration_rms,
    validate_hold_mode,
)


def test_compute_hover_target_transforms_camera_point_and_adds_height() -> None:
    transform = np.eye(4)
    transform[:3, 3] = [0.1, -0.2, 0.3]

    result = compute_hover_target((0.2, 0.0, 0.4), transform, hover_offset_m=0.1)

    assert result == pytest.approx((0.3, -0.2, 0.8))


def test_validate_calibration_rms_rejects_unsafe_value() -> None:
    with pytest.raises(ValueError, match="RMS"):
        validate_calibration_rms(0.031, max_rms_m=0.03)


def test_validate_hold_mode_requires_execute() -> None:
    with pytest.raises(ValueError, match="execute"):
        validate_hold_mode(execute=False, hold_position=True)
