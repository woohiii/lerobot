"""Tests for camera-point conversion used by object-point capture."""

from __future__ import annotations

import numpy as np
import pytest

from custom_scripts.vision_pick_place.so101_graspnet_pick_and_place.capture_object_point import (
    select_valid_depth,
)


def test_select_valid_depth_returns_clicked_pixel() -> None:
    depth = np.zeros((4, 4), dtype=np.uint16)
    depth[2, 1] = 550

    assert select_valid_depth(depth, 1, 2) == (1, 2, 550.0)


def test_select_valid_depth_rejects_zero_pixel() -> None:
    with pytest.raises(ValueError, match="depth"):
        select_valid_depth(np.zeros((4, 4), dtype=np.uint16), 1, 2)
