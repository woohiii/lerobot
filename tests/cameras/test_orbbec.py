import time
from unittest.mock import Mock

import numpy as np
import pytest

from lerobot.cameras.orbbec.camera_orbbec import OrbbecCamera
from lerobot.cameras.orbbec.configuration_orbbec import OrbbecCameraConfig


def test_depth_visualization_is_rgb_and_preserves_invalid_pixels():
    camera = OrbbecCamera(OrbbecCameraConfig(depth_as_viz=True))
    depth = np.array([[0, 200, 2000]], dtype=np.uint16)

    visualized = camera._depth_to_viz(depth)

    assert visualized.shape == (1, 3, 3)
    assert visualized.dtype == np.uint8
    np.testing.assert_array_equal(visualized[0, 0], [0, 0, 0])
    assert not np.array_equal(visualized[0, 1], visualized[0, 2])


def test_depth_visualization_requires_valid_range():
    with pytest.raises(ValueError, match="Depth visualization range"):
        OrbbecCameraConfig(depth_viz_min_mm=1000, depth_viz_max_mm=1000)


def test_camera_rejects_stale_depth_frame():
    camera = OrbbecCamera(OrbbecCameraConfig())
    camera._thread = Mock(is_alive=Mock(return_value=True))
    camera._depth = np.zeros((240, 320, 1), dtype=np.uint16)
    camera._timestamp = time.monotonic() - 1

    with pytest.raises(TimeoutError, match="stale"):
        camera.read_latest_depth(max_age_ms=10)
