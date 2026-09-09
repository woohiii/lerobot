"""Hardware-free tests for production adapter seams."""

from __future__ import annotations

import numpy as np
import pytest

from custom_scripts.vision_pick_place.so101_graspnet_pick_and_place.adapters import (
    GeminiRoiDetector,
    ObjectNotDetectedError,
    RgbdCameraAdapter,
)
from custom_scripts.vision_pick_place.so101_graspnet_pick_and_place.models import CameraIntrinsics


def test_gemini_detector_converts_xywh_detection_to_roi() -> None:
    detector = GeminiRoiDetector(detector=lambda image, prompt: (11.0, 12.0, 30.0, 40.0))

    roi = detector.detect(np.zeros((80, 100, 3), dtype=np.uint8), "tennis ball")

    assert (roi.x0, roi.y0, roi.x1, roi.y1) == (11, 12, 41, 52)


def test_gemini_detector_raises_on_missing_detection() -> None:
    detector = GeminiRoiDetector(detector=lambda image, prompt: None)

    with pytest.raises(ObjectNotDetectedError):
        detector.detect(np.zeros((10, 10, 3), dtype=np.uint8), "object")


def test_rgbd_camera_adapter_rejects_unregistered_stream() -> None:
    source = RgbdCameraAdapter(
        camera=object(),
        intrinsics=CameraIntrinsics(100.0, 100.0, 1.0, 1.0),
        registration_verified=False,
    )

    with pytest.raises(RuntimeError, match="registration"):
        source.read()
