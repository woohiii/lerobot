"""Tests for the hand-eye calibration command."""

from __future__ import annotations

import json

import numpy as np
import pytest

from custom_scripts.vision_pick_place.so101_graspnet_pick_and_place.calibrate_hand_eye import main


def _write_points(tmp_path):
    camera = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    base = camera + np.array([0.2, -0.1, 0.3])
    camera_path = tmp_path / "camera.npy"
    base_path = tmp_path / "base.npy"
    np.save(camera_path, camera)
    np.save(base_path, base)
    return camera_path, base_path


def test_calibration_cli_writes_transform_and_error(tmp_path) -> None:
    camera_path, base_path = _write_points(tmp_path)
    output = tmp_path / "hand_eye.json"

    assert main([str(camera_path), str(base_path), "--output", str(output)]) == 0
    document = json.loads(output.read_text())
    assert document["rms_error_m"] == pytest.approx(0.0)
    assert np.asarray(document["camera_to_base"]).shape == (4, 4)


def test_calibration_cli_rejects_error_above_threshold(tmp_path) -> None:
    camera_path, base_path = _write_points(tmp_path)
    noisy = np.load(base_path)
    noisy[0, 0] += 1.0
    np.save(base_path, noisy)

    with pytest.raises(SystemExit):
        main(
            [
                str(camera_path),
                str(base_path),
                "--output",
                str(tmp_path / "out.json"),
                "--max-rms-m",
                "0.01",
            ]
        )
