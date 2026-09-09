"""Tests for joint TCP calibration output validation."""

from __future__ import annotations

import json

import numpy as np
import pytest

from custom_scripts.vision_pick_place.so101_graspnet_pick_and_place.calibrate_joint_tcp import (
    save_joint_calibration,
)
from custom_scripts.vision_pick_place.so101_graspnet_pick_and_place.calibration import (
    JointTcpCalibrationResult,
)


def _result(rms: float) -> JointTcpCalibrationResult:
    return JointTcpCalibrationResult(np.eye(4), np.array([0.01, 0.02, 0.03]), rms)


def test_save_joint_calibration_rejects_rms_without_overwriting(tmp_path) -> None:
    output = tmp_path / "hand_eye.json"
    output.write_text("old")

    with pytest.raises(ValueError, match="exceeds threshold"):
        save_joint_calibration(_result(0.02), output, max_rms_m=0.005)

    assert output.read_text() == "old"


def test_save_joint_calibration_writes_tcp_offset(tmp_path) -> None:
    output = tmp_path / "hand_eye.json"

    save_joint_calibration(_result(0.004), output, max_rms_m=0.005)
    payload = json.loads(output.read_text())

    assert payload["calibration_type"] == "joint_tcp"
    assert payload["tcp_offset_m"] == [0.01, 0.02, 0.03]
    assert payload["rms_error_m"] == pytest.approx(0.004)
