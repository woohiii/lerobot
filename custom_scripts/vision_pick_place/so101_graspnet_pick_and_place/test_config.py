from __future__ import annotations

import json

import numpy as np
import pytest

from custom_scripts.vision_pick_place.so101_graspnet_pick_and_place.config import (
    ConfigError,
    load_pipeline_config,
)


def _document(*, calibrated: bool) -> dict[str, object]:
    return {
        "calibration_confirmed": calibrated,
        "camera_intrinsics": {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0},
        "camera_to_base": np.eye(4).tolist(),
        "center_exclusion_half_width_m": 0.05,
        "arms": [
            {
                "name": "left",
                "port": "/dev/left",
                "workspace": [-0.5, -0.06, -0.3, 0.3, 0.0, 0.4],
                "bin_pose": [-0.2, 0.1, 0.15],
            }
        ],
    }


def test_load_pipeline_config_requires_explicit_calibration_for_execution(tmp_path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(_document(calibrated=False)))

    config = load_pipeline_config(path)
    assert config.arms[0].port == "/dev/left"
    with pytest.raises(ConfigError, match="calibration"):
        config.require_execution_ready()


def test_load_pipeline_config_rejects_non_homogeneous_transform(tmp_path) -> None:
    document = _document(calibrated=True)
    document["camera_to_base"] = np.zeros((4, 4)).tolist()
    path = tmp_path / "config.json"
    path.write_text(json.dumps(document))

    with pytest.raises(ConfigError, match="homogeneous"):
        load_pipeline_config(path)


def test_load_pipeline_config_does_not_treat_string_false_as_confirmed(tmp_path) -> None:
    document = _document(calibrated=True)
    document["calibration_confirmed"] = "false"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(document))

    with pytest.raises(ConfigError, match="boolean"):
        load_pipeline_config(path)
