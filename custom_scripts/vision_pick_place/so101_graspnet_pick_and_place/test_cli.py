"""CLI preflight tests."""

from __future__ import annotations

import json

from custom_scripts.vision_pick_place.so101_graspnet_pick_and_place.cli import main


def test_dry_run_reports_calibration_gate_without_hardware(tmp_path, capsys) -> None:
    config = {
        "calibration_confirmed": False,
        "camera_intrinsics": {"fx": 500, "fy": 500, "cx": 320, "cy": 240},
        "camera_to_base": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        "center_exclusion_half_width_m": 0.05,
        "arms": [{"name": "left", "workspace": [-1, -0.1, -1, 1, 0, 1], "bin_pose": [-0.2, 0, 0.2]}],
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))

    assert main(["--config", str(path), "--dry-run"]) == 0
    assert "dry-run" in capsys.readouterr().out
