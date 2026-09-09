from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from . import local_cli
from .local_cli import format_preview
from .local_planner import LocalPickPlacePlan
from .models import CameraIntrinsics, DetectedObject, RgbdFrame, TaskCommand
from .routing import ArmRoute, WorkspaceBounds


def test_format_preview_contains_plan_and_explicit_confirmation() -> None:
    source = DetectedObject("red block", (1, 2, 4, 5), 0.9)
    destination = DetectedObject("blue tray", (5, 6, 9, 9), 0.8, "destination")
    plan = LocalPickPlacePlan(
        TaskCommand(source, destination),
        (0.1, 0.2, 0.3),
        (0.2, 0.1, 0.4),
        ArmRoute("left", WorkspaceBounds(-1, -0.05, -1, 1, 0, 1), (0, 0, 0)),
    )

    preview = format_preview(plan)

    assert "red block" in preview
    assert "blue tray" in preview
    assert "left" in preview
    assert "yes" in preview
    assert "motor" in preview.lower()


def test_main_refuses_single_frame_input_before_calling_stable_planner(monkeypatch, tmp_path) -> None:
    frame = RgbdFrame(
        rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        depth=np.full((2, 2), 500, dtype=np.uint16),
        intrinsics=CameraIntrinsics(100, 100, 0, 0),
    )
    command = TaskCommand(
        DetectedObject("block", (0, 0, 1, 1), 0.9),
        DetectedObject("tray", (1, 1, 2, 2), 0.9, "destination"),
    )
    config = SimpleNamespace(
        camera_to_base=np.eye(4),
        arms=[],
        center_exclusion_half_width_m=0.02,
    )

    monkeypatch.setattr(local_cli, "load_pipeline_config", lambda _: config)
    monkeypatch.setattr(local_cli, "_load_frame", lambda *_: frame)
    monkeypatch.setattr(
        local_cli,
        "OllamaVlmClient",
        lambda **_: SimpleNamespace(detect_objects=lambda *_: [], parse_task=lambda *_: command),
    )
    monkeypatch.setattr(
        local_cli,
        "plan_local_task",
        lambda *_args, **_kwargs: pytest.fail("single-frame CLI must not call the stable planner"),
    )

    with pytest.raises(SystemExit) as excinfo:
        local_cli.main(["--config", str(tmp_path / "config.json"), "--command", "block to tray", "--dry-run", "--snapshot", str(tmp_path)])

    assert excinfo.value.code == 2


def test_connected_left_execution_preflights_and_registers_retreat_before_executor_moves(monkeypatch) -> None:
    """Fails if the execute path can command the executor before validated retreat registration."""
    plan = LocalPickPlacePlan(
        command=None,  # type: ignore[arg-type]
        source_xyz=(0.1, 0.2, 0.03),
        destination_xyz=(0.2, 0.1, 0.04),
        arm=ArmRoute("left", WorkspaceBounds(-1, 1, -1, 1, -1, 1), (0, 0, 0), port="fake"),
    )
    calls: list[str] = []
    arm = SimpleNamespace(gripper_xyz=lambda: np.array([0.0, 0.0, 0.2]))
    waypoints = SimpleNamespace(retreat=(0.2, 0.1, 0.14))

    monkeypatch.setattr(local_cli, "build_and_validate_waypoints", lambda *args, **kwargs: calls.append("preflight") or waypoints)

    class Adapter:
        def __init__(self, arm) -> None:
            calls.append("adapter")

        def set_validated_retreat_waypoint(self, waypoint) -> None:
            assert waypoint == waypoints.retreat
            calls.append("retreat")

        def verify_grasp(self) -> bool:
            return True

        def verify_pose(self) -> bool:
            return True

    class Executor:
        def __init__(self, **kwargs) -> None:
            calls.append("executor")

        def execute_validated(self, actual_waypoints) -> None:
            assert actual_waypoints is waypoints
            calls.append("move")

    monkeypatch.setattr(local_cli, "SO101ArmAdapter", Adapter)
    monkeypatch.setattr(local_cli, "PickAndPlaceExecutor", Executor)

    local_cli._execute_connected_left_arm(plan, arm)

    assert calls == ["preflight", "adapter", "retreat", "executor", "move"]
