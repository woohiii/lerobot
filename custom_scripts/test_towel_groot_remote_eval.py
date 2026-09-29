"""Safety-gate unit tests for remote towel-policy evaluation."""

# ruff: noqa: D100, D103

from unittest.mock import Mock

import numpy as np
import pytest
import torch
from towel_groot_remote_eval import (
    TASK_BY_CONDITION,
    normalize_camera_aliases,
    send_action_safely,
    trigger_emergency_stop,
    validate_action_chunk,
    validate_latency,
    validate_observation,
    validate_relative_target,
)


def test_task_mapping_is_exact():
    assert TASK_BY_CONDITION["corner"].startswith("Unfold a towel with a folded corner")
    assert TASK_BY_CONDITION["edge"].startswith("Unfold a towel with one edge folded")
    assert TASK_BY_CONDITION["wrinkled"].startswith("Unfold a heavily wrinkled towel")


def test_orbbec_camera_aliases_match_training_feature_names():
    observation = {"astra": "rgb", "astra_depth": "depth"}

    assert normalize_camera_aliases(observation) == {
        "astra_rgb": "rgb",
        "astra_depth_viz": "depth",
    }


@pytest.mark.parametrize("actions", [torch.zeros((17, 12)), np.zeros((16, 11)), np.full((2, 12), np.nan)])
def test_action_gate_rejects_wrong_shape_or_nonfinite(actions):
    with pytest.raises(ValueError):
        validate_action_chunk(actions)


def test_action_gate_accepts_finite_12d_chunk():
    assert validate_action_chunk(torch.zeros((16, 12))).shape == (16, 12)


def test_observation_gate_requires_all_cameras_and_12d_state():
    obs = {f"joint_{i}.pos": float(i) for i in range(12)}
    obs.update(
        {
            key: np.zeros(shape, dtype=np.uint8)
            for key, shape in {
                "left_wrist": (480, 640, 3),
                "right_wrist": (480, 640, 3),
                "astra_rgb": (240, 320, 3),
                "astra_depth_viz": (240, 320, 3),
            }.items()
        }
    )
    validate_observation(obs)
    del obs["astra_rgb"]
    with pytest.raises(ValueError, match="astra_rgb"):
        validate_observation(obs)


def test_latency_gate_rejects_stale_and_future_timestamps():
    validate_latency(10.0, now=10.2, max_latency=0.5)
    with pytest.raises(ValueError, match="stale"):
        validate_latency(9.0, now=10.0, max_latency=0.5)
    with pytest.raises(ValueError, match="future"):
        validate_latency(10.5, now=10.0, max_latency=0.5)


def test_relative_target_gate_limits_motion():
    validate_relative_target(np.zeros(12), np.ones(12), max_delta=5)
    with pytest.raises(ValueError, match="relative target"):
        validate_relative_target(np.zeros(12), np.full(12, 6), max_delta=5)


def test_dry_run_never_calls_hardware_send():
    sent = []

    result = send_action_safely("dry-run", sent.append, {"joint.pos": 1.0})

    assert sent == []
    assert result == {"joint.pos": 1.0}


def test_emergency_stop_disables_both_buses_and_stops_client():
    client = Mock()

    trigger_emergency_stop(client)

    client.robot.left_arm.bus.disable_torque.assert_called_once_with()
    client.robot.right_arm.bus.disable_torque.assert_called_once_with()
    client.shutdown_event.set.assert_called_once_with()
