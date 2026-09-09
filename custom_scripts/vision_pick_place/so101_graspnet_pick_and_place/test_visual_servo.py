"""Tests for bounded XY visual-servo correction."""

from __future__ import annotations

import numpy as np
import pytest
import sys
import types

from custom_scripts.vision_pick_place.so101_graspnet_pick_and_place.visual_servo import (
    _load_local_qwen_detector,
    _build_execution_arm,
    _select_valid_depth_pixel,
    compute_xy_nudge,
    hold_tcp_pose,
    validate_args,
)


def test_compute_xy_nudge_transforms_target_minus_current() -> None:
    transform = np.eye(4)
    current = np.array([0.1, 0.2, 0.6])
    target = np.array([0.13, 0.16, 0.6])

    assert compute_xy_nudge(current, target, transform, max_nudge_m=0.05) == pytest.approx((0.03, -0.04))


def test_compute_xy_nudge_clamps_large_correction() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        compute_xy_nudge(np.zeros(3), np.array([0.1, 0.0, 0.0]), np.eye(4), max_nudge_m=0.04)


def test_load_local_qwen_detector_uses_local_module(monkeypatch) -> None:
    detector = lambda image, prompt: None
    module_name = "custom_scripts.vision_pick_place.task_red_cube_to_bin_new_gripper.perception_qwen"
    monkeypatch.setitem(sys.modules, module_name, types.SimpleNamespace(detect_qwen=detector))

    assert _load_local_qwen_detector() is detector


def test_select_valid_depth_pixel_stays_inside_detection_box() -> None:
    depth = np.zeros((20, 20), dtype=np.uint16)
    depth[8, 9] = 580
    assert _select_valid_depth_pixel(depth, (10, 10), (6, 6, 8, 8)) == (9, 8)
    assert _select_valid_depth_pixel(depth, (10, 10), (0, 0, 5, 5)) is None


def test_execution_arm_uses_left_arm_factory(monkeypatch) -> None:
    ports: list[str] = []

    class FakeArm:
        def __init__(self, *, port: str) -> None:
            ports.append(port)

    monkeypatch.setattr(
        "custom_scripts.vision_pick_place.so101_graspnet_pick_and_place.left_arm.LeftSOArm101", FakeArm
    )
    _build_execution_arm("/dev/ttyACM3")
    assert ports == ["/dev/ttyACM3"]


def test_hold_tcp_pose_enables_torque_at_the_current_pose() -> None:
    events: list[str] = []
    joints = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    class FakeArm:
        def connect(self) -> None:
            events.append("connect")

        def enable_torque(self) -> None:
            events.append("enable_torque")

        def get_joint_deg(self) -> np.ndarray:
            events.append("get_joint_deg")
            return joints

        def disconnect(self) -> None:
            events.append("disconnect")

    assert hold_tcp_pose("/dev/ttyACM3", arm_factory=lambda _port: FakeArm()) == pytest.approx(joints)
    assert events == ["connect", "enable_torque", "get_joint_deg", "disconnect"]


def test_single_step_allows_a_smaller_nudge_than_multistep_default() -> None:
    validate_args(multi_step=False, step_nudge_m=0.04, max_nudge_m=0.01)
    with pytest.raises(ValueError, match="step-nudge"):
        validate_args(multi_step=True, step_nudge_m=0.04, max_nudge_m=0.01)
