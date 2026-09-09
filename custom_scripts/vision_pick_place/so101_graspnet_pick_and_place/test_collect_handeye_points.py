"""Tests for markerless depth-pixel selection."""

from __future__ import annotations

import numpy as np
import pytest

from custom_scripts.vision_pick_place.so101_graspnet_pick_and_place.collect_handeye_points import (
    auto_select_depth_pixel,
    build_target_depth_pixels,
    find_nearest_valid_depth_pixel,
    resolve_depth_pixel,
    prepare_manual_collection_arm,
    is_tcp_at_table_height,
    is_sample_near_home,
    is_sample_within_anchor_envelope,
    is_vertical_grasp_pose,
    validate_joint_points,
)


def test_auto_select_depth_pixel_returns_nearest_valid_cluster_center() -> None:
    depth = np.full((10, 10), 800, dtype=np.uint16)
    depth[2:4, 6:8] = 400

    u, v = auto_select_depth_pixel(depth)

    assert (u, v) == (6, 2)


def test_auto_select_depth_pixel_rejects_empty_depth() -> None:
    with pytest.raises(RuntimeError, match="valid depth"):
        auto_select_depth_pixel(np.zeros((4, 4), dtype=np.uint16))


def test_find_nearest_valid_depth_pixel_expands_search_for_occlusion() -> None:
    depth = np.zeros((40, 40), dtype=np.uint16)
    depth[20, 30] = 500

    assert find_nearest_valid_depth_pixel(depth, 20, 20, radius=6, max_radius=16) == (30, 20)


def test_resolve_depth_pixel_rejects_occluded_tcp_by_default() -> None:
    depth = np.zeros((40, 40), dtype=np.uint16)
    depth[20, 30] = 500

    assert resolve_depth_pixel(depth, 20, 20) is None
    assert resolve_depth_pixel(depth, 20, 20, allow_fallback=True) == (30, 20)


def test_build_target_depth_pixels_supports_more_than_eight_points() -> None:
    points = build_target_depth_pixels(12)

    assert len(points) == 12
    assert all(0 <= u < 320 and 0 <= v < 240 for u, v in points)
    assert build_target_depth_pixels(12, enabled=False) == []


def test_validate_joint_points_requires_finite_matrix() -> None:
    joints = validate_joint_points(np.zeros((6, 5)), expected_count=6)

    assert joints.shape == (6, 5)
    with pytest.raises(ValueError, match="joint points"):
        validate_joint_points(np.zeros((5, 5)), expected_count=6)


def test_manual_collection_releases_torque_after_connect() -> None:
    class Arm:
        def __init__(self) -> None:
            self.released = False

        def release_torque(self) -> None:
            self.released = True

    arm = Arm()
    prepare_manual_collection_arm(arm)
    assert arm.released


def test_tcp_table_height_gate_rejects_points_above_tolerance() -> None:
    assert is_tcp_at_table_height(-0.0042, table_z=-0.0042, tolerance=0.015)
    assert not is_tcp_at_table_height(0.03, table_z=-0.0042, tolerance=0.015)


def test_sample_gate_keeps_wrist_fixed_and_limits_home_radius() -> None:
    home_xyz = (0.20, 0.0, -0.0042)
    home_joints = [0.0, 0.0, 0.0, 86.0, 0.0, 99.0]
    assert is_sample_near_home(
        (0.25, 0.05, -0.0042), [0.0, 0.0, 0.0, 89.0, 3.0, 99.0], home_xyz, home_joints,
        max_xy_radius=0.12, wrist_tolerance=5.0,
    )
    assert not is_sample_near_home(
        (0.25, 0.05, -0.0042), [0.0, 0.0, 0.0, 95.0, 3.0, 99.0], home_xyz, home_joints,
        max_xy_radius=0.12, wrist_tolerance=5.0,
    )
    assert not is_sample_near_home(
        (0.35, 0.05, -0.0042), [0.0, 0.0, 0.0, 89.0, 3.0, 99.0], home_xyz, home_joints,
        max_xy_radius=0.12, wrist_tolerance=5.0,
    )


def test_anchor_envelope_allows_inner_radius_and_blocks_farther_radius() -> None:
    anchor_xyz = (0.28, 0.0, -0.04)
    assert is_sample_within_anchor_envelope(
        (0.20, 0.05, -0.04), anchor_xyz, max_base_radius_m=0.285,
    )
    assert not is_sample_within_anchor_envelope(
        (0.30, 0.0, -0.04), anchor_xyz, max_base_radius_m=0.285,
    )


def test_vertical_grasp_gate_uses_tool_axis() -> None:
    vertical = np.eye(4)
    vertical[:3, 2] = (0.0, 0.0, -1.0)
    tilted = vertical.copy()
    tilted[:3, 2] = (0.8, 0.0, -0.6)
    assert is_vertical_grasp_pose(vertical, max_tilt_deg=20.0)
    assert not is_vertical_grasp_pose(tilted, max_tilt_deg=20.0)
