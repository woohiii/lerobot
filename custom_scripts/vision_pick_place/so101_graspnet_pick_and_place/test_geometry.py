import numpy as np
import pytest

from .geometry import build_roi_point_cloud, transform_points
from .models import CameraIntrinsics, ObjectRoi, RgbdFrame


def _frame() -> RgbdFrame:
    rgb = np.array(
        [
            [[10, 11, 12], [20, 21, 22], [30, 31, 32]],
            [[40, 41, 42], [50, 51, 52], [60, 61, 62]],
        ],
        dtype=np.uint8,
    )
    depth = np.array([[1000, 0, 3000], [2000, 4000, 5000]], dtype=np.uint16)
    return RgbdFrame(rgb=rgb, depth=depth, intrinsics=CameraIntrinsics(100, 100, 1, 0))


def test_build_roi_point_cloud_backprojects_depth_in_meters_and_keeps_rgb():
    points = build_roi_point_cloud(_frame(), ObjectRoi(0, 0, 3, 2))

    assert points.shape == (5, 6)
    np.testing.assert_allclose(points[0, :3], [-0.01, 0.0, 1.0])
    np.testing.assert_array_equal(points[0, 3:], [10, 11, 12])
    np.testing.assert_allclose(points[-1, :3], [0.05, 0.05, 5.0])


def test_build_roi_point_cloud_filters_optional_z_range():
    points = build_roi_point_cloud(_frame(), ObjectRoi(0, 0, 3, 2), min_z=1.5, max_z=4.0)

    np.testing.assert_allclose(points[:, 2], [3.0, 2.0, 4.0])


@pytest.mark.parametrize("roi", [ObjectRoi(-1, 0, 2, 2), ObjectRoi(0, 0, 4, 2), ObjectRoi(2, 1, 1, 2)])
def test_build_roi_point_cloud_rejects_out_of_bounds_or_empty_roi(roi):
    with pytest.raises(ValueError):
        build_roi_point_cloud(_frame(), roi)


def test_transform_points_applies_homogeneous_transform():
    points = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 2.0]])
    transform = np.eye(4)
    transform[:3, 3] = [10.0, 20.0, 30.0]

    np.testing.assert_allclose(transform_points(points, transform), points + [10, 20, 30])


def test_transform_points_rejects_wrong_shapes():
    with pytest.raises(ValueError):
        transform_points(np.ones((3, 2)), np.eye(4))
    with pytest.raises(ValueError):
        transform_points(np.ones((3, 3)), np.eye(3))


def test_build_roi_point_cloud_uses_sdk_xyz_without_intrinsics():
    xyz = np.zeros((2, 2, 3), dtype=np.float64)
    xyz[:] = [0.12, -0.04, 0.50]
    frame = RgbdFrame(
        rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        depth=np.full((2, 2), 500, dtype=np.uint16),
        intrinsics=None,
        xyz_camera=xyz,
    )
    points = build_roi_point_cloud(frame, ObjectRoi(0, 0, 2, 2))
    np.testing.assert_allclose(points[:, :3], np.tile([0.12, -0.04, 0.50], (4, 1)))
