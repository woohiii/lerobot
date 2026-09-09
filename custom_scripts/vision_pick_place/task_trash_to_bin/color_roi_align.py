"""RGB 영상을 depth 카메라의 실제 시야각에 맞추는 독립 유틸리티."""

from __future__ import annotations

import cv2
import numpy as np


def crop_rgb_to_depth_fov(
    rgb_image: np.ndarray,
    roi_x: int,
    roi_y: int,
    roi_width: int,
    roi_height: int,
    depth_width: int,
    depth_height: int,
) -> np.ndarray:
    """RGB를 depth 실제 시야각 ROI로 자른 뒤 depth 해상도로 맞춘다.

    이 처리는 ros_astra_camera의 color_roi_x, color_roi_y,
    color_roi_width, color_roi_height 파라미터와 같은 개념이다.
    ROI 값은 카메라 캘리브레이션 결과를 호출자가 전달해야 한다.
    """
    rgb_roi = rgb_image[roi_y : roi_y + roi_height, roi_x : roi_x + roi_width]
    return cv2.resize(rgb_roi, (depth_width, depth_height))


if __name__ == "__main__":
    dummy_rgb = np.zeros((12, 16, 3), dtype=np.uint8)
    aligned_rgb = crop_rgb_to_depth_fov(
        dummy_rgb,
        roi_x=2,
        roi_y=3,
        roi_width=8,
        roi_height=6,
        depth_width=4,
        depth_height=3,
    )
    assert aligned_rgb.shape == (3, 4, 3)
