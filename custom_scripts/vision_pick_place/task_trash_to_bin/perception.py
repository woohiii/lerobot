"""Click-based coordinate estimation for an arbitrary (color/shape-agnostic)
trash object.

2026-09-08: trimmed copy of ../task_red_cube_to_bin_new_gripper/perception.py.
No HSV color/shape detection here - there is no generic detector for an
unknown-shaped trash object, so this task only supports a user-clicked
pixel (click_grasp_trash.py), not a wrist-cam closed-loop servo.

Coordinate estimation is NOT a full pixel+depth -> camera-frame 3D point ->
base-frame transform via a proper 6-DOF T_cam_to_base extrinsic - see the
sibling task's perception.py docstring for the full reasoning. What's used:
  - xy: a 2D homography (Astra RGB pixel -> robot-base xy on the table
    plane), from calibrate_camera.py's touch-point calibration
    (homography.json). Only valid for objects sitting on the table plane.
  - z: a HEIGHT DELTA at the clicked pixel (table depth reading minus the
    clicked patch's depth reading, both from Astra), added on top of the
    independently-measured TABLE_Z. No camera-to-base extrinsic needed.
This is a coarse guess only - task_state_machine.descend_and_grasp()'s
contact detection during descent, not this estimate's absolute accuracy, is
what actually decides when the gripper has reached the object.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

import cv2
import numpy as np

try:
    from . import config
except ImportError:
    import config


RGBD_PREVIEW_SCALE = 2


@dataclass
class Detection:
    cx: float  # pixel x of the object's centroid
    cy: float  # pixel y of the object's centroid
    area: float
    bbox: tuple[int, int, int, int]  # x, y, w, h


class PublishedFrameSource:
    """Reads whatever camera_hub.py/astra_s_live.py last published to `path`
    instead of opening the camera device itself (two processes can't both
    hold a UVC/OpenNI2 device open for streaming). Same cv2.VideoCapture-
    shaped isOpened()/read()/release() so it can substitute for one."""

    def __init__(self, path: str, stale_timeout_s: float = config.FRAME_STALE_TIMEOUT_S):
        self.path = path
        self.stale_timeout_s = stale_timeout_s

    def _fresh(self) -> bool:
        return os.path.exists(self.path) and (time.time() - os.path.getmtime(self.path)) < self.stale_timeout_s

    def isOpened(self) -> bool:
        return self._fresh()

    def read(self):
        if not self._fresh():
            return False, None
        frame = cv2.imread(self.path)
        return (frame is not None), frame

    def release(self) -> None:
        pass


def load_fresh_depth(depth_path: str = config.ASTRA_DEPTH_MM_PATH) -> np.ndarray | None:
    if not os.path.exists(depth_path) or (time.time() - os.path.getmtime(depth_path)) >= config.FRAME_STALE_TIMEOUT_S:
        return None
    try:
        depth_mm = np.load(depth_path)
    except (OSError, ValueError):
        return None
    return depth_mm if depth_mm.ndim == 2 else None


def rgbd_overlay(bgr: np.ndarray, depth_mm: np.ndarray | None) -> np.ndarray | None:
    if depth_mm is None or bgr.ndim != 3 or not np.any(depth_mm):
        return None
    depth_mm = _apply_registered_depth_offset(depth_mm)
    depth_vis = cv2.normalize(depth_mm, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)
    if bgr.shape[:2] != depth_color.shape[:2]:
        bgr = cv2.resize(bgr, (depth_color.shape[1], depth_color.shape[0]))
    return cv2.addWeighted(bgr, 0.55, depth_color, 0.45, 0)


def _apply_registered_depth_offset(depth_mm: np.ndarray) -> np.ndarray:
    offset_x, offset_y = config.ASTRA_DEPTH_REGISTERED_OFFSET_PX
    if offset_x == 0 and offset_y == 0:
        return depth_mm
    transform = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
    return cv2.warpAffine(
        depth_mm,
        transform,
        (depth_mm.shape[1], depth_mm.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def depth_valid_mask(depth_mm: np.ndarray | None, output_shape: tuple[int, ...]) -> np.ndarray | None:
    if depth_mm is None or depth_mm.ndim != 2:
        return None
    valid = _apply_registered_depth_offset(depth_mm) > 0
    # 구조광의 고립된 1~2픽셀 dropout을 메우되 큰 FOV 무효 영역은 보존한다.
    valid = cv2.morphologyEx(
        valid.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8)
    ).astype(bool)
    output_h, output_w = output_shape[:2]
    if valid.shape != (output_h, output_w):
        valid = cv2.resize(valid.astype(np.uint8), (output_w, output_h), interpolation=cv2.INTER_NEAREST).astype(bool)
    return valid


def _depth_registration_active() -> bool:
    status_path = config.ASTRA_DEPTH_REGISTRATION_STATUS_PATH
    if os.path.exists(status_path) and (time.time() - os.path.getmtime(status_path)) < config.FRAME_STALE_TIMEOUT_S:
        try:
            with open(status_path) as f:
                return bool(json.load(f)["registered"])
        except (OSError, ValueError, KeyError):
            pass
    return config.ASTRA_DEPTH_REGISTERED_TO_COLOR


def rgb_px_to_homography_px(px: float, py: float, frame_shape: tuple[int, ...]) -> tuple[float, float]:
    frame_h, frame_w = frame_shape[:2]
    return px * config.FRAME_W / frame_w, py * config.FRAME_H / frame_h


def is_frame_corrupted(bgr_frame: np.ndarray) -> bool:
    """Flags USB frame-tearing (a real, validated issue on the cheap wrist
    UVC camera specifically): (a) a noisy multicolor band - several rows in
    a row with an abnormally large jump from the row above, and (b) a solid
    anomalous color block - a tall run of near-identical rows (an all-
    zero/garbage USB transfer decodes to a flat, often greenish block) whose
    color sits far from the rest of the frame's average."""
    row_means = bgr_frame.mean(axis=1)
    diffs = np.abs(np.diff(row_means, axis=0)).sum(axis=1)
    noisy_band = int((diffs > 25).sum()) >= 4 or bool((diffs > 100).any())

    flat = diffs < 3
    max_run = run = best_start = 0
    for i, f in enumerate(flat):
        if f:
            run += 1
            if run > max_run:
                max_run, best_start = run, i - run + 1
        else:
            run = 0
    h = bgr_frame.shape[0]
    block_color_far = False
    if max_run >= h * 0.12:
        block_mean = row_means[best_start : best_start + max_run].mean(axis=0)
        block_color_far = float(np.abs(block_mean - row_means.mean(axis=0)).sum()) > 60
    return bool(noisy_band or block_color_far)


def _load_homography() -> np.ndarray | None:
    if not config.HOMOGRAPHY_PATH.exists():
        return None
    with open(config.HOMOGRAPHY_PATH) as f:
        data = json.load(f)
    return np.array(data["homography"], dtype=float)


_HOMOGRAPHY = _load_homography()


# 2026-09-09: 처음엔 homography.json의 touch-taught 점들의 bounding box +
# 마진으로 안전영역을 판정했으나, 이 방식은 (1) 클릭이 캘리브레이션에 쓴 좁은
# 화면 패치 밖으로 조금만 벗어나도 호모그래피 자체가 투영변환 특성상 폭주해
# 엉뚱한 xy를 내놓고, (2) taught box는 몇 개의 클릭 점에 좌우되는 임의값이라
# 실제 팔의 물리적 도달범위와 무관했다. 대신 probe_reach.py 실측으로 확인된
# "베이스로부터 반경 MAX_REACH_XY_M 안쪽은 안전, 밖은 IK가 못 미친다"는 원호
# 경계를 그대로 판정 기준으로 쓴다 - 팔 기구학과 직접 대응되는 값이라 캘리브
# 점 개수/분포에 흔들리지 않는다.
def is_xy_within_safe_workspace(x: float, y: float) -> bool:
    """False if (x, y)가 로봇 베이스(0,0) 기준 config.MAX_REACH_XY_M 반경
    밖이면 - 그 밖은 실측으로 IK가 도달 못 하는 것으로 확인된 영역."""
    return (x**2 + y**2) ** 0.5 <= config.MAX_REACH_XY_M


def _tested_arc_theta_range_deg() -> tuple[float, float]:
    """Angular sector (deg, atan2 from base) that CALIB_POINTS_XY actually
    probed - probe_reach.py confirmed every one of those targets reachable,
    so their angles bound the arc we trust. A true semicircle (+-90deg) is
    NOT physically real at this hover height: probe_reach_hover13.py and the
    earlier probe both showed the arm loses reach fast past ~+-27deg (z sags
    5-6cm, IK fails) - forcing a wider arc would just mark untested,
    likely-unreachable area as safe."""
    if not config.HOMOGRAPHY_PATH.exists():
        return (-30.0, 30.0)
    with open(config.HOMOGRAPHY_PATH) as f:
        pts = json.load(f).get("robot_points") or []
    if not pts:
        return (-30.0, 30.0)
    thetas = [float(np.degrees(np.arctan2(y, x))) for x, y in pts]
    return (min(thetas), max(thetas))


def reach_boundary_polygon_px(frame_shape: tuple[int, ...]) -> np.ndarray | None:
    """Pixel-space polygon (N, 2 int32) of the trusted safe-reach wedge: base
    origin (0, 0) + an arc at config.MAX_REACH_XY_M spanning the empirically
    tested angular sector, mapped through the homography's inverse. Only
    near-taught points get inverse-mapped, so - unlike testing every pixel's
    forward homography, or centering/widening the arc past the taught patch
    (both blow up projectively, see 2026-09-09 history) - this stays a clean
    arc. cv2.fillPoly the result for a safe-region mask. None if no
    homography yet."""
    if _HOMOGRAPHY is None:
        return None
    h, w = frame_shape[:2]
    h_inv = np.linalg.inv(_HOMOGRAPHY)
    theta_lo, theta_hi = _tested_arc_theta_range_deg()
    thetas = np.radians(np.linspace(theta_lo, theta_hi, 24))
    r = config.MAX_REACH_XY_M
    robot_pts = [(0.0, 0.0)] + [(r * np.cos(t), r * np.sin(t)) for t in thetas]
    px_pts = []
    for x, y in robot_pts:
        mapped = h_inv @ np.array([x, y, 1.0])
        if abs(mapped[2]) < 1e-9:
            continue
        hx, hy = mapped[0] / mapped[2], mapped[1] / mapped[2]
        px_pts.append((hx * w / config.FRAME_W, hy * h / config.FRAME_H))
    return np.array(px_pts, dtype=np.int32) if len(px_pts) >= 3 else None


def pixel_to_xy(px: float, py: float, homography: np.ndarray | None = _HOMOGRAPHY) -> tuple[float, float] | None:
    """Coarse robot-base (x, y) for an arbitrary Astra-RGB pixel via the
    table-plane homography - see this module's docstring for why this isn't
    a full 3D backprojection. None if there's no homography yet."""
    if homography is None:
        return None
    mapped = homography @ np.array([px, py, 1.0])
    if abs(mapped[2]) < 1e-9:
        return None
    return float(mapped[0] / mapped[2]), float(mapped[1] / mapped[2])


def estimate_height_at_px_m(
    px: float,
    py: float,
    rgb_path: str = config.ASTRA_RGB_FRAME_PATH,
    depth_path: str = config.ASTRA_DEPTH_MM_PATH,
) -> float | None:
    """Astra-depth height DELTA at an RGB click, independent of object color/shape.

    The clicked RGB-pixel patch is mapped into Astra depth's native resolution;
    the whole-frame table median is subtracted from the patch median and
    implausible readings return None.
    """
    ret, color = PublishedFrameSource(rgb_path).read()
    if not ret or color is None:
        return None
    if not os.path.exists(depth_path) or (time.time() - os.path.getmtime(depth_path)) >= config.FRAME_STALE_TIMEOUT_S:
        return None
    try:
        depth_mm = np.load(depth_path)
    except (OSError, ValueError):
        return None

    if depth_mm.ndim != 2 or color.ndim < 2:
        return None

    depth_h, depth_w = depth_mm.shape
    color_h, color_w = color.shape[:2]
    if _depth_registration_active():
        # OpenNI2 DEPTH_TO_COLOR registration puts both frames in color-camera
        # coordinates; only the published resolutions still differ.
        offset_x, offset_y = config.ASTRA_DEPTH_REGISTERED_OFFSET_PX
        center_x = px * depth_w / color_w - offset_x
        center_y = py * depth_h / color_h - offset_y
    else:
        # Astra S nominal FOV: color 62.0 x 48.6 deg, depth 58.4 x 45.5 deg.
        # Map about each optical center using tan(FOV/2), not a naive aspect
        # resize, when a driver cannot expose OpenNI2 image registration.
        x_scale = np.tan(np.deg2rad(62.0 / 2)) / np.tan(np.deg2rad(58.4 / 2))
        y_scale = np.tan(np.deg2rad(48.6 / 2)) / np.tan(np.deg2rad(45.5 / 2))
        center_x = depth_w * (0.5 + (px / color_w - 0.5) * x_scale)
        center_y = depth_h * (0.5 + (py / color_h - 0.5) * y_scale)

    radius = config.HEIGHT_SAMPLE_RADIUS_PX
    radius_x = radius * depth_w / color_w
    radius_y = radius * depth_h / color_h
    x0, y0 = max(0, int(np.floor(center_x - radius_x))), max(0, int(np.floor(center_y - radius_y)))
    x1, y1 = min(depth_w, int(np.ceil(center_x + radius_x))), min(depth_h, int(np.ceil(center_y + radius_y)))
    if x1 <= x0 or y1 <= y0:
        return None

    patch_valid = depth_mm[y0:y1, x0:x1]
    patch_valid = patch_valid[patch_valid > 0]
    table_valid = depth_mm[depth_mm > 0]
    if patch_valid.size < config.HEIGHT_SAMPLE_MIN_VALID_PIXELS or table_valid.size < 100:
        return None

    height_m = (float(np.median(table_valid)) - float(np.median(patch_valid))) / 1000.0
    if not (config.OBJECT_HEIGHT_MIN_M <= height_m <= config.OBJECT_HEIGHT_MAX_M):
        return None
    return height_m
