"""Object detection and 3D-ish coordinate estimation.

Per the spec's §4: HSV color segmentation is the primary method (no GPU/
training data needed, low latency, very stable for a single solid-color
object) - the interface (`Detection`, `detect_red_cube`, `detect_black_bin`,
each taking a raw BGR frame and returning a `Detection | None`) is deliberately
narrow so a YOLO-based implementation could be swapped in later without
touching any caller, if HSV's real failure mode (lighting sensitivity, or a
same-colored object in frame) ever proves worse in practice than it has been
today. It was tried today (YOLO-World, zero-shot, no training data available)
and performed worse than this HSV+shape approach on the actual objects in
this workspace - see this task's design notes.

Coordinate estimation is NOT a full pixel+depth -> camera-frame 3D point ->
base-frame transform via a proper 6-DOF T_cam_to_base extrinsic - that would
need a real hand-eye calibration (ArUco/checkerboard + cv2.calibrateHandEye),
which doesn't exist yet. What's used instead, and what's actually validated
against real hardware:
  - xy: a 2D homography (Astra RGB pixel -> robot-base xy on the table
    plane), from calibrate_camera.py's touch-point calibration
    (homography.json). Only valid for objects sitting on the table plane -
    which is exactly this task.
  - z: not from Astra depth's absolute reading at all (no camera-to-base
    transform to turn that into a base-frame coordinate) - instead a HEIGHT
    DELTA (table depth reading minus cube depth reading, both from Astra),
    which needs no extrinsics to be useful, added on top of the
    independently-measured TABLE_Z.
This is a coarse guess only - FINE_SERVO (see task_state_machine.py) is what
actually gets the gripper precisely onto the object, using the wrist camera
in closed loop, not trusting this estimate's absolute accuracy.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

import cv2
import numpy as np

import config


@dataclass
class Detection:
    cx: float  # pixel x of the object's centroid
    cy: float  # pixel y of the object's centroid
    area: float
    bbox: tuple[int, int, int, int]  # x, y, w, h


class PublishedFrameSource:
    """Reads whatever camera_hub.py/astra_s_live.py last published to `path`
    instead of opening the camera device itself (see this module's docstring
    on why). Same cv2.VideoCapture-shaped isOpened()/read()/release() so it
    can substitute for one."""

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


def _best_candidate(contours, min_area, max_area, min_solidity, aspect_range):
    """Largest contour passing an area range + shape filter (solidity,
    aspect ratio) - not just the largest same-colored blob, so a hand/arm/
    cable in frame can't out-vote the real object. See config.py's
    MIN_CUBE_SOLIDITY comment for how these thresholds were tuned against a
    real miss."""
    best, best_area = None, 0.0
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        hull_area = cv2.contourArea(cv2.convexHull(c))
        if hull_area <= 0 or (area / hull_area) < min_solidity:
            continue
        _, _, w, h = cv2.boundingRect(c)
        if h == 0 or not (aspect_range[0] <= (w / h) <= aspect_range[1]):
            continue
        if area > best_area:
            best, best_area = c, area
    return best, best_area


def _detect(bgr_frame, lower_ranges_upper, min_area, max_area_frac, min_solidity, aspect_range, kernel_size):
    hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
    mask = None
    for lower, upper in lower_ranges_upper:
        m = cv2.inRange(hsv, lower, upper)
        mask = m if mask is None else cv2.bitwise_or(mask, m)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c, area = _best_candidate(contours, min_area, config.FRAME_AREA_HINT * max_area_frac, min_solidity, aspect_range)
    if c is None:
        return None
    m = cv2.moments(c)
    if m["m00"] == 0:
        return None
    bbox = cv2.boundingRect(c)
    return Detection(cx=m["m10"] / m["m00"], cy=m["m01"] / m["m00"], area=area, bbox=bbox)


# 2026-09-01: added for qwen_click_pick_place.py's closed-loop fix - a real
# llm_pick_place.py run grasped nothing (open-loop homography+depth-delta
# alone wasn't accurate enough - see that file's own notes). detect_red_cube/
# detect_black_bin already prove HSV blob detection is fast enough for
# wrist-cam closed-loop servoing (task_state_machine.fine_servo calls detect_fn
# every frame) - the problem was only ever having a FIXED range for just two
# known objects. These two functions bootstrap the same fast _detect() machinery
# for an ARBITRARY object: sample its color once from wherever Qwen found it
# (perception_qwen.detect_all_qwen, one call, not per-frame), then reuse that
# sampled range as a real per-frame wrist-cam detect_fn - no per-frame LLM
# calls (perception_qwen's own module docstring already flagged that as too
# slow for a servo loop).
def sample_hsv_ranges(
    bgr_frame: np.ndarray, bbox: tuple[int, int, int, int], hue_tol: int = 12, sat_tol: int = 70, val_tol: int = 70
) -> list[tuple[tuple[int, int, int], tuple[int, int, int]]]:
    """Median HSV inside bbox +/- tolerance, as one or two (lower, upper)
    ranges ready for _detect()/build_color_detect_fn. Two ranges (same split
    as config.LOWER_RED_1/2) if the sampled hue sits near the 0/180 wrap seam -
    a real red object can otherwise get a range that misses its own other
    half."""
    x, y, w, h = bbox
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(bgr_frame.shape[1], x + w), min(bgr_frame.shape[0], y + h)
    hsv = cv2.cvtColor(bgr_frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    hue, sat, val = (float(np.median(hsv[..., i])) for i in range(3))
    s_lo, s_hi = max(0, sat - sat_tol), min(255, sat + sat_tol)
    v_lo, v_hi = max(0, val - val_tol), min(255, val + val_tol)
    if hue - hue_tol < 0 or hue + hue_tol > 179:
        # wraps around the hue circle - split into a low-end and high-end range
        return [
            ((0, s_lo, v_lo), (min(179, (hue + hue_tol) % 180), s_hi, v_hi)),
            ((max(0, (hue - hue_tol) % 180), s_lo, v_lo), (179, s_hi, v_hi)),
        ]
    return [((hue - hue_tol, s_lo, v_lo), (hue + hue_tol, s_hi, v_hi))]


def build_color_detect_fn(
    hsv_ranges: list[tuple[tuple[int, int, int], tuple[int, int, int]]],
    min_area: float = 80.0, max_area_frac: float = 0.6, min_solidity: float = 0.5,
    aspect_range: tuple[float, float] = (0.2, 5.0), kernel_size: int = 5,
):
    """Returns a detect_fn(bgr_frame) -> Detection|None over hsv_ranges, same
    Detection shape as detect_red_cube/detect_black_bin - drop-in for
    task_state_machine.fine_servo. Looser shape-filter defaults than the
    cube/bin's own (min_solidity/aspect_range) since this runs against
    whatever real object shape Qwen happened to find, not specifically a cube."""

    def detect_fn(bgr_frame: np.ndarray) -> Detection | None:
        return _detect(bgr_frame, hsv_ranges, min_area, max_area_frac, min_solidity, aspect_range, kernel_size)

    return detect_fn


def detect_red_cube(bgr_frame: np.ndarray) -> Detection | None:
    return _detect(
        bgr_frame,
        [(config.LOWER_RED_1, config.UPPER_RED_1), (config.LOWER_RED_2, config.UPPER_RED_2)],
        config.MIN_CUBE_CONTOUR_AREA, config.MAX_CUBE_AREA_FRAC, config.MIN_CUBE_SOLIDITY, config.CUBE_ASPECT_RANGE,
        kernel_size=5,
    )


def detect_black_bin(bgr_frame: np.ndarray) -> Detection | None:
    return _detect(
        bgr_frame,
        [(config.LOWER_BLACK, config.UPPER_BLACK)],
        config.MIN_BIN_CONTOUR_AREA, config.MAX_BIN_AREA_FRAC, config.MIN_BIN_SOLIDITY, config.BIN_ASPECT_RANGE,
        kernel_size=7,
    )


def _load_homography() -> np.ndarray | None:
    if not config.HOMOGRAPHY_PATH.exists():
        return None
    with open(config.HOMOGRAPHY_PATH) as f:
        data = json.load(f)
    return np.array(data["homography"], dtype=float)


_HOMOGRAPHY = _load_homography()


def _load_calib_robot_bounds() -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Bounding box (x_range, y_range) of homography.json's own touch-taught
    robot_points - see is_xy_within_safe_workspace's docstring for why this
    exists. None if no homography/calib file yet."""
    if not config.HOMOGRAPHY_PATH.exists():
        return None
    with open(config.HOMOGRAPHY_PATH) as f:
        pts = json.load(f).get("robot_points")
    if not pts:
        return None
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return (min(xs), max(xs)), (min(ys), max(ys))


_CALIB_ROBOT_BOUNDS = _load_calib_robot_bounds()

# 2026-09-01: added alongside llm_pick_place.py - homography.json only has 4
# touch-taught points (see its own contents), so pixel_to_xy's linear map is
# ALREADY extrapolating for anything outside that small patch (e.g. a real
# object used in that session's own Qwen testing sat partly outside the
# taught pixel range). A wrong/hallucinated LLM detection extrapolated the
# same way could send the arm toward an uncalibrated, possibly joint-limit-
# straining part of the workspace with no other guard catching it before
# motion starts - move_to_xyz's own MAX_MOVE_DELTA_DEG cap only limits how
# FAR a single move goes from the arm's CURRENT pose, not whether the target
# itself is somewhere ever verified safe. Margin is generous (not the exact
# taught box) specifically so real objects near-but-outside the 4 taught
# points still get through - this is a sanity gate against a badly wrong
# result, not a tight reachability fence.
WORKSPACE_MARGIN_M = 0.08


def is_xy_within_safe_workspace(x: float, y: float) -> bool:
    """False if (x, y) falls outside the calibrated touch-points' bounding
    box + WORKSPACE_MARGIN_M - callers should refuse to move there rather
    than trust a detection that extrapolated this far past anything ever
    physically verified. True (no opinion) if no calibration exists yet,
    same permissive default pixel_to_xy itself falls back to."""
    if _CALIB_ROBOT_BOUNDS is None:
        return True
    (x_lo, x_hi), (y_lo, y_hi) = _CALIB_ROBOT_BOUNDS
    return (
        x_lo - WORKSPACE_MARGIN_M <= x <= x_hi + WORKSPACE_MARGIN_M
        and y_lo - WORKSPACE_MARGIN_M <= y <= y_hi + WORKSPACE_MARGIN_M
    )


def pixel_to_xy(px: float, py: float, homography: np.ndarray | None = _HOMOGRAPHY) -> tuple[float, float] | None:
    """Coarse robot-base (x, y) for an arbitrary Astra-RGB pixel via the
    table-plane homography - see this module's docstring for why this isn't
    a full 3D backprojection. None if there's no homography yet. Factored
    out of estimate_xy_from_astra so a user-clicked pixel (click_pick_place.py)
    can reuse the same transform as an auto-detected one."""
    if homography is None:
        return None
    mapped = homography @ np.array([px, py, 1.0])
    if abs(mapped[2]) < 1e-9:
        return None
    return float(mapped[0] / mapped[2]), float(mapped[1] / mapped[2])


def estimate_xy_from_astra(
    detect_fn, rgb_path: str = config.ASTRA_RGB_FRAME_PATH, homography: np.ndarray | None = _HOMOGRAPHY
) -> tuple[float, float] | None:
    """Coarse robot-base (x, y) for whatever detect_fn finds in the Astra
    RGB view - None if Astra isn't running/in view, or nothing detected."""
    ret, color = PublishedFrameSource(rgb_path).read()
    if not ret or color is None:
        return None
    det = detect_fn(color)
    if det is None:
        return None
    return pixel_to_xy(det.cx, det.cy, homography)


def estimate_cube_height_m(
    rgb_path: str = config.ASTRA_RGB_FRAME_PATH, depth_path: str = config.ASTRA_DEPTH_MM_PATH
) -> float | None:
    """Astra-depth height DELTA for the detected cube above the table -
    table depth (median over the whole frame - the table dominates the
    view) minus cube depth (median within the detection's bbox, scaled into
    depth's lower native resolution). No camera-to-base extrinsic needed for
    a delta - see this module's docstring. None if unavailable/implausible;
    every caller must have a TABLE_Z fallback for exactly that reason."""
    ret, color = PublishedFrameSource(rgb_path).read()
    if not ret or color is None:
        return None
    det = detect_red_cube(color)
    if det is None:
        return None
    if not os.path.exists(depth_path) or (time.time() - os.path.getmtime(depth_path)) >= config.FRAME_STALE_TIMEOUT_S:
        return None
    try:
        depth_mm = np.load(depth_path)
    except (OSError, ValueError):
        return None

    sx, sy = depth_mm.shape[1] / color.shape[1], depth_mm.shape[0] / color.shape[0]
    bx, by, bw, bh = det.bbox
    x0, y0 = max(0, int(bx * sx)), max(0, int(by * sy))
    x1, y1 = min(depth_mm.shape[1], int((bx + bw) * sx)), min(depth_mm.shape[0], int((by + bh) * sy))
    if x1 <= x0 or y1 <= y0:
        return None

    cube_valid = depth_mm[y0:y1, x0:x1]
    cube_valid = cube_valid[cube_valid > 0]
    table_valid = depth_mm[depth_mm > 0]
    if cube_valid.size < 5 or table_valid.size < 100:
        return None

    height_m = (float(np.median(table_valid)) - float(np.median(cube_valid))) / 1000.0
    if not (config.CUBE_HEIGHT_MIN_M <= height_m <= config.CUBE_HEIGHT_MAX_M):
        return None
    return height_m
