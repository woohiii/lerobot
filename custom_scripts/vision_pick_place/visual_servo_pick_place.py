"""Real-time visual-servoing pick-and-place - camera only, no touch-point
calibration and no homography (replaces calibrate_camera.py's approach).

Why this works without ever knowing a pixel<->robot-frame mapping: the RGB
camera used here is the WRIST camera, mounted on the gripper itself. Moving
the arm visibly shifts a stationary target's position in that camera's frame,
so "the target's pixel offset from image-center" IS the servo error - driving
it to zero centers the gripper over the target directly in pixel space. No
robot-frame coordinates for the cube or bin are ever computed.

The one thing that genuinely differs between arms/mounts is *which way* an
(x, y) nudge moves the image (mount angle, camera orientation). Rather than
hardcode that, each servo phase starts with a tiny 3-probe self-test (nudge
+x, nudge +y, watch the pixel shift each time) to build a local 2x2 image
Jacobian by finite differences, then inverts it to convert future pixel error
directly into the right xy correction. This is the whole "calibration step"
this script needs, and it's automatic, target-only, and takes under a second.

Sequence: search -> center on red cube -> descend -> grasp -> lift -> search
-> center on black bin -> release -> retreat home.

Safety: every underlying move goes through robot_control.py's existing joint-
limit clamps, per-call delta caps, and stall/collision detection (retreats
and raises CollisionDetected if the arm stops making progress mid-move - see
robot_control.py's docstring). This script adds bounded search sweeps and
iteration caps on top so nothing here can search or servo forever.

2026-08-26: the wrist cam still does all the xy centering (eye-in-hand pixel
error, as above) - the Astra S (a separate, external, fixed-mount camera) has
no role in xy at all, since that would need a proper camera-to-robot-base
extrinsic/homography calibration that doesn't exist yet (calibrate_camera.py
has the machinery for this but was never actually completed - see that
file's docstring). What Astra S *can* contribute without any such
calibration: its depth channel gives a camera-relative height for the cube
(distance from Astra to the cube's top vs. distance from Astra to the bare
table - a delta, not an absolute robot-frame coordinate, so no extrinsics
needed) - see estimate_cube_height_m() and its use in descend_and_grasp.
This replaces blindly descending to a fixed TABLE_Z guess (which is why a
real cube shorter than assumed, or the arm just missing it in xy, both ended
up reading as "reached bottom with no contact") with an actual per-attempt
height estimate, while keeping the existing collision-detect stall-check as
the safety backstop regardless of whether the depth estimate is trustworthy.
"""

from __future__ import annotations

import json
import os
import time

import cv2
import numpy as np

from camera_utils import ASTRA_DEPTH_MM_PATH, ASTRA_RGB_FRAME_PATH, PublishedDepthSource, PublishedFrameSource, WRIST_FRAME_PATH
from cube_detector import Detection, detect_black_bin, detect_red_cube, is_frame_corrupted
from robot_control import CollisionDetected, RobotController

FRAME_W, FRAME_H = 640, 480
IMG_CENTER = np.array([FRAME_W / 2, FRAME_H / 2])

# 2026-08-26: calibrate_camera.py's touch-point homography (Astra RGB pixel
# -> robot-frame xy on the table plane) - loaded once at import if it exists.
# Used by search_for_target as a coarse first guess (skip/narrow the blind
# SEARCH_OFFSETS grid) before handing off to the wrist cam for fine
# centering, same division of labor as estimate_cube_height_m() for z: Astra
# gives a rough absolute-frame number, the wrist cam does the actual precise
# work. None (not an error) if calibrate_camera.py hasn't been run yet -
# every caller already has the blind-grid fallback for that.
HOMOGRAPHY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "homography.json")


def _load_homography() -> np.ndarray | None:
    if not os.path.exists(HOMOGRAPHY_PATH):
        return None
    with open(HOMOGRAPHY_PATH) as f:
        data = json.load(f)
    return np.array(data["homography"], dtype=float)


_HOMOGRAPHY = _load_homography()


def estimate_xy_from_astra(
    detect_fn, rgb_path: str = ASTRA_RGB_FRAME_PATH, homography: np.ndarray | None = _HOMOGRAPHY
) -> tuple[float, float] | None:
    """Coarse real-world (x, y) estimate of whatever detect_fn finds in the
    Astra RGB view, via the touch-point homography. Returns None (never
    raises) if there's no homography file, Astra isn't running/in view, or
    nothing was detected.

    rgb_path/homography default to the real published path and the loaded
    calibration but are overridable - sim_dry_run.py's unit test passes a
    known synthetic homography and an isolated temp frame instead, same
    reasoning as estimate_cube_height_m's rgb_path/depth_path."""
    if homography is None:
        return None
    astra_cap = PublishedFrameSource(rgb_path)
    ret, color = astra_cap.read()
    if not ret or color is None:
        return None
    det = detect_fn(color)
    if det is None:
        return None
    pt = np.array([det.cx, det.cy, 1.0])
    mapped = homography @ pt
    if abs(mapped[2]) < 1e-9:
        return None
    return float(mapped[0] / mapped[2]), float(mapped[1] / mapped[2])

# 2026-08-26: visually inspected saved wrist-cam frames (the gripper's own
# jaw tips are visible in-shot, since this is an eye-in-hand mount) after a
# run converged to <17px error at IMG_CENTER, descended cleanly to TABLE_Z,
# and still closed on nothing - three separate times. In every saved frame
# the jaw tips sit off IMG_CENTER (320, 240), toward the bottom of the frame
# - the wrist camera doesn't look straight down the gripper's own grasp
# axis, so "cube centered in the image" was never the same thing as "cube
# between the jaws". Driving the servo error to zero against THIS point
# instead should be the fix.
#
# The first reading (270, 300) was taken from a badly overexposed frame
# (background blown to solid white - see camera_hub.py's WRIST_V4L2_CTRLS
# fix, applied the same day) where the jaw edges were hard to place
# precisely. After that exposure fix, a clean, high-contrast frame at the
# same idle pose read the jaw gap noticeably further left, around (195,
# 275) - used here instead. Re-measure again from a fresh saved frame if
# grasps still miss; this is an empirical reading, not a proper calibration.
#
# This is a fixed mechanical camera-to-jaw offset (eye-in-hand, rigidly
# mounted) so it should hold across arm poses - the one caveat is
# wrist_roll: solve_ik is free to pick any wrist_roll for a given xyz (see
# kinematics_helper.py), so if it ever lands on a very different roll than
# the runs this was measured from, the jaws (and this offset) would
# visually rotate around IMG_CENTER instead of staying put.
GRASP_TARGET_PX = np.array([195.0, 275.0])

# Reachable, previously-validated hover pose (center of the rectangle used in
# the earlier touch-point calibration attempt, raised well above the table for
# a wide search view and clearance). Used to start searching, not to end on.
HOME_XYZ = (0.23, 0.0, 0.13)

# 2026-08-26: where the user physically drove the arm back to and confirmed
# as "초기위치" (the resting/idle position) after today's TABLE_Z incident -
# read back directly from the live robot, not derived/guessed. Used only for
# the return-to-rest moves at the end of a run (success or abort), not as the
# search-start pose (HOME_XYZ, above, stays a raised hover position - this is
# a low folded-in resting pose, not a good vantage point to search from).
RETURN_HOME_XYZ = (0.10259099, 0.00435801, -0.02739574)
# 2026-08-26: this was 0.045 all session, sourced from "an earlier IK test"
# that apparently only checked reachability, never actual table contact.
# Root cause of nearly every "reached target height, zero contact" failure
# today, even with excellent (<1cm) xy centering: TABLE_Z was ~45mm ABOVE
# the real table. Found by physically running the gripper down onto the
# table with the user watching and reading back the xyz right at the moment
# of real contact: (0.122, -0.0003, -0.0003) - z was essentially 0.000, not
# 0.045. Set with a small 3mm margin above that literal contact point (the
# collision-detect stall check is still the real safety net for anything
# actually in the way - this is a floor for the no-contact-yet blind
# portion of a descend, not a promise nothing will be touched before it).
TABLE_Z = 0.003

PIXEL_TOLERANCE = 22.0  # px - "centered enough" - widened slightly: real runs show a
# few px of jitter even standing still (detection noise / minor settling), and 18
# was tight enough that a run which reached 12.9px error still needed one more
# lucky frame it didn't get. Now a secondary sanity bound, not the primary
# gate - see PHYSICAL_TOLERANCE_M in servo_to_target for why.
PHYSICAL_TOLERANCE_M = 0.004  # 4mm - the real convergence gate (see servo_to_target's
# comment on why pixel error alone isn't enough once the image Jacobian is anisotropic).
# A real cube here is a few cm across; 4mm gives real margin without demanding
# unrealistic precision from a self-calibrated, noisy local linearization.
CENTER_STABLE_FRAMES = 3  # was 5 - still robust against one noisy reading, less
# exposed to Jacobian staleness (see the re-estimation comment below) eating the
# stability streak over a longer window.
# 2026-08-26: raised from 50 alongside the MAX_STEP_M cut - a smaller, stable
# step naturally takes more iterations to close the same distance, and the
# new PHYSICAL_TOLERANCE_M gate is a genuinely tighter target than the old
# pixel-only one. sim_dry_run.py's cases settle around iteration 40-55 with
# the smaller step; 90 leaves real margin without letting a truly stuck
# run spin forever (each iteration is a real robot move + camera read, not
# free, so this still isn't "large").
MAX_SERVO_ITERS = 90
PROBE_DELTA_M = 0.008  # size of the two self-calibration probe nudges
SERVO_GAIN = 0.5  # <1: apply half the Jacobian-computed correction per step, damps overshoot
# 2026-08-26: lowered from 0.02 (20mm) after adding PHYSICAL_TOLERANCE_M exposed a real
# instability - a 20mm step taken mostly along the image Jacobian's HIGH-sensitivity
# direction (confirmed anisotropic on real hardware, ~6.6x between axes) swings the
# detected pixel position by tens of px, overshooting past zero-error and back every
# iteration instead of settling. The servo kept oscillating around 10-25px error and
# would never reach 4mm real distance, no matter how many iterations it got. 6mm keeps
# even the most sensitive direction's single-step pixel swing well inside a settleable
# range; confirmed via sim_dry_run.py that this converges cleanly to <3mm where 20mm
# didn't converge at all within MAX_SERVO_ITERS.
MAX_STEP_M = 0.006  # per-iteration nudge cap, independent of robot_control's own caps
BROYDEN_MIN_STEP_M = 0.0025  # below this, skip the Broyden update - see its comment in servo_to_target

LIFT_M = 0.08
BIN_DESCEND_M = 0.05  # descend this far after centering over the bin, before releasing

# Search sweep offsets from HOME_XYZ (meters), tried in order until the target
# is seen. Small plus/diagonal grid - bounded and safe, not a blind spiral.
SEARCH_OFFSETS = [
    (0.0, 0.0),
    (0.03, 0.0), (0.06, 0.0), (-0.03, 0.0), (-0.06, 0.0),
    (0.0, 0.04), (0.0, 0.08), (0.0, -0.04), (0.0, -0.08),
    (0.03, 0.04), (-0.03, 0.04), (0.03, -0.04), (-0.03, -0.04),
]


def get_pixel(cap: cv2.VideoCapture, detect_fn, tries: int = 8) -> tuple[Detection | None, np.ndarray | None]:
    """Reads a few frames (camera may be mid-transfer, or the last-published
    file may have been a torn USB frame - camera_hub.py already filters those
    before publishing, but this checks again in case an older hub is running)
    and returns the first detection found, or (None, None) if the target
    isn't visible right now."""
    for _ in range(tries):
        ret, frame = cap.read()
        if not ret or frame is None or is_frame_corrupted(frame):
            time.sleep(0.03)
            continue
        det = detect_fn(frame)
        if det is not None:
            return det, frame
        time.sleep(0.03)
    return None, None


def search_for_target(rc: RobotController, cap: cv2.VideoCapture, detect_fn, name: str) -> bool:
    print(f"\n[탐색] {name}을(를) 찾는 중...")

    # Astra-homography coarse guess first, before falling back to the blind
    # grid below - see estimate_xy_from_astra()'s comment. Bad/stale
    # homography or a wildly wrong estimate isn't fatal here: move_to_xyz's
    # own safety caps/CollisionDetected still apply, and either way this
    # just falls through to the grid search exactly as if Astra weren't
    # there at all.
    astra_xy = estimate_xy_from_astra(detect_fn)
    if astra_xy is not None:
        target = (astra_xy[0], astra_xy[1], HOME_XYZ[2])
        print(f"   [Astra] 대략적 위치 추정: ({astra_xy[0]:.3f}, {astra_xy[1]:.3f}) - 바로 이동합니다.")
        try:
            rc.move_to_xyz_converge(target, tolerance_m=0.015, max_iters=10)
            time.sleep(0.2)
            det, _ = get_pixel(cap, detect_fn, tries=10)
            if det is not None:
                print(f"   [Astra 추정 성공] 손목캠에서도 확인됨: pixel=({det.cx:.0f},{det.cy:.0f})")
                return True
            print("   [Astra 추정] 손목캠에서 안 보임 - 기존 탐색 그리드로 대체합니다.")
        except CollisionDetected as e:
            print(f"   [Astra 추정] 이동 중 충돌 감지 ({e}) - 기존 탐색 그리드로 대체합니다.")

    for dx, dy in SEARCH_OFFSETS:
        target = (HOME_XYZ[0] + dx, HOME_XYZ[1] + dy, HOME_XYZ[2])
        rc.move_to_xyz_converge(target, tolerance_m=0.015, max_iters=10)
        time.sleep(0.2)
        det, _ = get_pixel(cap, detect_fn, tries=10)
        if det is not None:
            print(f"   찾음: offset={dx:+.2f},{dy:+.2f} pixel=({det.cx:.0f},{det.cy:.0f})")
            return True
    print(f"   [실패] {name}을(를) 탐색 범위 내에서 찾지 못했습니다.")
    return False


COARSE_STEP_M = 0.025
COARSE_MAX_ITERS = 16
COARSE_TARGET_PX = 140.0  # "close enough to hand off to the fine Jacobian servo"


def coarse_center(rc: RobotController, cap: cv2.VideoCapture, detect_fn) -> bool:
    """Direction-agnostic hill-climb, for when the target is found near a
    frame edge (seen for real: a cube at pixel x=5 of 640) rather than
    somewhere the small PROBE_DELTA_M Jacobian probe can safely explore - a
    wrong-guess probe direction there loses the target off-frame before the
    Jacobian can even be estimated. Tries one axis/sign at a time, keeps going
    while it's helping, flips sign then axis when it stops - no assumption
    about which way image axes map to robot axes, same as the Jacobian
    approach, just cruder and safer for a badly off-center start."""
    axis = 0  # 0=x, 1=y
    sign = 1.0
    for it in range(COARSE_MAX_ITERS):
        det, _ = get_pixel(cap, detect_fn, tries=6)
        if det is None:
            print("   [coarse] 타겟을 놓쳤습니다.")
            return False
        dist = float(np.linalg.norm(GRASP_TARGET_PX - np.array([det.cx, det.cy])))
        if dist <= COARSE_TARGET_PX:
            print(f"   [coarse] 충분히 중앙 근처 (오차 {dist:.0f}px) - 정밀 서보로 전환합니다.")
            return True

        dx = COARSE_STEP_M * sign if axis == 0 else 0.0
        dy = COARSE_STEP_M * sign if axis == 1 else 0.0
        try:
            rc.nudge_xy(dx, dy)
        except CollisionDetected as e:
            print(f"   [coarse] 충돌 감지로 중단: {e}")
            return False
        time.sleep(0.15)

        det2, _ = get_pixel(cap, detect_fn, tries=6)
        if det2 is None:
            print(f"   [coarse] axis={axis} sign={sign:+.0f} 이동 후 타겟을 놓쳐서 되돌립니다.")
            rc.nudge_xy(-dx, -dy)
            time.sleep(0.15)
            sign, axis = (-1.0, axis) if sign > 0 else (1.0, 1 - axis)
            continue

        new_dist = float(np.linalg.norm(GRASP_TARGET_PX - np.array([det2.cx, det2.cy])))
        print(f"   [coarse] iter {it}: axis={axis} sign={sign:+.0f} {dist:.0f}px -> {new_dist:.0f}px")
        if new_dist < dist:
            continue  # helping - keep going the same way
        rc.nudge_xy(-dx, -dy)  # not helping - undo and try the other option
        time.sleep(0.15)
        sign, axis = (-1.0, axis) if sign > 0 else (1.0, 1 - axis)

    print("   [coarse] 반복 한도 내에 충분히 중앙에 도달하지 못했습니다.")
    return False


def estimate_jacobian(rc: RobotController, cap: cv2.VideoCapture, detect_fn) -> np.ndarray | None:
    """3-probe self-calibration: nudge +x then +y by a small known amount and
    watch how the target's pixel position shifts each time. Returns the 2x2
    matrix J where J @ [dx, dy] ~= [dpixel_x, dpixel_y], or None if the target
    was lost during probing (caller should re-search and retry)."""
    base, _ = get_pixel(cap, detect_fn)
    if base is None:
        return None
    base_px = np.array([base.cx, base.cy])

    rc.nudge_xy(PROBE_DELTA_M, 0.0)
    time.sleep(0.2)
    dx_det, _ = get_pixel(cap, detect_fn)

    rc.nudge_xy(-PROBE_DELTA_M, PROBE_DELTA_M)
    time.sleep(0.2)
    dy_det, _ = get_pixel(cap, detect_fn)

    rc.nudge_xy(0.0, -PROBE_DELTA_M)  # back to base
    time.sleep(0.2)

    if dx_det is None or dy_det is None:
        print("   [경고] 자가보정 중 타겟을 놓쳤습니다.")
        return None

    dx_px = np.array([dx_det.cx, dx_det.cy]) - base_px
    dy_px = np.array([dy_det.cx, dy_det.cy]) - base_px
    J = np.column_stack([dx_px / PROBE_DELTA_M, dy_px / PROBE_DELTA_M])
    if abs(np.linalg.det(J)) < 1e-3:
        print("   [경고] 자코비안이 거의 특이(degenerate)합니다 - 픽셀 변화가 너무 작습니다.")
        return None
    print(f"   자코비안 추정 완료:\n{J}")
    return J


def servo_to_target(rc: RobotController, cap: cv2.VideoCapture, detect_fn, name: str) -> bool:
    """Closed-loop pixel-error servo. Returns True once centered for
    CENTER_STABLE_FRAMES consecutive frames, False if it couldn't converge or
    lost the target for good."""
    if not search_for_target(rc, cap, detect_fn, name):
        return False

    # A target found right at a frame edge (seen for real: pixel x=5 of 640,
    # from the search grid landing there) can't safely go straight into the
    # small-probe Jacobian estimate - a wrong-guess probe direction loses it
    # off-frame before the Jacobian is even computed. Coarse-center first;
    # it's a no-op (one distance check, no moves) if already close.
    if not coarse_center(rc, cap, detect_fn):
        print("   [실패] 대략적인 중앙 정렬에 실패했습니다.")
        return False

    J = estimate_jacobian(rc, cap, detect_fn)
    if J is None:
        print("   자코비안 추정 실패 - 재탐색 후 재시도합니다.")
        if not search_for_target(rc, cap, detect_fn, name):
            return False
        if not coarse_center(rc, cap, detect_fn):
            print("   [실패] 재탐색 후 대략적인 중앙 정렬에 실패했습니다.")
            return False
        J = estimate_jacobian(rc, cap, detect_fn)
        if J is None:
            print("   [실패] 자가보정에 두 번째로 실패했습니다. 중단합니다.")
            return False
    J_inv = np.linalg.inv(J)

    print(f"\n[서보잉] {name} 중앙 정렬 시작...")
    stable = 0
    lost_streak = 0
    best_err = float("inf")
    stall_count = 0
    reestimates = 0
    # Broyden's-rank-one online update state: the pixel position and the
    # exact xy step actually applied on the previous iteration, so this
    # iteration's fresh detection gives one more (input, output) sample to
    # refine J with - see BROYDEN_UPDATE below for why.
    prev_px: np.ndarray | None = None
    prev_step: np.ndarray | None = None
    for it in range(MAX_SERVO_ITERS):
        det, _ = get_pixel(cap, detect_fn, tries=6)
        if det is None:
            lost_streak += 1
            print(f"   iter {it}: 타겟 놓침 ({lost_streak})")
            prev_px = None  # gap in the trail - don't attribute the next
            prev_step = None  # position jump to a step we can't verify happened cleanly
            if lost_streak >= 5:
                print("   [실패] 타겟을 계속 놓쳐서 중단합니다.")
                return False
            continue
        lost_streak = 0
        cur_px = np.array([det.cx, det.cy])

        # Broyden update: refine J with this iteration's real (step ->
        # pixel-shift) sample instead of only trusting the one-off finite-
        # difference probe from estimate_jacobian. Standard rank-one
        # correction for uncalibrated image-based visual servoing (e.g.
        # Jagersand et al., Hosoda & Asada - Broyden's method is the classic
        # cheap online update for the image Jacobian): given the step u just
        # applied and the pixel shift it actually produced, nudge J so
        # J@u matches that observation exactly, while disturbing J as little
        # as possible in directions orthogonal to u. This runs every
        # iteration (free - reuses motion already happening) instead of only
        # on staleness, which is what let the fixed-J version drift from
        # 12.9px back out to 95px+ before a full re-probe ever triggered.
        if prev_step is not None and prev_px is not None:
            step_norm2 = float(prev_step @ prev_step)
            # 2026-08-26: a real run diverged hard right after this update -
            # error climbed 9.7px -> 224px over ~20 iterations, never
            # recovering, immediately following a small (~2mm) step. Root
            # cause: Broyden's update divides by step_norm2, so a small step
            # amplifies whatever pixel-measurement noise rode along with it
            # (a few px of real detection jitter, already documented
            # elsewhere in this file) into a wildly wrong correction to J -
            # worst exactly when the servo is close and taking small,
            # careful steps, which is the regime where a bad J matters most.
            # BROYDEN_MIN_STEP_M skips the update entirely below that step
            # size instead of trusting a low-signal sample - J just stays
            # what it was (still gets refreshed on the next big-enough step,
            # or via the existing stall-triggered full re-probe below).
            if step_norm2 > BROYDEN_MIN_STEP_M**2:
                d_px = cur_px - prev_px
                predicted = J @ prev_step
                J_candidate = J + np.outer(d_px - predicted, prev_step) / step_norm2
                if abs(np.linalg.det(J_candidate)) > 1e-3:
                    J = J_candidate
                    J_inv = np.linalg.inv(J)
                # else: candidate update would make J near-singular (e.g. a
                # near-zero or noise-dominated step) - keep the previous J
                # rather than let one bad sample corrupt it.
        prev_px = cur_px

        err_px = GRASP_TARGET_PX - cur_px
        err_norm = float(np.linalg.norm(err_px))

        # 2026-08-26: a real run converged to 10-11px error (well inside
        # PIXEL_TOLERANCE) and STILL missed the cube entirely on descend.
        # Root cause, found by actually inspecting that run's estimated J:
        # its singular values were 8152 and 1233 px/m - a ~6.6x anisotropy,
        # meaning the SAME pixel error can correspond to a very different
        # real-world miss depending on which direction it points. A pixel-
        # space tolerance alone doesn't account for this: 10px in the
        # sensitive direction is ~1.2mm off, but 10px in the sloppy
        # direction is ~8mm off - comfortably enough to miss a ~35mm cube.
        # dxy below is what nudge_xy would apply to close the remaining
        # error - its norm is what actually matters (real distance to the
        # target), not the raw pixel count, so it's now the primary
        # convergence gate. PIXEL_TOLERANCE is kept as a secondary sanity
        # bound (guards against a badly wrong J making a real miss look like
        # a tiny physical distance).
        dxy = J_inv @ err_px
        physical_err_m = float(np.linalg.norm(dxy))
        if err_norm <= PIXEL_TOLERANCE and physical_err_m <= PHYSICAL_TOLERANCE_M:
            stable += 1
            best_err = min(best_err, err_norm)
            stall_count = 0
            prev_step = None  # no move this iteration - nothing to attribute next delta to
            print(
                f"   iter {it}: 중앙 정렬됨 (오차 {err_norm:.1f}px / 실거리 {physical_err_m*1000:.1f}mm, "
                f"{stable}/{CENTER_STABLE_FRAMES})"
            )
            if stable >= CENTER_STABLE_FRAMES:
                return True
            continue
        stable = 0

        # Even with the per-step Broyden refinement above, a long run of bad
        # luck (noisy detections, a probe that happened to land near the
        # local linearization's blind spot) can still stall. Keep the full
        # re-probe as a fallback of last resort - now needed far less often
        # since Broyden keeps J tracking the real local behavior continuously.
        if err_norm < best_err - 2.0:
            best_err = err_norm
            stall_count = 0
        else:
            stall_count += 1
        # 2026-08-26: a real run's *initial* estimate_jacobian probe came out
        # simply wrong (not stale, wrong from iteration 0) and error grew
        # ~monotonically 156px -> 253px over 19 iterations, each one a near-
        # max-size step in a consistently unhelpful direction, before the old
        # stall_count>=10 threshold ever triggered a re-probe - by then the
        # target was long gone off-frame. Broyden's per-step refinement
        # doesn't fully save this case: on a badly wrong starting J it takes
        # many samples to correct, same order as the divergence itself.
        # diverging catches this much faster: a big, definite jump past the
        # best-seen error (not just "no improvement") is a stronger signal
        # of a wrong J than 10 iterations of noise-level non-improvement -
        # no need to wait that long once it's this clear. stall_count's
        # threshold is also lowered (10 -> 5) for the ordinary case.
        diverging = err_norm > best_err + 40.0
        if (stall_count >= 5 or diverging) and reestimates < 4:
            reason = "발산 감지" if diverging else "진전 없음"
            print(f"   iter {it}: {reason} (최소 {best_err:.1f}px, 현재 {err_norm:.1f}px) - 자코비안 재추정합니다.")
            new_J = estimate_jacobian(rc, cap, detect_fn)
            if new_J is not None:
                J = new_J
                J_inv = np.linalg.inv(new_J)
                reestimates += 1
                best_err = err_norm
            stall_count = 0
            prev_px = None  # estimate_jacobian made its own untracked probe moves
            prev_step = None
            continue

        # dxy was already computed above (against the J_inv in effect at the
        # top of this iteration) - still valid here since the only way to
        # reach this line is via the "no progress" branch just above, which
        # `continue`s immediately whenever it actually changes J_inv.
        step = SERVO_GAIN * dxy
        mag = float(np.linalg.norm(step))
        # 2026-08-26: a fixed MAX_STEP_M was still too coarse for the final
        # approach once PHYSICAL_TOLERANCE_M made convergence genuinely
        # tight - sim reproduced a persistent limit-cycle oscillation
        # (10-17px error, step pegged at the 6mm cap in the low-sensitivity
        # direction every single iteration, alternating sign, never
        # settling within 90 iterations). A smaller cap once the target is
        # already visually close gives the low-sensitivity direction room to
        # take several smaller corrective steps instead of one that
        # overshoots past zero-error and back, every time.
        effective_max_step = MAX_STEP_M if err_norm > 30.0 else MAX_STEP_M * 0.4
        if mag > effective_max_step:
            step = step * (effective_max_step / mag)
        print(f"   iter {it}: 오차 {err_norm:.1f}px -> 이동 dx={step[0]*1000:.1f}mm dy={step[1]*1000:.1f}mm")
        try:
            rc.nudge_xy(float(step[0]), float(step[1]))
        except CollisionDetected as e:
            print(f"   [중단] 서보잉 중 충돌 감지: {e}")
            return False
        prev_step = step  # what we actually applied - next iteration's detection tells us what it did
        time.sleep(0.1)

    print(f"   [실패] {MAX_SERVO_ITERS}회 내에 중앙 정렬되지 않았습니다.")
    return False


# Real-world evidence this was needed: a 2026-08-26 run printed "바닥까지
# 도달 (접촉 없이)" (reached TABLE_Z with no contact - meaning the gripper's
# xy wasn't actually over the cube when it descended) then "그리퍼 닫음", and
# the script called that a success purely because nothing raised
# CollisionDetected during descend - but the user watching live confirmed
# no cube was ever caught. "did the descend hit something" and "is there
# actually an object between the jaws right now" are different facts, and
# only the second one is what matters. A parallel-jaw gripper without a
# force/tactile sensor can still answer the second question cheaply: command
# fully-closed and look at where it actually stops. Closing on nothing
# reaches (near) the same fully-closed reading every time; closing on a
# ~2-3cm cube stops well short of that, wedged against the object. This is
# the standard low-cost substitute for force feedback (see e.g. gripper-
# width/"lost-grip" signals used instead of tactile sensors in recent
# low-cost grasping work) and it only needs the position feedback this
# servo already reads every iteration - no new hardware.
GRIPPER_EMPTY_CLOSED_PCT = 5.0  # TODO: tune from a real empty-close reading -
# this is a placeholder guess, not yet measured on this unit.
GRASP_DETECT_MARGIN_PCT = 8.0  # final pos must clear empty-closed by at least this much

# Astra-depth-informed descend height, see this file's module docstring for
# why only a height DELTA (no extrinsics needed) is used, not a full xy fix.
CUBE_HEIGHT_MIN_M = 0.005  # below this, treat the reading as noise, not a real cube
CUBE_HEIGHT_MAX_M = 0.06  # above this, treat it as a bad reading (this cube is a few cm)
DESCEND_MARGIN_M = 0.005  # stop this far short of the estimated cube-top - let contact
# detection (not the depth estimate) catch the last few mm, same reasoning as leaving
# TABLE_Z itself as a backstop rather than trusting either signal completely alone.


def estimate_cube_height_m(rgb_path: str = ASTRA_RGB_FRAME_PATH, depth_path: str = ASTRA_DEPTH_MM_PATH) -> float | None:
    """Astra-depth-based estimate of how far the cube's top sticks up above
    the table, independent of the wrist cam and requiring no camera-to-robot
    calibration: table distance (median over the whole depth frame - the
    table dominates the view) minus cube distance (median within the Astra
    RGB detection's bbox, scaled into depth's lower native resolution) is a
    height delta, not an absolute coordinate, so no extrinsics are needed to
    turn it into something useful - see this file's module docstring.
    Returns None (never raises) if Astra isn't running, the cube isn't in
    its view, or the reading falls outside a plausible range - every caller
    must already have a TABLE_Z fallback for exactly this reason.

    rgb_path/depth_path default to the real published paths but are
    overridable - sim_dry_run.py's synthetic-data test passes isolated temp
    paths instead, so it doesn't race the real astra_s_live.py (if one
    happens to be running at the same time, on the same machine) overwriting
    the shared files mid-test."""
    astra_cap = PublishedFrameSource(rgb_path)
    ret, color = astra_cap.read()
    if not ret or color is None:
        return None
    det = detect_red_cube(color)
    if det is None:
        return None

    depth_mm = PublishedDepthSource(depth_path).read()
    if depth_mm is None:
        return None

    sx = depth_mm.shape[1] / color.shape[1]
    sy = depth_mm.shape[0] / color.shape[0]
    bx, by, bw, bh = det.bbox
    x0, y0 = max(0, int(bx * sx)), max(0, int(by * sy))
    x1, y1 = min(depth_mm.shape[1], int((bx + bw) * sx)), min(depth_mm.shape[0], int((by + bh) * sy))
    if x1 <= x0 or y1 <= y0:
        return None

    cube_region = depth_mm[y0:y1, x0:x1]
    cube_valid = cube_region[cube_region > 0]
    table_valid = depth_mm[depth_mm > 0]
    if cube_valid.size < 5 or table_valid.size < 100:
        return None  # not enough valid (non-zero) pixels to trust a median

    cube_depth_mm = float(np.median(cube_valid))
    table_depth_mm = float(np.median(table_valid))
    height_m = (table_depth_mm - cube_depth_mm) / 1000.0
    if not (CUBE_HEIGHT_MIN_M <= height_m <= CUBE_HEIGHT_MAX_M):
        return None
    return height_m


def descend_and_grasp(rc: RobotController) -> bool:
    """Returns True only if the final gripper position is consistent with
    something actually wedged between the jaws - see the note above on why
    "descend didn't hit anything" isn't sufficient evidence by itself."""
    cur = rc.current_gripper_xyz()

    cube_height_m = estimate_cube_height_m()
    if cube_height_m is not None:
        target_z = min(TABLE_Z + cube_height_m - DESCEND_MARGIN_M, cur[2])
        print(f"   [Astra] 큐브 높이 추정: {cube_height_m*1000:.1f}mm -> 목표 z={target_z:.4f} (TABLE_Z={TABLE_Z} 대신 사용)")
    else:
        target_z = TABLE_Z
        print("   [Astra] 높이 추정 실패 (뎁스 데이터 없음/큐브 안 보임) - 기존 TABLE_Z로 하강합니다.")

    target = (cur[0], cur[1], target_z)
    print(f"\n[하강] {cur} -> {target}")
    contacted = False
    try:
        rc.move_to_xyz(target, steps=25, step_delay_s=0.05, enforce_cap=False, stall_check=True)
        print("   1차 목표 높이까지 도달 (접촉 없이).")
    except CollisionDetected:
        # Expected/desired outcome most of the time: the gripper came down
        # onto the cube before reaching TABLE_Z and the collision safety net
        # correctly stopped it there and retreated slightly - that's exactly
        # where we want to be to grasp, so this isn't treated as a failure.
        print("   하강 중 접촉 감지 (큐브에 닿은 것으로 판단) - 현재 위치에서 집기를 시도합니다.")
        contacted = True

    # 2026-08-26: a real run centered to 0.6mm real distance (about as good
    # as this servo gets) and STILL found nothing at the Astra-estimated
    # height - the depth estimate itself had undershot across several runs
    # (23mm/26mm/37mm readings that don't obviously agree with each other).
    # Contact detection, not the depth estimate, is what actually decides
    # "found it" - the estimate only ever saved most of the blind-descent
    # distance when it's roughly right. So if phase 1 made no contact, keep
    # easing down the remaining distance to TABLE_Z instead of accepting
    # "reached the estimate, nothing there" as the final answer.
    if not contacted and target_z > TABLE_Z + 1e-4:
        cur2 = rc.current_gripper_xyz()
        remaining = (cur2[0], cur2[1], TABLE_Z)
        print(f"   접촉 없음 - 남은 구간을 TABLE_Z까지 이어서 하강합니다: {remaining}")
        try:
            rc.move_to_xyz(remaining, steps=25, step_delay_s=0.06, enforce_cap=False, stall_check=True)
            print("   TABLE_Z까지 도달 (접촉 없이) - 큐브가 그리퍼 아래에 없었을 수 있습니다.")
        except CollisionDetected:
            print("   2차 하강 중 접촉 감지 (큐브에 닿은 것으로 판단) - 현재 위치에서 집기를 시도합니다.")
            contacted = True
    time.sleep(0.2)
    final_pct = rc.set_gripper_pct_converge(0.0)
    time.sleep(0.3)
    grasped = final_pct > GRIPPER_EMPTY_CLOSED_PCT + GRASP_DETECT_MARGIN_PCT
    if grasped:
        print(f"   그리퍼 닫음 - 최종 {final_pct:.1f}% (빈 상태 기준 {GRIPPER_EMPTY_CLOSED_PCT}%보다 유의미하게 높음 -> 뭔가 집힌 것으로 판단).")
    else:
        print(f"   그리퍼 닫음 - 최종 {final_pct:.1f}% (빈 상태 기준 {GRIPPER_EMPTY_CLOSED_PCT}%에 가까움 -> 아무것도 못 집은 것으로 판단).")
    return grasped


def main():
    # 2026-08-26: follower 전용 USB 어댑터 보드(시리얼 5B3D042173)가 고장나 데이터 라인
    # 응답이 완전히 끊겨 leader 보드(5B3D042390)를 follower 팔에 대신 꽂아 씀 - udev가
    # 시리얼 번호로 심볼릭 링크를 걸어주기 때문에 /dev/so101_follower가 아니라
    # /dev/ttyACM0로 잡힌다. 원래 보드를 고치거나 교체하면 /dev/so101_follower로 되돌릴 것.
    rc = RobotController(port="/dev/ttyACM0")
    rc.connect()
    cap = PublishedFrameSource(WRIST_FRAME_PATH)
    if not cap.isOpened():
        print(
            "[오류] 손목캠 프레임이 없습니다. 먼저 다른 터미널에서 camera_hub.py를 "
            "~/lerobot_song_venv로 실행해주세요:\n"
            "  source ~/lerobot_song_venv/bin/activate && "
            "python custom_scripts/vision_pick_place/camera_hub.py"
        )
        rc.disconnect()
        return

    try:
        print("현재 관절각(deg):", rc.get_joint_deg())
        print("\n3초 후 시작합니다 (Ctrl+C로 중단 가능)...")
        time.sleep(3)

        rc.set_gripper_pct_converge(100.0)  # start open
        print("\n[이동] 홈 포즈로 이동:", HOME_XYZ)
        rc.move_to_xyz_converge(HOME_XYZ, tolerance_m=0.015, max_iters=20)

        if not servo_to_target(rc, cap, detect_red_cube, "빨간 큐브"):
            print("\n[중단] 큐브를 집지 못했습니다. 큐브 위치/조명을 확인하고 다시 실행해주세요.")
            rc.move_to_xyz_converge(RETURN_HOME_XYZ, tolerance_m=0.015, max_iters=20)
            return

        if not descend_and_grasp(rc):
            print(
                "\n[중단] 그리퍼가 아무것도 못 집은 것으로 판단됩니다 (최종 위치가 빈 상태와 "
                "비슷함). 큐브를 못 잡은 채로 들어올리지 않고 여기서 멈춥니다 - 그리퍼를 다시 "
                "열고 홈으로 돌아갑니다."
            )
            rc.set_gripper_pct_converge(100.0)
            rc.move_to_xyz_converge(RETURN_HOME_XYZ, tolerance_m=0.015, max_iters=20)
            return

        print("\n[상승] 집은 채로 들어올립니다.")
        rc.move_z(LIFT_M, steps=20, step_delay_s=0.05)

        cur = rc.current_gripper_xyz()
        rc.move_to_xyz((cur[0], cur[1], HOME_XYZ[2]), steps=20, step_delay_s=0.05, enforce_cap=False)

        if not servo_to_target(rc, cap, detect_black_bin, "검은 쓰레기통"):
            print("\n[중단] 쓰레기통을 찾지 못했습니다. 큐브를 든 채로 초기 위치로 복귀합니다 - 직접 확인해주세요.")
            rc.move_to_xyz_converge(RETURN_HOME_XYZ, tolerance_m=0.015, max_iters=20)
            return

        print(f"\n[하강] 쓰레기통 중앙 위에서 {BIN_DESCEND_M*100:.0f}cm 하강합니다.")
        try:
            rc.move_z(-BIN_DESCEND_M, steps=15, step_delay_s=0.05)
        except CollisionDetected:
            # A bin has walls/a rim - contact on the way down is a real
            # possibility, not necessarily wrong. Same handling as the cube
            # descent: stop where the safety net stopped it and release there.
            print("   하강 중 접촉 감지 (쓰레기통 벽/바닥에 닿은 것으로 판단) - 현재 위치에서 놓습니다.")

        print("\n[놓기] 그리퍼를 엽니다.")
        rc.set_gripper_pct_converge(100.0)
        time.sleep(0.3)

        print(f"\n[상승] 놓은 뒤 {BIN_DESCEND_M*100:.0f}cm 다시 올라옵니다.")
        rc.move_z(BIN_DESCEND_M, steps=15, step_delay_s=0.05)

        print("\n[복귀] 초기 위치로 복귀합니다.")
        rc.move_to_xyz_converge(RETURN_HOME_XYZ, tolerance_m=0.015, max_iters=20)

        print("\n완료.")
    except CollisionDetected as e:
        print(f"\n[중단] 충돌 감지로 작업을 중단했습니다: {e}")
    except KeyboardInterrupt:
        print("\n[중단] 사용자가 중단했습니다.")
    finally:
        cap.release()
        rc.disconnect()


if __name__ == "__main__":
    main()
