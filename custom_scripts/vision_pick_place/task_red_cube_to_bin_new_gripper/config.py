"""Central config for the red-cube -> bin task: every tunable number in one
place, per the 2026-08-26 spec's requirement that nothing be hardcoded deep
inside a state/module. Every constant below with a date comment was found or
tuned against real hardware today, not guessed - see
~/.claude memory (orbbec-astra-s-lerobot.md) for the full incident-by-
incident history if a number here ever needs revisiting.

2026-09-01: copy of ../task_red_cube_to_bin/config.py for the physically
replaced gripper (NEW GRIPPER) - original left untouched as reference.
Everything below is carried over unchanged EXCEPT what's marked "NEW GRIPPER"
- those depend on the physical jaw and need re-measuring on the new hardware
before this task is trusted again.
"""

from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent
VISION_DIR = TASK_DIR.parent  # ~/lerobot/custom_scripts/vision_pick_place - shared camera publish paths live here

# --- Robot connection -------------------------------------------------------
# 2026-08-26: the follower arm's OWN USB adapter board (serial 5B3D042173) was
# dead, so this ran on the leader arm's board (/dev/ttyACM0) instead. 2026-09-01:
# reverted per that note's own instruction - today's teleoperate session drove
# leader (/dev/so101_leader) and follower (/dev/so101_follower) simultaneously
# without error, so the follower's own board is confirmed alive again.
# 2026-09-07: confirmed via `uv run lerobot-find-port` (physical unplug/replug,
# not a guess) that /dev/so101_follower (serial 5B3D042390) is the physically
# RIGHT follower arm - every constant below this point was measured through
# this exact port, so it's the right arm's calibration, not the left's. See
# LEFT_OVERRIDES below for the physically-left arm (still unmeasured).
FOLLOWER_PORT = "/dev/so101_follower"

JOINT_LIMITS_DEG = {
    "shoulder_pan": (-118.0, 118.0),
    "shoulder_lift": (-105.0, 105.0),
    "elbow_flex": (-98.0, 98.0),
    "wrist_flex": (-102.0, 102.0),
    "wrist_roll": (-179.0, 179.0),
    "gripper": (-8.0, 99.0),  # percent-open, not degrees - so_follower.py's RANGE_0_100 norm mode
}
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
ALL_JOINTS = ARM_JOINTS + ["gripper"]

MAX_RELATIVE_TARGET_DEG = 15.0  # lerobot's own per-send_action clamp
MAX_MOVE_DELTA_DEG = 40.0  # outright-refuse-the-whole-move cap, before any motion is sent

STALL_THRESHOLD_DEG = 10.0  # actual-vs-commanded lag that counts as "not really moving"
STALL_CHECK_EVERY = 3  # steps between stall checks - ordinary servo catch-up lag isn't a stall
STALL_CONSECUTIVE = 3  # consecutive stalled checks before treating it as a real block

URDF_PATH = VISION_DIR / "so101_urdf" / "so_arm101.urdf"
IK_TARGET_FRAME = "gripper_frame_link"

# 2026-08-26: measured directly - forcing ANY nonzero orientation_weight on
# this 5-DOF arm's IK blew position error up to 16-253mm across the
# workspace (tested at weights 0.05-1.0, from both a neutral and the arm's
# real current-pose starting guess - not a starting-guess artifact). 5
# joints can't independently satisfy position (3 DoF) + orientation (3 DoF)
# almost anywhere in this workspace. Position-only IK plus the wrist-cam
# closed-loop servo (which re-measures the camera-to-gripper relationship
# fresh every approach, see FINE_SERVO state) is what actually handles a
# floating gripper orientation, not a hard IK constraint.
IK_ORIENTATION_WEIGHT = 0.0
IK_ITERATIONS = 6  # placo's solver needs several passes fed back as the next guess to land within ~1mm

# --- Table / grasp geometry --------------------------------------------------
# 2026-08-26: was 0.045 all session on an unverified "earlier IK test" value.
# Root cause of nearly every real grasp failure today: the user physically
# drove the gripper down onto the table and read back the live xyz at actual
# contact - (0.122, -0.0003, -0.0003). z was ~0.000, not 0.045 - a 45mm error.
# NEW GRIPPER (2026-09-01): re-measured via probe_table_height_manual.py
# (hand-guided - the auto-descent probe failed to detect real contact twice,
# see that script's docstring) - real contact xyz (0.161, -0.006, -0.0083).
# z is ~8.3mm lower than the old gripper's contact plane (~0.000) - consistent
# with a physically longer new jaw. Same 3mm-margin convention as before.
TABLE_Z = -0.0042  # 3mm 여유 포함
CUBE_HEIGHT_MIN_M = 0.005  # below this, an Astra height reading is treated as noise
CUBE_HEIGHT_MAX_M = 0.06  # above this, treated as a bad reading (this cube is a few cm)
# TODO: measure/tune on the real Astra scene; RGB-pixel radius sampled around a click for depth height estimation.
HEIGHT_SAMPLE_RADIUS_PX = 8
DESCEND_MARGIN_M = 0.005  # stop this far short of the Astra-estimated cube top - let
# contact detection (not the depth estimate) catch the last few mm

LIFT_M = 0.08
BIN_DESCEND_M = 0.05

# --- Poses -------------------------------------------------------------------
# Raised hover pose to SEARCH from (wide camera view, clearance) - not a
# resting position. Center of the rectangle used in calibrate_camera.py's
# touch-point calibration.
SEARCH_HOVER_XYZ = (0.23, 0.0, 0.13)

# 2026-08-26: NOT hardcoded here as a fallback of last resort - main.py reads
# the arm's actual joint positions live at session start and uses THAT as
# home_pose, per the spec's "read_joint_positions(), no hardcoding"
# requirement. This constant only exists as a sanity-check reference (the
# pose the user physically drove the arm to and confirmed as "초기위치" once
# today) in case a caller needs a value before ever connecting.
REFERENCE_IDLE_XYZ = (0.10259099, 0.00435801, -0.02739574)

# Fixed hover pose above the trash bin - click_grasp_bimanual.py's place step
# moves here and opens the gripper after a successful grasp+lift. The bin is
# stationary hardware on the trolley, so (unlike the clicked pick target)
# this is measured once by hand, not detected live - see measure_bin_pose.py
# (same hand-guided-then-read-FK method as TABLE_Z). None until measured;
# the place step refuses to run (holds the object instead) while this is None.
BIN_POSE_XYZ: tuple[float, float, float] | None = (0.3666, -0.0423, 0.0098)  # 2026-09-07: right-arm hand-guided bin pose

# --- Camera / vision -----------------------------------------------------
FRAME_W, FRAME_H = 640, 480
IMG_CENTER = (FRAME_W / 2.0, FRAME_H / 2.0)

# 2026-08-26: measured from real saved wrist-cam frames (the gripper's own
# jaw tips are visible in-shot, eye-in-hand mount) - the wrist camera does
# NOT look straight down the gripper's grasp axis, so "cube centered in the
# image" (IMG_CENTER) was never the same thing as "cube between the jaws".
# Caveat: since IK_ORIENTATION_WEIGHT=0.0 lets wrist_roll float, this offset
# could rotate around IMG_CENTER if a run lands on a very different roll
# than it was measured from - re-measure from a fresh saved frame if grasps
# keep missing.
# NEW GRIPPER (2026-09-01): re-measured via measure_grasp_target_px.py
# against the new gripper's jaw tips in a live wrist-cam frame (brightness
# bumped to 60 first - the tuned brightness=0/gamma=1 was too dark to see
# the jaw clearly on this camera/gripper combo).
GRASP_TARGET_PX = (221.0, 362.0)

# Published-frame paths: camera_hub.py / astra_s_live.py (the sibling
# vision_pick_place/ scripts) are the sole owners of the actual camera
# devices and publish here via atomic write - this task reads those files
# rather than opening any camera device itself, since two processes can't
# both hold a UVC/OpenNI2 device open for streaming. Same paths, defined
# fresh here rather than imported, per this task's own config being self-
# contained.
WRIST_FRAME_PATH = "/tmp/vsp_wrist.png"
ASTRA_RGB_FRAME_PATH = "/tmp/vsp_astra_rgb.png"
ASTRA_DEPTH_MM_PATH = "/tmp/vsp_astra_depth_mm.npy"
ASTRA_IR_FRAME_PATH = "/tmp/vsp_astra_ir.png"
FRAME_STALE_TIMEOUT_S = 5.0

HOMOGRAPHY_PATH = VISION_DIR / "homography.json"

# 2026-09-01: the robot's own base/gripper-mount hardware sits in a fixed
# spot of the Astra RGB view (Astra is a fixed external camera - it doesn't
# move with the arm), consistently the bottom-right corner across every real
# frame checked this session. Qwen's own prompt instruction to ignore robot
# hardware isn't reliable enough alone - a real live frame got the perforated
# white servo housing back labeled "white box with holes", a plausible-
# looking but real false positive that would otherwise try to grasp the
# robot's own gripper. A static pixel-region exclusion is simpler and more
# robust than trying to filter by label text (perception_qwen.py's callers
# apply this, not perception.py's HSV path, which never had this problem).
ROBOT_EXCLUSION_BBOX_PX = (450, 230, 640, 480)  # (x0, y0, x1, y1) - re-measure if the camera/mount ever moves

# HSV thresholds - see perception.py for how these were tuned (sampled
# against real miss-frames, not guessed).
LOWER_RED_1 = (0, 60, 25)
UPPER_RED_1 = (10, 255, 255)
LOWER_RED_2 = (170, 60, 25)
UPPER_RED_2 = (180, 255, 255)
MIN_CUBE_CONTOUR_AREA = 200

LOWER_BLACK = (0, 0, 0)
UPPER_BLACK = (180, 90, 70)
MIN_BIN_CONTOUR_AREA = 800

# Shape filter (solidity + aspect ratio + max-area cap), added 2026-08-26 so
# a hand/arm/cable in frame can't out-vote the real target just by being the
# largest same-colored blob. MIN_SOLIDITY lowered from an initial 0.85 the
# same day after it wrongly rejected a real, texture-noisy cube frame
# (measured solidity 0.837, just under 0.85) - 0.65 keeps a wide margin
# below that reading while staying far above an actual hand's ~0.41.
FRAME_AREA_HINT = FRAME_W * FRAME_H
MAX_CUBE_AREA_FRAC = 0.5
MIN_CUBE_SOLIDITY = 0.65
CUBE_ASPECT_RANGE = (0.4, 2.5)
MAX_BIN_AREA_FRAC = 0.7
MIN_BIN_SOLIDITY = 0.6
BIN_ASPECT_RANGE = (0.3, 3.0)

# 2026-08-26: this specific wrist camera ("USB 2.0 PC Cam") exposes no
# exposure_auto/exposure_absolute control at all (checked live via
# `v4l2-ctl --list-ctrls`) - true manual exposure isn't possible. What fixed
# real, badly-overexposed frames (background blown to solid white) was
# dropping brightness/gamma to their minimums; contrast/saturation changes
# made it worse or did nothing.
WRIST_V4L2_CTRLS = {"brightness": 0, "gamma": 1}

# --- Search sweep (fallback if the Astra coarse guess misses) ---------------
SEARCH_OFFSETS = [
    (0.0, 0.0),
    (0.03, 0.0), (0.06, 0.0), (-0.03, 0.0), (-0.06, 0.0),
    (0.0, 0.04), (0.0, 0.08), (0.0, -0.04), (0.0, -0.08),
    (0.03, 0.04), (-0.03, 0.04), (0.03, -0.04), (-0.03, -0.04),
]

# --- FINE_SERVO tuning (all found/tuned against real hardware 2026-08-26) --
PIXEL_TOLERANCE = 22.0  # secondary sanity bound only - see PHYSICAL_TOLERANCE_M
PHYSICAL_TOLERANCE_M = 0.004  # the real convergence gate: remaining pixel error run
# through the estimated Jacobian's inverse, in meters - see task_state_machine.py's
# FINE_SERVO state for why pixel error alone isn't enough once the image Jacobian
# is anisotropic (a real reading had singular values 8152 vs 1233 px/m, ~6.6x).
CENTER_STABLE_FRAMES = 3
MAX_SERVO_ITERS = 90
PROBE_DELTA_M = 0.008
SERVO_GAIN = 0.5
MAX_STEP_M = 0.006  # was 0.02 - too large a step in the sensitive axis caused endless
# oscillation once PHYSICAL_TOLERANCE_M made convergence genuinely tight
CLOSE_ERR_PX = 30.0  # once error is already under this, shrink the step cap further
CLOSE_MAX_STEP_FRAC = 0.4  # to CLOSE_ERR_PX * CLOSE_MAX_STEP_FRAC... see below
BROYDEN_MIN_STEP_M = 0.0025  # below this step size, skip the Broyden update - a small
# step amplifies ordinary detection noise into a wrong correction to J
STALL_ITERS = 5  # non-improving iterations before a full Jacobian re-probe
DIVERGE_PX = 40.0  # a jump this far past the best-seen error triggers an immediate
# re-probe, without waiting for STALL_ITERS
MAX_REESTIMATES = 4

COARSE_STEP_M = 0.025
COARSE_MAX_ITERS = 16
COARSE_TARGET_PX = 140.0

# --- Grasp verification -----------------------------------------------------
# NEW GRIPPER (2026-09-01): fully measured now via calibrate_grasp.py +
# manual_grasp_calibration.py on the new jaw. 5 empty-closed trials
# (1.8-2.0%, max 2.0%) and 5 cube-closed trials (19.0-20.2%, min 19.0%) -
# clean separation, no overlap. margin = (cube_min - empty_max) / 2 =
# (19.0 - 2.0) / 2 = 8.5, per calibrate_grasp.py's own formula.
GRIPPER_EMPTY_CLOSED_PCT = 2.0
GRASP_DETECT_MARGIN_PCT = 8.5  # judge threshold = 10.5%

# 2026-09-01: set_pct_converge had no "stop pushing if actually stuck"
# check - wedged against an object (the normal, expected way a grasp
# succeeds) it kept re-issuing a fresh ~35pt command every ~0.34s for the
# full max_iters=15 regardless of whether the jaw was moving at all, driving
# current into the jam the whole time. Firmware Protection_Current/
# Overload_Torque (see kinematics.py's _protect_arm_motors, mirrored onto
# the gripper by so_follower.py's own configure()) caps peak current either
# way, but cutting the stall duration short is extra margin against heat
# buildup on repeated grasp attempts, same STALL_THRESHOLD_DEG/
# STALL_CONSECUTIVE pattern kinematics.py already uses for the arm joints.
GRIPPER_STALL_EPS_PCT = 1.0  # position change below this doesn't count as progress
GRIPPER_STALL_CONSECUTIVE = 2  # consecutive no-progress iterations before bailing early

# --- Retry -------------------------------------------------------------------
MAX_GRASP_ATTEMPTS = 3  # per the spec's "실패 시 재시도 로직: 최대 N회"

# --- Left-arm overrides ------------------------------------------------
# Physically different arm: different port, different measured table-contact
# height, gripper jaw thresholds, wrist-cam grasp target pixel, etc. Must be
# re-measured for the left arm the same way the right arm's values above were
# (calibrate_grasp.py / probe_table_height_manual.py / measure_grasp_target_px.py)
# before LEFT is trusted - these are PLACEHOLDER copies of RIGHT's values until
# then, not measured for the left arm yet.
#
# 2026-09-07: identified via `uv run lerobot-find-port` (user unplugged/replugged
# each cable) - the flat/default constants above (TABLE_Z, GRASP_TARGET_PX, etc.)
# were all measured through /dev/so101_follower, which that same check confirmed
# is the physically-RIGHT follower arm (serial 5B3D042390). So the physically-LEFT
# follower (serial 5B14029976, /dev/ttyACM3 as of that check) is the one that's
# actually still unmeasured - this dict (and every "left"/"right" label in this
# module) is now aligned to PHYSICAL left/right, not to some other convention,
# specifically so --left-port always means "plug in the physically-left arm here"
# and can't be silently swapped with --right-port. No stable udev alias exists
# for this serial yet (only so101_follower/so101_leader are aliased, see
# /etc/udev/rules.d/99-serial.rules), so the by-id path is used here instead of a
# raw ttyACMn path to survive port renumbering on replug.
LEFT_OVERRIDES = {
    "FOLLOWER_PORT": "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14029976-if00",
    "TABLE_Z": -0.0042,  # 2026-09-07: left-arm hand-guided table contact, 3mm margin
    "CUBE_HEIGHT_MIN_M": CUBE_HEIGHT_MIN_M,
    "CUBE_HEIGHT_MAX_M": CUBE_HEIGHT_MAX_M,
    "DESCEND_MARGIN_M": DESCEND_MARGIN_M,
    "REFERENCE_IDLE_XYZ": REFERENCE_IDLE_XYZ,
    "GRASP_TARGET_PX": (410.0, 152.0),  # 2026-09-07: left wrist-camera jaw-tip click
    "WRIST_FRAME_PATH": "/tmp/vsp_wrist_left.png",  # distinct path - camera_hub publishes per-side
    "BIN_POSE_XYZ": (0.2334, 0.0495, 0.0188),  # 2026-09-07: left-arm hand-guided bin pose
    "ROBOT_EXCLUSION_BBOX_PX": ROBOT_EXCLUSION_BBOX_PX,
    "WRIST_V4L2_CTRLS": WRIST_V4L2_CTRLS,
    "GRIPPER_EMPTY_CLOSED_PCT": 1.6,
    "GRASP_DETECT_MARGIN_PCT": 31.4,
}

# Snapshot of the original flat (right-arm) values, keyed the same as
# LEFT_OVERRIDES, captured now (module load time) before apply_side() can
# ever run - apply_side("right") restores from this, not from whatever
# apply_side("left") last left in the globals.
RIGHT_DEFAULTS = {key: globals()[key] for key in LEFT_OVERRIDES}


class _SideView:
    """Read-only view of config for one arm side: attribute lookups fall back
    to this module's own globals for anything not in the overrides dict, so
    callers that only need shared constants (JOINT_LIMITS_DEG, IK_ITERATIONS,
    HSV thresholds, etc.) don't need a side-specific copy of everything."""

    def __init__(self, overrides: dict):
        self._overrides = overrides

    def __getattr__(self, name):
        if name in self._overrides:
            return self._overrides[name]
        return globals()[name]


_RIGHT_VIEW = _SideView({})
_LEFT_VIEW = _SideView(LEFT_OVERRIDES)


def for_side(side: str) -> "_SideView":
    """Returns a config view for "left" or "right" - attribute access (e.g.
    for_side("left").TABLE_Z) resolves to that side's override if one
    exists, else falls back to this module's flat (right-arm) constants.
    Existing code that does `import config; config.TABLE_Z` is untouched and
    keeps meaning "right arm" - for_side() is purely additive for bimanual
    callers."""
    if side == "left":
        return _LEFT_VIEW
    if side == "right":
        return _RIGHT_VIEW
    raise ValueError(f"unknown side {side!r}, expected 'left' or 'right'")


def apply_side(side: str) -> None:
    """Overwrites this module's own flat globals (TABLE_Z, GRASP_TARGET_PX,
    etc.) with the given side's values, so task_state_machine.py/
    perception.py/gripper.py's existing bare `config.X` reads pick up the
    left arm's calibration without any changes to those files. Only safe
    because the two arms run strictly sequentially in one process (never
    concurrently) - see orchestrator_bimanual.py. Call this BEFORE running a
    given arm's task_state_machine.run(), every time you switch arms.
    # ponytail: mutates shared module globals - fine for one-process
    # sequential execution, would need per-process config (or threading
    # for_side() through every callsite) if/when true concurrent bimanual
    # execution is ever built.
    """
    if side not in ("left", "right"):
        raise ValueError(f"unknown side {side!r}, expected 'left' or 'right'")
    overrides = LEFT_OVERRIDES if side == "left" else {}
    for key, value in {**RIGHT_DEFAULTS, **overrides}.items():
        globals()[key] = value
