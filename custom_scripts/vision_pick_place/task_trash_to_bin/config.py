"""Central config for the trash -> bin task: every tunable number in one
place, same convention as the sibling task_red_cube_to_bin_new_gripper/.

2026-09-08: trimmed copy of ../task_red_cube_to_bin_new_gripper/config.py for
the color/shape-agnostic "pick up trash, drop in bin" MVP - RIGHT ARM ONLY
(left-arm calibration still unverified, see
task_red_cube_to_bin_new_gripper/CALIBRATION_HANDOFF.md). No per-side
overrides, no HSV/shape-detector constants, no FINE_SERVO/search-sweep
tuning constants - this task has no wrist-cam closed-loop servo (no generic
detector exists for an arbitrary clicked object), so only click_grasp_trash.py's
IK-hover-then-descend-and-grasp path is supported. Everything physically
measured below (ports, TABLE_Z, GRASP_TARGET_PX, BIN_POSE_XYZ, etc.) is
carried over unchanged from the right arm's own calibration - same hardware.
"""

from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent
VISION_DIR = TASK_DIR.parent  # ~/lerobot/custom_scripts/vision_pick_place - shared camera publish paths live here

# --- Robot connection -------------------------------------------------------
# 2026-09-07: confirmed via `uv run lerobot-find-port` (physical unplug/replug,
# not a guess) that /dev/so101_follower (serial 5B3D042390) is the physically
# RIGHT follower arm - every constant below this point was measured through
# this exact port. 2026-09-08: right-arm only today, left-arm calibration
# unverified - see task_red_cube_to_bin_new_gripper/CALIBRATION_HANDOFF.md.
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

# 2026-08-26 (measured on the sibling task, same physical arm): forcing ANY
# nonzero orientation_weight on this 5-DOF arm's IK blew position error up to
# 16-253mm across the workspace. Position-only IK; the last few mm are
# handled by contact detection during descent, not a hard IK constraint.
IK_ORIENTATION_WEIGHT = 0.0
IK_ITERATIONS = 6  # placo's solver needs several passes fed back as the next guess to land within ~1mm

# --- Table / grasp geometry --------------------------------------------------
# Real contact xyz measured on the new gripper via probe_table_height_manual.py
# (see sibling task's config.py history) - same 3mm-margin convention.
TABLE_Z = -0.0042  # 3mm 여유 포함

# 2026-09-09 probe_reach.py 실측: CALIB_HOVER_Z(TABLE_Z+0.05)에서 베이스로부터
# 이 반경을 넘어가면 z가 테이블 쪽으로 sag하며 IK가 목표에 도달하지 못한다
# (0.27m는 성공, 0.29~0.32m대는 실패 확인). 안전영역 판정은 이 값 기준 원호로 한다.
MAX_REACH_XY_M = 0.27

# This is an arbitrary trash task, not the sibling small-cube task: retain
# only readings that are above Astra's near-table noise yet cover ordinary
# cups/packaging. Descent contact detection remains the safety limit.
OBJECT_HEIGHT_MIN_M = 0.002
OBJECT_HEIGHT_MAX_M = 0.12
HEIGHT_SAMPLE_RADIUS_PX = 8  # RGB-pixel radius sampled around a click for depth height estimation
HEIGHT_SAMPLE_MIN_VALID_PIXELS = 25
DESCEND_MARGIN_M = 0.005  # stop this far short of the Astra-estimated object top - let
# contact detection (not the depth estimate) catch the last few mm

LIFT_M = 0.08
BIN_DESCEND_M = 0.05

# --- Poses -------------------------------------------------------------------
# Raised hover pose to move to before descending - not a resting position.
#
# 2026-09-09: z was 0.13, probe_reach_hover13.py showed that height has almost
# no lateral swing (any y offset failed with 5-6cm z-sag, only pure y=0 x-reach
# worked). Lowered to match TABLE_Z+0.05 (=CALIB_HOVER_Z in calibrate_camera.py),
# the height probe_reach.py already confirmed reaches y=+-0.10, x up to 0.27
# cleanly - same height MAX_REACH_XY_M/the reach-arc overlay are calibrated for.
SEARCH_HOVER_XYZ = (0.23, 0.0, TABLE_Z + 0.05)

# Not hardcoded here as a fallback of last resort - main.py reads the arm's
# actual joint positions live at session start and uses THAT as home_pose.
# This constant only exists as a sanity-check reference.
REFERENCE_IDLE_XYZ = (0.10259099, 0.00435801, -0.02739574)

# Fixed hover pose above the trash bin - click_grasp_trash.py's place step
# moves here and opens the gripper after a successful grasp+lift. Measured
# once by hand (same method as TABLE_Z), not detected live. None until
# measured; the place step refuses to run (holds the object instead) while
# this is None.
BIN_POSE_XYZ: tuple[float, float, float] | None = (0.3666, -0.0423, 0.0098)  # 2026-09-07: right-arm hand-guided bin pose

# --- Camera / vision -----------------------------------------------------
FRAME_W, FRAME_H = 640, 480
IMG_CENTER = (FRAME_W / 2.0, FRAME_H / 2.0)

# Wrist-cam jaw-tip pixel, measured on the real hardware - kept for parity
# with the sibling task's config even though this task's script doesn't run
# a wrist-cam closed-loop servo (no generic per-object detector exists);
# left here in case a future generic detector wants it.
GRASP_TARGET_PX = (221.0, 362.0)

# Published-frame paths: camera_hub.py / astra_s_live.py (the sibling
# vision_pick_place/ scripts) are the sole owners of the actual camera
# devices and publish here via atomic write - this task reads those files
# rather than opening any camera device itself.
WRIST_FRAME_PATH = "/tmp/vsp_wrist.png"
ASTRA_RGB_FRAME_PATH = "/tmp/vsp_astra_rgb.png"
ASTRA_DEPTH_MM_PATH = "/tmp/vsp_astra_depth_mm.npy"
ASTRA_DEPTH_REGISTRATION_STATUS_PATH = "/tmp/vsp_astra_depth_registration.json"
ASTRA_IR_FRAME_PATH = "/tmp/vsp_astra_ir.png"
FRAME_STALE_TIMEOUT_S = 5.0
# astra_s_depth_hub.py and astra_s_live.py request OpenNI2 registration. Set
# false only when a device/driver cannot provide it, to use the FOV fallback.
ASTRA_DEPTH_REGISTERED_TO_COLOR = True
# registration 뒤에도 남는 고정 오프셋을 depth 영상에 적용한다. 양수 x/y는
# depth를 각각 오른쪽/아래로 이동시키며, 실제 측정 전에는 반드시 (0, 0)을 유지한다.
ASTRA_DEPTH_REGISTERED_OFFSET_PX = (0, 0)

HOMOGRAPHY_PATH = VISION_DIR / "homography.json"

# 2026-09-01 (measured on the sibling task, same physical rig): the robot's
# own base/gripper-mount hardware sits in a fixed spot of the Astra RGB view
# (Astra is a fixed external camera). Kept for parity even though this
# task's click-only targeting doesn't run an auto-detector that would need
# it today.
ROBOT_EXCLUSION_BBOX_PX = (450, 230, 640, 480)  # (x0, y0, x1, y1) - re-measure if the camera/mount ever moves

# --- Grasp verification -----------------------------------------------------
# Fully measured via calibrate_grasp.py + manual_grasp_calibration.py on the
# new jaw (same physical gripper as the sibling task).
GRIPPER_EMPTY_CLOSED_PCT = 2.0
GRASP_DETECT_MARGIN_PCT = 8.5  # judge threshold = 10.5%

GRIPPER_STALL_EPS_PCT = 1.0  # position change below this doesn't count as progress
GRIPPER_STALL_CONSECUTIVE = 2  # consecutive no-progress iterations before bailing early

# --- Retry -------------------------------------------------------------------
MAX_GRASP_ATTEMPTS = 3  # per the spec's "실패 시 재시도 로직: 최대 N회"
