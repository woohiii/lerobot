"""SO-101 red-cube -> bin pick-and-place, ported directly from a working
reference (`~/Downloads/calibrate_and_pick (1)/calibrate_and_pick.py` -
Windows, COM port, general "can" object) to this machine's real hardware
(Linux, /dev/ttyACM0, wrist cam only). Deliberately a single interactive
file, same philosophy as the reference, NOT the modular config.py/
perception.py/kinematics.py/task_state_machine.py package this project also
has under task_red_cube_to_bin/ - this is a from-scratch rebuild on a
different, much simpler design:

  - Pixel -> joint-space is one directly-fit affine matrix (matrix_M @
    [u, v, 1] -> 6 raw servo goal positions), taught by physically moving the
    gripper to touch/hover over the target at a handful of points and
    recording (pixel, joint) pairs - NOT a URDF/IK + camera-extrinsic
    pipeline. No top-down orientation constraint, no wrist-cam fine visual
    servo loop - one detection, one mapped move, closed-loop only in the
    sense of "verify after closing the gripper, retry from scratch if it
    missed".
  - Raw `scservo_sdk` register writes (this robot's actual STS3215 servos:
    protocol_version=0, baudrate=1,000,000, IDs 1-6 = shoulder_pan,
    shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper - confirmed
    against lerobot's own so_follower.py motor table, not guessed) instead of
    lerobot's SOFollower wrapper.
  - Grasp success verified by the gripper servo's OWN encoder position
    (J6/servo 6) against a manually taught empty-vs-holding threshold, not a
    vision check.

What's ADDED here, not in the reference file as received: the reference
loads `calib_matrix.json`'s `matrix_M` but never shows how it's produced
(`calib_pts_pixel`/`calib_pts_joints` are declared, never appended to or
fit) - this version adds that missing half as new [C]/[M] keys, described
below, since without it there is no way to ever populate matrix_M for a new
object/camera/workspace.

2026-08-26 update 1: YOLO WAS tried first here (yolo11n.pt, COCO-pretrained),
per the user's initial choice - confirmed live, exactly as this section
originally warned, that it detects nothing on the actual cube (COCO has no
"cube"/"block" class) and instead misclassified the gripper's own visible
finger structure as "toilet". Swapped to the HSV+contour detector this
project already validated (same approach as vision_pick_place/perception.py
and task_red_cube_to_bin/perception.py's detect_red_cube, inlined here
rather than imported to keep this one file self-contained per the user's
own choice of a single-file design) - see detect_target_current_position()
below.

2026-08-26 update 2: wrist cam (eye-in-hand) alone turned out to be
structurally wrong for the pixel->joint MATRIX specifically - not a
detection-quality problem, a geometry one. matrix_M's whole premise (one
fixed pixel->joint-space affine map) only makes sense if pixel position has
a stable relationship to real-world position, which requires the camera's
viewpoint to not move independently of what it's imaging. A wrist-mounted
camera moves WITH the arm, so after any autonomous move (e.g. ending at
place_pose, aimed at the bin) the same camera is no longer looking anywhere
near the cube at all - confirmed live: [A] failed 3/3 retries immediately,
"화면에서 큐브를 찾지 못했습니다", because the arm was still sitting at the
last-taught pose and the wrist cam simply wasn't pointed at the cube's
location any more.

2026-08-26 update 3: per explicit user direction, NOT a swap - all three
cameras run together, each doing the job it's actually suited for:
  - **Astra S RGB** (`orbbec_color_camera.py`'s `ThreadedOrbbecRGBDCamera` -
    this project's own already-working OpenNI2 driver, reused as-is) is the
    FIXED eye-to-hand view matrix_M is fit against and that
    detect_target_current_position()/[C] calibration both read from - fixes
    update 2's problem directly, since its viewpoint never moves regardless
    of arm pose.
  - **Astra S depth** (same class, comes for free alongside its color
    stream) filters detections by plausible working-surface distance
    (DEPTH_MIN_MM/DEPTH_MAX_MM) - rejects an HSV false-positive that isn't
    actually sitting at table depth (e.g. something red in the far
    background), which color alone can't tell apart. This is the literal
    "정확도" ask - depth as a same-color-blob-vs-real-object filter, not
    used for XYZ (still no camera-to-robot extrinsic calibration here, same
    as this project's other tracks - see project memory).
  - **Wrist cam** is now a secondary VERIFICATION step, not the primary
    detector: right before closing the gripper (after matrix_M's predicted
    move already got the arm there), one wrist-cam frame is checked for a
    red blob too - since by that point the arm has actually moved close to
    the cube, the wrist cam's eye-in-hand view is finally expected to
    contain it. A miss here aborts that attempt and retries the whole
    search instead of blindly closing on nothing - see
    execute_smart_pick_and_place()'s step "2.5".

**Old calib_matrix.json's `matrix_M`/calibration points from the wrist-cam-
only era are invalid after update 2's camera-source switch (different
camera, different pixel space) and were cleared then** -
home_pose/place_pose/gripper thresholds are joint-space, camera-independent,
and were kept throughout both updates.

2026-08-26 update 4: a single [C] key (read whatever pixel Astra currently
sees + whatever joints the arm is currently at) has a real problem once the
gripper is actually AT the cube for real, not just briefly passed through by
a hand: the arm/gripper itself sits between Astra's fixed viewpoint and the
cube, so Astra can't see the cube at all at the exact moment [C] needs a
pixel - not a transient occlusion the update-3 retry window can wait out
(the arm has to be there to record the joint target; it won't move out of
its own way). Split into two keys: [V] snapshots the pixel while the arm is
still clear, [C] (pressed after moving the gripper in) pairs that saved
pixel with the joints at the grasp position. Decouples "where Astra sees it"
from "where the arm has to be to grasp it" - the two can now happen at
different arm positions instead of needing an impossible simultaneous view.

2026-08-27 update 6: move_smoothly()'s inner loop only ever wrote servo goals
and time.sleep()'d - it never read a camera frame or called cv2.imshow()/
waitKey() while a move was in progress. Every click triggers do_pick/do_place,
which chains several ~2-3s move_smoothly() calls back to back (do_pick alone
is ~7s of moving) - for that whole stretch the 3 windows just showed their
last frame with no repaint (imshow needs a following waitKey to actually pump
the GUI and redraw - see check_emergency()), so the live view visibly froze on
every click ("클릭을 하면 화면이 다 멈추는데"), including exactly the moment
(gripper closing) the user most wants to watch live on the wrist cam. Fixed by
pulling the "read cameras + detect + draw HUD + imshow" block that used to
live only in the main loop out into render_frames(), and calling it once per
interpolation step inside move_smoothly() too (not just once per main-loop
tick) - the arm now redraws all 3 windows at the same ~30-step/duration_sec
cadence it already uses for interpolation, so the video keeps playing live
through the whole pick/place sequence instead of freezing. current_target_center/
current_j6/joints are now set as module globals by render_frames() (used to be
locals of the main loop) since move_smoothly() itself needs to trigger a
refresh, not just the main loop.

2026-08-26 update 5: per explicit user direction, HSV auto-detection removed
from TASK EXECUTION entirely (still used for [V]'s pixel snapshot during
calibration, and for the live on-screen bounding-box overlay - just not for
deciding what to pick/place any more). Also motivated by two things found the
same day: (1) this gripper+4cm-cube combo makes the J6 encoder verification
fundamentally unable to tell "centered grip" from "corner catch" (both read
~1757 regardless of torque 50%/75%/100%), so no fully-automatic grasp-success
gate was trustworthy anyway; (2) a real move_smoothly() stall-abort during
the ordinary hover-in move exposed how fragile automatic multi-step
"search/verify/retry" logic gets once every step is speculative. Replaced
with direct human-in-the-loop targeting: click the Astra RGB window, the
arm goes there and attempts a pick (do_pick) - no automatic success
judgment, the person looks at the result (especially the wrist-cam window)
and decides what to click next: the bin (do_place, closes the loop) if it
looks grasped, or the cube again (another do_pick attempt) if not. [P]
(fixed taught bin pose) is gone - place target is now whatever gets clicked,
same as pick, both going through the same predict_joints_from_pixel(u, v).
Also added in the same pass: a real stall/collision check in move_smoothly()
(this reference-derived design had NONE before - a bad height prediction
would just keep pushing into the table) - commanded-vs-actual servo lag
past STALL_THRESHOLD_TICKS for STALL_CONSECUTIVE checks aborts the move in
place; enabled only for the descend steps (do_pick's step 2, do_place's
step 2) where table contact is the real risk, not for ordinary large hover/
transport moves (an early cut at STALL_THRESHOLD_TICKS=150 false-tripped on
those - normal servo catch-up lag under a big fast joint change measured
279 ticks with nothing actually blocking it; raised to 400).

2026-08-27 update 7: root-caused "그리퍼가 닫히긴 하는데 꽉 안 물림" - two
stacked findings, not one. (a) `Torque_Limit` (RAM, addr 48 - the only thing
this script had ever written) was capped underneath by `Max_Torque_Limit`
(EEPROM, addr 16), which was still 500 (50%) from some earlier, unrelated job -
explains why torque 50%/75%/100% all produced identical results earlier
(already hitting that 50% ceiling regardless). Fixed by also writing
Max_Torque_Limit=1000 at startup (EEPROM write needs Lock=0 first, then
Lock=1 again - see ADDR_STS_LOCK). (b) Even after that fix, direct register
polling (Present_Load/Present_Current, see scratchpad/grip_force_diag*.py)
showed contact does reach ~90% load/high current - but the servo's own
overload protection then kicks in ~2-4s later and either throttles torque
down hard or cuts it to zero (gripper springs back open) - a real safety
feature of the servo itself, deliberately NOT bypassed (risk of actually
damaging the motor, which the user explicitly flagged). The old sequence's
slow gripper-close speed (400, shared with every other joint) plus a fixed
1.2s post-close dwell meant contact alone ate ~2.1-2.7s, leaving almost no
protected-window time for the lift step. Fixed by giving the gripper its own
faster close speed (1000 - picked conservatively, not maxed, again per the
user's damage concern) and replacing the fixed dwell with real contact
detection (close_gripper_with_contact_detect() - polls Present_Position until
it stops changing, then returns immediately) so the lift starts as soon as
contact happens instead of after a fixed guess.

2026-08-27 update 8: this follower arm is shared with other people, not
dedicated to this project - added check_port_not_busy() (lsof on PORT) before
ever opening it, so this script refuses to start (rather than fighting for
the port) if something else already has it open. Doesn't defend against every
possible case (e.g. someone else's software that doesn't hold the port open
the same way), but covers the direct "two of my own/similar scripts collide"
case cheaply. The `finally:` block already released camera/torque/port on any
exit path (including Ctrl-C) before this - worth remembering to stop this
script with an interrupt (not a hard kill) so that keeps happening, and to
not leave it running idle when not actively testing.

2026-08-27 update 9: "Astra Depth" window switched from the driver's default
JET (rainbow) colormap to grayscale (near=bright, far=dark, no-return=black),
per a reference video the user shared. First cut normalized against
DEPTH_MIN_MM/DEPTH_MAX_MM (the cube-detection filter range) and the user
caught a real bug from a screenshot - real depth data outside that narrow
band was being blacked out too, since that range was tuned for filtering
detections, not for display. Fixed to auto-scale per-frame to whatever valid
depth is actually visible (render_grayscale_depth()) - DEPTH_MIN_MM/MAX_MM
stay untouched for their original detection-filter purpose. Also added a
hover-to-read-raw-mm readout on that window (depth_hover_pixel) so "does this
region actually have depth data" can be checked directly instead of guessed
from how dark a pixel looks.

2026-08-27 update 10: getting an actual real grasp+lift working took several
more real-click iterations on top of update 7's Max_Torque_Limit fix, each
diagnosed from real log telemetry - see close_gripper_with_contact_detect()
and do_pick()'s docstrings/inline comments for the blow-by-blow (a
false-early contact-detect from position-stability with no minimum wait, then
from a position-stability signal that didn't actually mean force was
applied, then a lift step that relaxed the grip target instead of holding
it, then a final fix backing off from the extreme close target to a gentler
hold position right after contact so the servo's own overload protection
doesn't trip mid-lift). **User-confirmed working**: "잡았어, 놓치지 않고 잘
들어올렸어" - first fully successful real grasp+lift with this script.

2026-08-27 update 11: added a visual-only bin detection overlay
(find_bin_bbox(), reusing task_red_cube_to_bin/config.py's already-tuned
black-color thresholds) - draws a box on the Astra RGB window like the cube's,
so the bin's location is visible before clicking. Purely a visual aid: clicks
still target the literal clicked pixel (do_place), not auto-snapped to the
detected box - if detection is off, the person can still just click where
the bin actually is.

2026-08-27 update 12: real-use feedback after update 11 kept surfacing more
issues, each fixed in place (see inline comments at each site rather than
repeating here): calib_matrix.json's 10 taught points included one real
outlier (wrist_roll=438 vs. every other point's ~1000-1700) dragging fit RMSE
from 97.5 to 149.8 ticks - removed and refit, saved back to disk. do_place's
transport-to-bin also had the exact same "gripper target relaxes mid-move"
bug do_pick's lift had (update 10) - hardcoded 1750 instead of the actual
held position - fixed the same way (`carry_grip`), plus added an explicit
"rise to a safe height first" step (safe_high_j) since going straight from
the just-lifted pose to the bin's hover pose swept too low.

2026-08-27 update 13: two more real-use fixes. (a) Descending to the exact
taught "touch" height let the gripper rest on the table while trying to
close - the descent's stall_check apparently doesn't reliably catch flat,
straight-down surface contact (same blind spot already documented in this
project's MuJoCo track - see project memory). Added GRASP_HEIGHT_MARGIN, a
small conservative height buffer so the descent stops just above the taught
touch point instead of exactly at it. (b) Releasing into the bin used to
descend close to table height before opening - per explicit user request,
now releases from a mid-height (BIN_RELEASE_HEIGHT_OFFSET, an approximate
~15-20cm-above-bin equivalent - no real mm calibration exists in this raw-
servo design, so this is a starting guess to tune live) instead of lowering
all the way in.

2026-08-27 update 14: added an optional voice-command layer on top of the
already-working click flow, at the user's request ("llm을 추가해서 '빨간
큐브를 잡아서 쓰레기통에 넣어줘' 라는 걸 수행할 수 있어?"). Deliberately
does NOT touch calibration, grasp mechanics, or the do_pick/do_place
functions themselves - it only replaces "where did the click come from" with
"what did the LLM decide from the spoken command + the ALREADY-detected
cube/bin pixel positions (current_target_center/current_bin_center, both
existed before this update)". [L] starts a fixed-duration recording
(arecord, see VOICE_MIC_DEVICE - this machine's Astra S built-in mic, per
`arecord -l`; may need updating if device enumeration changes), transcribed
via Google's free speech API (`speech_recognition`'s recognize_google,
language=ko-KR) in a background thread, then sent to an LLM with a system
prompt describing the two available actions and the current detection
state, asking for a JSON action-plan array only. **Threading note**: the
background thread does `arecord`/STT/the API call ONLY - it never touches
cv2 (OpenCV's GUI calls aren't safe to call from multiple threads at once).
It just sets `voice_pending_plan`; the MAIN loop (same thread as every other
cv2 call) notices it next tick and actually runs do_pick/do_place, exactly
like click_pixel already works.

2026-08-27 update 15: the LLM call started on Claude (`claude-opus-5`), but
switched to Gemini (`gemini-3.5-flash-lite`, see GEMINI_MODEL) per explicit
user direction after confirming Claude Pro's subscription doesn't cover API
usage (separate billing) while Google's API has a genuine no-billing-required
free tier - better fit for a feature this the size of "parse one sentence
into two coordinates." Uses `google-genai` (`client.models.generate_content()`
+ `GenerateContentConfig(system_instruction=..., response_mime_type=
"application/json")`, response `.text`) - the docs' own headline pattern
(`client.interactions.create()`) was skipped because its Python signature is
a loosely-typed `**body: Any` passthrough that can't be verified offline,
whereas `generate_content`'s types (config fields, response `.text` property)
were confirmed directly against the installed package before writing this.
Needs `GEMINI_API_KEY` or `GOOGLE_API_KEY` set in the environment - the user
provided a key via a local `~/.gemini_key.env` file (never pasted into chat,
`chmod 600`) sourced right before launch; **confirmed working live** with a
real spoken command end to end. `pip`-equivalent deps (`google-genai`,
`SpeechRecognition`) were added via `uv pip install --python .venv/bin/python3
...` (ad hoc, not added to this project's own pyproject.toml/uv.lock since
they're this one script's optional feature, not a core project dependency).

2026-08-27 update 18: replaced the single global affine `matrix_M` with a
per-prediction local fit. Motivated by directly measured accuracy, not
guesswork: leave-one-out cross-validation (the only honest way to compare -
in-sample error always looks better with more parameters even when it's
pure overfitting) on this session's calib points showed the global affine
model works, but a k-nearest-neighbors local affine refit (predict using
only the LOCAL_FIT_K closest calib points, not all of them) does noticeably
better - 153.6 -> 134.8 ticks RMSE, about a 12% real improvement. A tempting
alternative (add quadratic terms `u^2, v^2, uv` to the SAME global fit, no
new data needed) was tried and rejected the same way: it looked much better
in-sample (100.1 -> 49.5) but was actually *worse* under LOOCV (153.6 ->
192.5) - classic overfitting with only 12 points and 6 free parameters per
joint, blowing up badly on one edge point (597-tick LOOCV error). See
predict_joints_from_pixel()'s and report_calib_accuracy()'s own comments.
[M] no longer fits-and-saves a matrix (there's no single matrix any more) -
it now just prints the current LOOCV accuracy estimate on demand, purely
diagnostic.

Keys:
  [G] teach gripper "holding" encoder value (cube in jaws, closed on it)
  [E] teach gripper "empty" encoder value (jaws closed on nothing)
  [H] teach home/idle pose (current joints) - do_place returns here when done
  [V] step 1 of a calibration point: snapshot the cube's CURRENT Astra
      pixel - press this BEFORE moving the gripper in, while the arm is
      still clear of the cube (see 2026-08-26 update 4 below for why this
      is now two keys instead of one).
  [C] step 2: move the gripper (torque off, [R]) to hover/touch the cube,
      then press C - combines the pixel [V] captured with the CURRENT joint
      positions into one calibration pair. Requires a pending [V] snapshot.
      Repeat at >=6 well-spread positions across the workspace before
      fitting (more points = more robust fit; the reference's own matrix is
      a 3-parameter-per-joint affine fit, so 3 points is the bare minimum
      and 4-5+ is what this project's own homography calibration needed in
      practice - see project memory).
  [M] (update 18: repurposed) print a leave-one-out cross-validated accuracy
      estimate for the current calib points - saves nothing, purely
      diagnostic. There's no single matrix to "fit and save" any more - see
      predict_joints_from_pixel()'s own comment.
  [X] clear all calibration points (start over)
  Click (Astra RGB window) - go pick at that pixel, or place there if
      already holding something (alternates every click; see update 5).
  [R] release torque (free the arm by hand - use this before teaching points)
  [L] voice command (records VOICE_RECORD_SECONDS, sends transcript + current
      cube/bin detection state to Gemini, runs whatever pick/place plan it
      returns - see updates 14/15 above)
  [ESC] emergency stop | [Q] quit

Run: uv run python3 custom_scripts/vision_pick_place/red_cube_calib_pick.py
(from ~/lerobot - needs the main venv for scservo_sdk/cv2/numpy, already
installed there. No YOLO/torch dependency any more - see the 2026-08-26
update above.)
"""

import json
import os
import subprocess
import tempfile
import threading
import time

import cv2
import numpy as np
import speech_recognition as sr
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from scservo_sdk import COMM_SUCCESS, PacketHandler, PortHandler

from orbbec_color_camera import ThreadedOrbbecRGBDCamera

# --- 설정 (이 컴퓨터/이 로봇에 맞게 조정됨) -----------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2026-08-26: 팔로워암 자체 USB 어댑터 보드가 죽어서 리더암 보드로 대체 중 -
# udev가 시리얼 번호로 심볼릭링크를 걸기 때문에 이 상태에서는
# /dev/so101_follower가 아니라 /dev/ttyACM0로 잡힘 (project memory 참고).
# 보드 교체되면 원래대로 되돌릴 것.
PORT = "/dev/ttyACM0"
BAUDRATE = 1_000_000  # lerobot의 feetech 드라이버 DEFAULT_BAUDRATE와 동일 (src/lerobot/motors/feetech/feetech.py)
PROTOCOL_VERSION = 0  # sts3215의 MODEL_PROTOCOL과 동일 (src/lerobot/motors/feetech/tables.py)
WRIST_CAM_INDEX = 4  # 손목캠("USB 2.0 PC Cam", 090c:b371)의 실제 캡처 노드 - v4l2-ctl --list-devices로 확인
# 카메라 공장 기본값 - 조명 조건에 따라 과다노출/저노출 둘 다 겪어봤으므로
# (project memory 참고) 필요하면 이 값을 즉석에서 조정할 것.
WRIST_V4L2_CTRLS = {"brightness": 113, "gamma": 2}
FPS = 20
MAX_RETRIES = 3

# Astra 깊이로 HSV 오탐(작업대 위가 아닌 배경의 빨간 물체 등)을 걸러내기 위한
# 타당한 범위(mm) - ThreadedOrbbecRGBDCamera 자체 기본 시각화 범위(350~800)와
# 동일하게 맞춤. 실측 후 필요하면 조정.
DEPTH_MIN_MM = 350
DEPTH_MAX_MM = 800


def apply_v4l2_ctrls(cam_index: int, ctrls: dict) -> None:
    for name, val in ctrls.items():
        subprocess.run(
            ["v4l2-ctl", "-d", f"/dev/video{cam_index}", "--set-ctrl", f"{name}={val}"],
            check=False, capture_output=True,
        )

# lerobot의 so_follower.py MOTORS 테이블과 동일한 서보 ID 매핑 (1~6) - 임의로
# 정한 게 아니라 이 로봇의 실제 배선과 일치하는 값.
ADDR_STS_TORQUE_ENABLE = 40
ADDR_STS_GOAL_ACC = 41
ADDR_STS_GOAL_POSITION = 42
ADDR_STS_GOAL_SPEED = 46
ADDR_STS_TORQUE_LIMIT = 48  # lerobot의 feetech tables.py STS_SMS_SERIES_CONTROL_TABLE과 동일 (0~1000 = 0~100%)
ADDR_STS_PRESENT_POSITION = 56
# 2026-08-27 update 7: 그리퍼가 "닫히긴 하는데 꽉 안 물린다"는 문제의 실제
# 원인을 찾음 - Torque_Limit(48, RAM)을 1000(100%)까지 올려도 그리퍼가
# 물리적으로 더 세게 안 잡혔던 건, EEPROM에 있는 상위 제한값
# Max_Torque_Limit(16)이 500(50%)으로 남아있어서 RAM 값을 그 이상 올려도
# 조용히 그 이하로 눌렸기 때문 (직접 레지스터를 읽어 확인함 - Lock=1,
# Max_Torque_Limit=500, Torque_Limit=1000이었는데 실제 파지력은 50%대로 계속
# 똑같았음 - 이전에 Torque_Limit을 50/75/100%로 바꿔도 J6 파지 결과가 항상
# 똑같이 나왔던 것도 이걸로 설명됨: 이미 이 상위 캡에 막혀 있었던 것).
# EEPROM 쓰기는 Lock(55)을 0으로 풀어야 되고, 쓴 뒤 다시 1로 잠가야 함.
ADDR_STS_MAX_TORQUE_LIMIT = 16  # EEPROM, 2바이트 - Torque_Limit(RAM)의 실질 상한
ADDR_STS_LOCK = 55  # EEPROM 쓰기 잠금: 0=해제(쓰기 가능), 1=잠금
ADDR_STS_PRESENT_LOAD = 60
ADDR_STS_PRESENT_CURRENT = 69

# 2026-08-27 update 7 (계속): Max_Torque_Limit을 고쳐도 여전히 "닫히긴 하는데
# 꽉 안 물림"이 재현돼서, 서보 레지스터(Present_Load/Present_Current)를 직접
# 실시간으로 찍어봄(스크래치패드의 grip_force_diag*.py) - 접촉 시 Load~900대/
# Current~350~400까지 올라가긴 하는데, 그 상태가 약 2~4초 넘게 지속되면 서보
# 자체 과부하/과전류 보호가 걸려서 힘이 확 빠지거나(부분 스로틀, Load~200대)
# 아예 토크가 꺼져버림(완전 차단, Load/Current 0 - 이 경우 그리퍼가 스프링백
# 으로 다시 열림). 둘 다 서보 자신을 보호하는 정상 동작이라 레지스터로
# 우회/무력화하지 않음(모터 손상 위험) - 대신 실측해보니 기존 코드는 그리퍼
# 닫는 속도(Goal_Speed=400, 전체 서보 공통값)가 느려서 "닫기 시작 -> 큐브에
# 실제로 닿기"까지만 2.1~2.7초가 걸렸고, 거기에 고정 1.2초 대기까지 더해서
# 들어올리기(step 4)를 시작할 때쯤엔 이미 저 보호 구간에 절반 넘게 들어가
# 있었음. 그리퍼 닫는 속도만 별도로 올리면(400 -> 1000, 완전히 빠르게 하면
# 충격이 커질까봐 보수적으로 2.5배만) 접촉까지 약 1.1초로 줄고, 접촉을
# 실시간으로 감지해서(위치가 더 안 변하는 시점) 곧장 리턴하므로 불필요한
# 대기 없이 남은 고출력 구간을 최대한 들어올리기에 쓸 수 있음.
GRIPPER_CLOSE_SPEED = 1000  # 그리퍼 전용 닫기 속도 - 다른 관절의 400보다 빠름

# 2026-08-27 update 10: 접촉을 확인한 뒤에도 계속 target_close_cmd(아주
# 공격적인, 물리적으로 도달 불가능한 목표)로 계속 밀면 들어올리는 내내 서보가
# 최대 전류를 계속 끄는 상태가 됨 - 실제 테스트에서 서보 LED가 깜빡이며
# (과부하 알람) 들어올리다가 놓치는 게 재현됨. 접촉 순간엔 확실히 물기 위해
# 세게 밀지만, 일단 접촉이 확인되면 실제로 멈춘 위치에서 이만큼만 더
# 파고드는 완화된 목표로 낮춰서 - 쥐는 힘은 유지하되(여전히 위치 오차가 있어
# 어느 정도 토크는 계속 걸림) 지속적인 최대 전류 상태를 피함. 서보 보호
# 자체를 우회하지 않고, 애초에 그 상태에 오래 머물지 않게 하는 접근.
GRIP_HOLD_MARGIN_TICKS = 50

# 2026-08-27 update 13: 사용자가 "그리퍼가 테이블에 닿아서 큐브를 집고 있어서
# 완벽하게 못잡는다"고 지적 - 하강 stall_check(400틱)이 이걸 못 잡아내는
# 것으로 보임: 평평한 바닥에 수직으로 눌리는 접촉은 각 관절이 개별적으로는
# 명령을 잘 따라가서(한쪽만 막히는 수평 장애물과 달리) 위치 오차가 크게 안
# 쌓이는 경우가 많음 - MuJoCo 시뮬레이션 트랙에서도 동일한 사각지대를 이미
# 확인한 적 있음(project memory 참고: "straight-down pressure against a flat
# surface apparently doesn't reliably trip the per-joint stall check"). stall
# 감지에만 기대는 대신, 하강 목표 자체를 보정점이 가르쳐준 "터치" 높이보다
# 살짝 위에서 멈추도록 안전 마진을 둠 - (shoulder_lift, elbow_flex) 오프셋,
# 값이 클수록 더 높이서 멈춤. 정확한 mm 환산은 없으니(이 프로젝트는 의도적으로
# 완전한 기구학 대신 직접 가르친 매핑을 씀) 작게 시작 - 실제로 보고 조정 필요.
# 2026-08-28: 사용자가 실제로 보고 "조금 더 내려가야 큐브를 잡는다"고 확인 -
# 마진을 줄여서 원래 가르친 터치 높이에 더 가깝게 감.
# 2026-08-28 update: (10,5)도 아직 부족, "5cm 정도 더" 필요하다고 확인 -
# mm 환산이 없어 정확히 못 맞추지만, 지난 단계와 비슷한 폭으로 한 번 더
# 낮춤 - 이번엔 원래 가르친 터치 높이(margin=0)를 넘어 음수로, 살짝 더 파고듦.
GRASP_HEIGHT_MARGIN = (-10, -5)

# update 13 (계속): 쓰레기통에 넣을 때 완전히 안까지 내려가서 여는 대신,
# 사용자 요청대로 쓰레기통 위 15~20cm 정도(근사치, 위와 같은 이유로 정확한
# cm 환산 없음)에서 그리퍼를 열어 떨어뜨리는 방식으로 변경 - 아래 do_place의
# hover(-250/-150, do_pick의 lift와 동일)보다는 낮고, 예전의 "살포시 하강"
# 오프셋(-65/-35, 테이블 근처)보다는 높은 중간 지점.
BIN_RELEASE_HEIGHT_OFFSET = (150, 80)

# 2026-08-26: 원래 500(50%, 이전 어떤 작업에서 EEPROM에 남아있던 값 - 이
# 스크립트가 설정한 적은 없었음) -> 750(75%)로 한 번 올렸다가, 그래도 파지가
# 확실치 않아 사용자 요청으로 100%까지 올림. 이 서보/큐브 조합에서는 최대
# 토크가 필요한 것으로 판단됨.
GRIPPER_TORQUE_LIMIT = 1000

CALIB_FILE = os.path.join(SCRIPT_DIR, "calib_matrix.json")

# 2026-08-27 update 14: 음성 명령 -> LLM 해석 -> pick/place 실행. 위 module
# docstring "update 14" 설명 참고.
# 2026-08-27 update 15: Claude API는 Pro 구독과 별개로 과금되는 걸 확인한 뒤
# (Claude Pro 구독에 API 사용량이 포함 안 됨), 사용자 요청으로 무료 등급이
# 있는 Gemini API로 교체 - google-genai SDK, client.models.generate_content()
# 사용 (문서가 안내하는 client.interactions.create()는 이 SDK 버전에서
# 타입이 느슨하게 열려있어(**body: Any) 오프라인으로 필드명을 검증할 수
# 없었음 - generate_content 쪽은 GenerateContentConfig/응답 .text 프로퍼티까지
# 실제 설치된 패키지에서 타입으로 확인 후 채택).
GEMINI_MODEL = "gemini-3.5-flash-lite"  # "가장 빠르고 비용 효율적인 3.5 모델" - 이 정도 단순 구조화 출력엔 충분
VOICE_MIC_DEVICE = "plughw:3,0"  # Astra S 내장 마이크 - `arecord -l`의 "card 3: S [ASTRA S]" 기준. 장치 순서 바뀌면 갱신 필요
VOICE_RECORD_SECONDS = 4
VOICE_STT_LANG = "ko-KR"

# HSV 빨간색 임계값 - task_red_cube_to_bin/config.py에서 이미 실측 튜닝된 값
# 그대로 재사용 (빨강은 hue가 0 근처에서 순환하므로 두 구간을 합침).
LOWER_RED_1, UPPER_RED_1 = (0, 60, 25), (10, 255, 255)
LOWER_RED_2, UPPER_RED_2 = (170, 60, 25), (180, 255, 255)
MIN_CUBE_CONTOUR_AREA = 200
MAX_CUBE_AREA_FRAC = 0.5  # 프레임 전체 대비 - 손/팔 등 큰 오탐 배제
MIN_CUBE_SOLIDITY = 0.65  # area/convexHullArea - 손가락처럼 오목한 모양 배제
CUBE_ASPECT_RANGE = (0.4, 2.5)

# 2026-08-27 update 11: 쓰레기통도 큐브처럼 화면에 박스로 표시해달라는 요청 -
# 검정색 임계값/면적/solidity/종횡비 값은 마찬가지로 task_red_cube_to_bin/
# config.py에서 이미 실측 튜닝된 값(LOWER_BLACK/UPPER_BLACK/MIN_BIN_CONTOUR_AREA/
# MAX_BIN_AREA_FRAC/MIN_BIN_SOLIDITY/BIN_ASPECT_RANGE) 그대로 재사용. 단,
# 이건 순전히 "어디 있는지 화면에 보여주는" 참고용 오버레이일 뿐 - 클릭
# 동작 자체는 원래대로 클릭한 픽셀을 그대로 목표로 씀(update 5), 탐지된
# 박스 중심으로 자동 스냅하지 않음 - 탐지가 틀렸을 때 사람이 여전히 직접
# 보정해서 클릭할 수 있어야 하므로.
LOWER_BLACK, UPPER_BLACK = (0, 0, 0), (180, 90, 70)
MIN_BIN_CONTOUR_AREA = 800
MAX_BIN_AREA_FRAC = 0.7
MIN_BIN_SOLIDITY = 0.6
BIN_ASPECT_RANGE = (0.3, 3.0)


def _find_color_bbox(bgr_frame, hsv_ranges, min_area, max_area_frac, min_solidity, aspect_range, kernel_size=5):
    """HSV 마스크(하나 이상의 범위를 합침) -> contour -> (면적/solidity/종횡비)
    필터 -> 가장 그럴듯한 후보 하나. 손/팔/케이블처럼 크지만 solidity 낮은
    오탐을 배제 - 이 프로젝트 perception.py의 검증된 방식 그대로. 큐브(빨강,
    두 구간)와 쓰레기통(검정, 한 구간) 둘 다 이 함수 하나로 처리 - 반환:
    (x1,y1,x2,y2) 또는 None."""
    hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
    mask = None
    for lower, upper in hsv_ranges:
        m = cv2.inRange(hsv, np.array(lower), np.array(upper))
        mask = m if mask is None else cv2.bitwise_or(mask, m)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    frame_area = bgr_frame.shape[0] * bgr_frame.shape[1]
    best, best_area = None, 0.0
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > frame_area * max_area_frac:
            continue
        hull_area = cv2.contourArea(cv2.convexHull(c))
        if hull_area <= 0 or (area / hull_area) < min_solidity:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if h == 0 or not (aspect_range[0] <= (w / h) <= aspect_range[1]):
            continue
        if area > best_area:
            best, best_area = (x, y, x + w, y + h), area
    return best


def find_red_cube_bbox(bgr_frame):
    return _find_color_bbox(
        bgr_frame, [(LOWER_RED_1, UPPER_RED_1), (LOWER_RED_2, UPPER_RED_2)],
        MIN_CUBE_CONTOUR_AREA, MAX_CUBE_AREA_FRAC, MIN_CUBE_SOLIDITY, CUBE_ASPECT_RANGE, kernel_size=5,
    )


def find_bin_bbox(bgr_frame):
    return _find_color_bbox(
        bgr_frame, [(LOWER_BLACK, UPPER_BLACK)],
        MIN_BIN_CONTOUR_AREA, MAX_BIN_AREA_FRAC, MIN_BIN_SOLIDITY, BIN_ASPECT_RANGE, kernel_size=7,
    )


def find_red_cube_bbox_depth_checked(color_bgr, depth_mm):
    """find_red_cube_bbox의 결과를 Astra 깊이로 한 번 더 검증 - 박스 영역의
    중앙값 깊이가 DEPTH_MIN_MM~DEPTH_MAX_MM(작업대 범위) 밖이면 배경의 다른
    빨간 물체로 보고 버림. depth_mm이 color_bgr보다 해상도가 낮을 수 있어
    (ThreadedOrbbecRGBDCamera 기본 색상 640x480 vs 깊이 320x240) 좌표를
    스케일링 - task_red_cube_to_bin/perception.py의 estimate_cube_height_m과
    동일한 방식."""
    box = find_red_cube_bbox(color_bgr)
    if box is None or depth_mm is None:
        return box
    x1, y1, x2, y2 = box
    sx, sy = depth_mm.shape[1] / color_bgr.shape[1], depth_mm.shape[0] / color_bgr.shape[0]
    dx1, dy1 = max(0, int(x1 * sx)), max(0, int(y1 * sy))
    dx2, dy2 = min(depth_mm.shape[1], int(x2 * sx)), min(depth_mm.shape[0], int(y2 * sy))
    if dx2 <= dx1 or dy2 <= dy1:
        return box  # 스케일링 결과가 비정상이면 깊이 검증 없이 그냥 통과시킴
    region = depth_mm[dy1:dy2, dx1:dx2]
    valid = region[region > 0]
    if valid.size < 5:
        return box  # 유효 깊이 픽셀이 너무 적음 - 판단 보류, 색상 결과만으로 통과
    median_depth = float(np.median(valid))
    if not (DEPTH_MIN_MM <= median_depth <= DEPTH_MAX_MM):
        return None  # 작업대 범위 밖 - 배경의 다른 빨간 물체로 판단, 오탐 처리
    return box


def render_grayscale_depth(depth_mm):
    """update 9: "Astra Depth" 창을 driver(orbbec_color_camera.py)가 주는
    기본 무지개(JET) 컬러맵 대신, 사용자가 참고 영상(Orbbec Astra S 데모)에서
    본 것과 같은 흑백 스타일로 직접 렌더링 - 가까울수록 밝은 흰색, 멀수록
    어두운 회색, 깊이값이 아예 없는(0, no-return) 픽셀만 완전한 검정.

    처음엔 DEPTH_MIN_MM~DEPTH_MAX_MM(350~800mm, 큐브 탐지용으로 좁게 잡은
    작업대 범위)로 정규화했었는데, 이건 "화면에 뭐가 보이는지"가 아니라
    "큐브 탐지 후보를 거를 범위"를 위한 값이라 - 실측해보니 실제로 깊이값이
    멀쩡히 있는 물체(예: 그 범위보다 살짝 가깝거나 먼 곳)까지 화면에서 검게
    지워져버리는 문제가 있었음 (사용자가 직접 스크린샷으로 지적). 탐지용
    범위와 화면 표시용 범위는 다른 목적이라 섞으면 안 됐던 것 - 이제
    프레임마다 실제 유효(0이 아닌) 깊이값들의 min~max로 자동 스케일해서,
    거리에 상관없이 "지금 보이는 모든 실측 깊이"가 그대로 흑백 대비로
    나오게 함. DEPTH_MIN_MM/DEPTH_MAX_MM 자체는 원래 목적(큐브 탐지 필터,
    find_red_cube_bbox_depth_checked)에만 그대로 씀 - 안 건드림."""
    if depth_mm is None:
        return None
    valid = depth_mm > 0
    gray = np.zeros(depth_mm.shape, dtype=np.uint8)
    if not np.any(valid):
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    vmin = float(depth_mm[valid].min())
    vmax = float(depth_mm[valid].max())
    if vmax <= vmin:
        gray[valid] = 255  # 유효 픽셀이 전부 같은 거리 - 구분할 대비가 없으니 밝게
    else:
        clipped = np.clip(depth_mm, vmin, vmax).astype(np.float32)
        # 가까울수록(값이 작을수록) 밝게 - 정규화 후 반전
        norm = ((vmax - clipped) / (vmax - vmin) * 255.0).astype(np.uint8)
        gray[valid] = norm[valid]
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# 2026-08-27 update 8: 이 SO-101 팔로워암은 여러 사람이 같이 쓰는 공용 장비 -
# 리눅스는 시리얼 포트를 배타적으로 잠가주지 않아서(같은 /dev/ttyACM0를 여러
# 프로세스가 동시에 열 수 있음), 이미 누가 쓰고 있는데 이 스크립트를 또
# 실행하면 같은 물리 시리얼 라인에 명령이 겹쳐 써져서 양쪽 다 오동작할 수
# 있음 - OS가 안 막아주니 직접 확인해야 함. lsof로 포트를 이미 쓰고 있는
# 프로세스가 있는지 먼저 확인하고, 있으면 뺏지 않고 그냥 실행을 거부함.
def check_port_not_busy(port: str) -> None:
    try:
        result = subprocess.run(["lsof", "-t", port], capture_output=True, text=True, timeout=3)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return  # lsof 자체가 없거나 타임아웃이면 확인 없이 진행 (기존 동작 유지)
    pids = [p for p in result.stdout.split() if p.strip() and int(p) != os.getpid()]
    if pids:
        print(f"[중단] {port}를 다른 프로세스(PID {', '.join(pids)})가 이미 사용 중입니다.")
        print("       다른 사람이 팔로워암을 쓰고 있을 수 있어 겹쳐서 실행하지 않습니다.")
        print("       (이 프로세스가 내가 예전에 띄운 좀비 프로세스인 게 확실하면 직접 종료 후 재시도)")
        raise SystemExit(1)


check_port_not_busy(PORT)

# --- 서보 모터 초기화 ------------------------------------------------------
portHandler = PortHandler(PORT)
packetHandler = PacketHandler(PROTOCOL_VERSION)

if not portHandler.openPort() or not portHandler.setBaudRate(BAUDRATE):
    print(f"[오류] {PORT} 포트 열기 실패")
    raise SystemExit(1)

torque_state = False
emergency_stop = False


def set_torque(enable=True):
    global torque_state
    val = 1 if enable else 0
    torque_state = enable
    for sid in range(1, 7):
        packetHandler.write1ByteTxRx(portHandler, sid, ADDR_STS_TORQUE_ENABLE, val)


def read_all_positions():
    positions = []
    for sid in range(1, 7):
        pos, comm, error = packetHandler.read2ByteTxRx(portHandler, sid, ADDR_STS_PRESENT_POSITION)
        positions.append(pos if comm == COMM_SUCCESS else 0)
    return positions


# --- 카메라 초기화 (Astra S RGB+Depth - 고정 탐지/보정용, 손목캠 - 파지 직전
# 확인용. 위 docstring "update 3" 참고) --------------------------------------
astra = ThreadedOrbbecRGBDCamera(width=640, height=480, fps=30)
if not astra.isOpened():
    print("[오류] Astra S 카메라를 열 수 없습니다 - 다른 프로세스가 점유 중인지 확인 필요")
    raise SystemExit(1)

apply_v4l2_ctrls(WRIST_CAM_INDEX, WRIST_V4L2_CTRLS)
wrist_cap = cv2.VideoCapture(WRIST_CAM_INDEX)
if not wrist_cap.isOpened():
    print(f"[오류] 손목캠(인덱스 {WRIST_CAM_INDEX})을 열 수 없습니다 - v4l2-ctl --list-devices로 확인 필요")
    raise SystemExit(1)
wrist_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
wrist_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

set_torque(False)

# update 7: RAM Torque_Limit뿐 아니라 EEPROM Max_Torque_Limit도 같이 올려야
# 실제로 반영됨 (위 ADDR_STS_MAX_TORQUE_LIMIT 정의부 설명 참고). EEPROM 쓰기라
# Lock을 풀었다가 다시 잠가야 함.
packetHandler.write1ByteTxRx(portHandler, 6, ADDR_STS_LOCK, 0)
packetHandler.write2ByteTxRx(portHandler, 6, ADDR_STS_MAX_TORQUE_LIMIT, GRIPPER_TORQUE_LIMIT)
packetHandler.write1ByteTxRx(portHandler, 6, ADDR_STS_LOCK, 1)
packetHandler.write2ByteTxRx(portHandler, 6, ADDR_STS_TORQUE_LIMIT, GRIPPER_TORQUE_LIMIT)
print(f"[*] 그리퍼 토크 제한: {GRIPPER_TORQUE_LIMIT / 10:.0f}% (RAM Torque_Limit + EEPROM Max_Torque_Limit 둘 다)")

# [V]/[C]로 모은 보정점 - calib_matrix.json에 매번 즉시 저장되므로(아래
# save_calib 호출 참고) 코드 수정 때문에 스크립트를 재시작해도 안 없어짐.
# 처음엔 빈 리스트, 곧이어 파일에서 있으면 이어서 불러옴.
calib_pts_pixel = []
calib_pts_joints = []
home_pose = None
# update 18: 전역 matrix_M 하나를 미리 계산해 저장해두는 방식을 버리고,
# predict_joints_from_pixel()이 클릭마다 가장 가까운 점들로 즉석 지역 적합을
# 하는 방식으로 바뀜 - 위 predict_joints_from_pixel 설명 참고. LOCAL_FIT_K는
# leave-one-out 교차검증으로 실측 비교해서 고른 값(K=3~6 중 5가 최선).
LOCAL_FIT_K = 5

# 그리퍼 파지 판별 기준값 (원본 파일의 캔 기준값 - 이 로봇/이 큐브로 [G]/[E]
# 다시 티칭하기 전까지의 자리표시자일 뿐, 그대로 믿으면 안 됨)
gripper_holding_val = 2340
gripper_empty_val = 1750
grasp_threshold = 2050

if os.path.exists(CALIB_FILE):
    try:
        with open(CALIB_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            # update 18: matrix_M을 더 이상 안 씀 - 파일에 예전 버전이 남아있어도
            # 그냥 무시(읽지도 않음). calib_points만 있으면 충분.
            home_pose = data.get("home_pose", None)
            gripper_holding_val = data.get("gripper_holding_val", gripper_holding_val)
            gripper_empty_val = data.get("gripper_empty_val", gripper_empty_val)
            grasp_threshold = data.get("grasp_threshold", int((gripper_holding_val + gripper_empty_val) / 2))
            saved_pts = data.get("calib_points", [])
            for pt in saved_pts:
                calib_pts_pixel.append(tuple(pt["pixel"]))
                calib_pts_joints.append(pt["joints"])
            print(f"[★ 설정 로드 완료] 파지판별 기준값: J6 >= {grasp_threshold}, 보정점 {len(calib_pts_pixel)}개 이어받음")
    except Exception as e:
        print(f"[안내] 로드 오류: {e}")


def save_calib(**updates) -> None:
    save_data = {}
    if os.path.exists(CALIB_FILE):
        with open(CALIB_FILE, "r", encoding="utf-8") as f:
            save_data = json.load(f)
    save_data.update(updates)
    with open(CALIB_FILE, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2)


print("\n" + "=" * 70)
print("  SO-101 빨간 큐브 -> 쓰레기통  |  픽셀->관절 직접 매핑 + 클릭 파지")
print("  [보정]")
print("    - 그리퍼에 큐브를 물려놓고                 -> [ G ] (Holding 등록)")
print("    - 큐브 빼고 그리퍼를 끝까지 닫은 상태에서   -> [ E ] (Empty 등록)")
print("    - 원하는 대기 자세에 두고                   -> [ H ] (Home 등록)")
print("    - 보정점(2단계):")
print("      1) 팔이 큐브를 안 가린 상태에서            -> [ V ] (픽셀 스냅샷)")
print("      2) 토크 해제([R]) 후 그리퍼를 큐브로 옮기고 -> [ C ] (보정점 완성)")
print("      (최소 3개, 권장 6개 이상, 작업공간 여러 지점에 고르게)")
print("    - 보정점을 다 모았으면                      -> [ M ] (행렬 계산+저장)")
print("    - 보정점을 전부 지우고 새로 하려면            -> [ X ] (초기화)")
print("      (보정점은 파일에 계속 저장되니 재시작해도 안 없어짐)")
print("  [실행 - HSV 자동탐지 없음, 직접 클릭]")
print("    - Astra RGB 창에서 큐브를 클릭             -> 그 위치로 이동해서 파지 시도")
print("    - (화면 보고 실제로 잡혔는지 직접 확인)")
print("    - 잡혔으면 쓰레기통 위치를 클릭              -> 그 위치로 이동해서 놓음")
print("    - 안 잡혔으면 큐브 위치를 다시 클릭          -> 처음부터 재시도")
print("    - [ L ]   : 음성 명령 (4초 녹음 -> Gemini 해석 -> 자동 pick/place)")
print("    - [ESC]   : 비상정지 | [ R ]: 토크 해제 | [ Q ]: 종료")
print("=" * 70 + "\n")

current_target_center = None
current_bin_center = None  # update 14: 음성/LLM 명령이 참조하는 쓰레기통 탐지 위치
current_j6 = 0
joints = [0, 0, 0, 0, 0, 0]
pending_pixel = None  # [V]로 스냅샷한, [C] 완성을 기다리는 픽셀 (update 4)


def check_emergency():
    global emergency_stop
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        emergency_stop = True
        print("\n[!!! 비상정지 발동 !!!] 즉시 정지합니다.")
        cur = read_all_positions()
        for sid, p in enumerate(cur, start=1):
            packetHandler.write2ByteTxRx(portHandler, sid, ADDR_STS_GOAL_POSITION, p)
        return True
    return False


# 2026-08-26: 이 스크립트엔 원래 충돌/멈춤 감지가 전혀 없었음 - matrix_M
# 예측이 살짝 낮게 나오면(실측: 테이블에 닿는 문제 발생) 그냥 계속 테이블을
# 밀어붙이게 되어 있었음. 명령한 위치와 실제 위치 차이(lag)가 계속 크면
# 뭔가에 막힌 것으로 보고 그 자리에서 멈춤 - task_red_cube_to_bin 쪽
# 모듈형 설계에서 이미 검증된 것과 같은 방식(там stall_check)을 이 raw
# scservo_sdk 버전에 맞게 이식.
# 2026-08-26: 150으로 시작했다가 완전히 정상적인(장애물 없는) hover 이동에서도
# lag=279틱이 실측돼서 400으로 올림 - 정상적인 서보 추종 지연과 진짜 막힘을
# 확실히 구분할 수 있는 여유를 둠.
# 2026-08-28: 재조립 이후 정밀 하강에서 lag=699로 막힘 감지 - 진짜 충돌인지,
# 관절이 뻑뻑해져서 서보 추종만 느려진 건지 구분이 안 됨. 재보정 전에 먼저
# 시험 삼아 700으로 올렸는데도 lag=866으로 또 막힘 - 700에서 1000으로 더 올림.
# 사용자가 직접 보면서 "조금 더 내려가야 한다"고 확인했으므로, 진짜 단단한
# 충돌이 아니라 뻑뻑해진 관절 저항으로 보고 더 밀어붙여봄.
# 2026-08-28 update: 1000에서도 lag=1063으로 또 걸림 (699→866→1063, 올릴 때마다
# lag도 같이 늘어나는 패턴 - 단단한 벽이라기보다 계속 밀리는 저항으로 보임).
# GRASP_HEIGHT_MARGIN을 더 낮췄으니(더 멀리 내려가야 함) 같이 올림.
STALL_THRESHOLD_TICKS = 1300
STALL_CHECK_EVERY = 5  # 스텝
STALL_CONSECUTIVE = 3  # 연속으로 이만큼 밀리면 진짜 막힌 것으로 판단


def render_frames():
    """Astra RGB / Depth / 손목캠 3개 창을 한 번 갱신. 원래 메인 루프에만
    있던 "카메라 읽기 -> 탐지 -> HUD 그리기 -> imshow" 블록을 그대로 뽑아낸
    것 - move_smoothly()도 스텝마다 이걸 불러서 이동 중에도 라이브 화면이
    끊기지 않게 함 (2026-08-27 update 6, 위 docstring 참고). current_target_center/
    current_j6/joints는 이제 이 함수가 갱신하는 모듈 전역값 - [G]/[E]/[H]/[C]
    키 핸들러가 메인 루프에서 그대로 참조함."""
    global current_target_center, current_bin_center, current_j6, joints
    joints = read_all_positions()
    current_j6 = joints[5]

    ret, frame, depth_vis = astra.read()
    if ret and frame is not None:
        depth_mm = astra.read_raw_depth_mm()
        box = find_red_cube_bbox_depth_checked(frame, depth_mm)
        current_target_center = None
        if box is not None:
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            current_target_center = (cx, cy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
            cv2.putText(frame, f"[TARGET] red_cube ({cx},{cy})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # update 11: 쓰레기통도 참고용으로 화면에 박스 표시 - 위 find_bin_bbox
        # 설명 참고. 클릭 좌표를 여기로 스냅하지 않음 - 순전히 "여기쯤 있다"를
        # 보여주는 시각적 참고용, 실제 목표는 여전히 클릭한 그 픽셀 그대로.
        bin_box = find_bin_bbox(frame)
        current_bin_center = None
        if bin_box is not None:
            bx1, by1, bx2, by2 = bin_box
            bcx, bcy = (bx1 + bx2) // 2, (by1 + by2) // 2
            current_bin_center = (bcx, bcy)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 200, 0), 3)
            cv2.circle(frame, (bcx, bcy), 6, (255, 200, 0), -1)
            cv2.putText(frame, f"[BIN] ({bcx},{bcy})", (bx1, by1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

        hud_info1 = f"GripJ6: {current_j6} (Hold:{gripper_holding_val} | Empty:{gripper_empty_val})"
        hud_info2 = "[G]Hold [E]Empty [H]Home [V]Snap [C]CalibPt [M]CheckAccuracy [X]Clear [R]Torque0 [L]Voice [Q]Quit | Click=Pick/Place"
        hud_info3 = (
            f"CalibPts: {len(calib_pts_pixel)}  |  ready: {has_enough_calib_points()}"
            f"  |  pending[V]: {pending_pixel}  |  holding: {holding_cube}"
        )
        cv2.putText(frame, hud_info1, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
        cv2.putText(frame, hud_info2, (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.putText(frame, hud_info3, (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)

        cv2.imshow(WINDOW_NAME, frame)
        gray_depth = render_grayscale_depth(depth_mm)
        if gray_depth is not None:
            if depth_hover_pixel is not None and depth_mm is not None:
                hx, hy = depth_hover_pixel
                if 0 <= hy < depth_mm.shape[0] and 0 <= hx < depth_mm.shape[1]:
                    raw_val = int(depth_mm[hy, hx])
                    cv2.circle(gray_depth, (hx, hy), 4, (0, 0, 255), 1)
                    label = f"{raw_val}mm" if raw_val > 0 else "no depth(0)"
                    cv2.putText(gray_depth, label, (min(hx + 8, gray_depth.shape[1] - 90), max(hy - 8, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            cv2.imshow(DEPTH_WINDOW_NAME, gray_depth)

    ret_w, wrist_frame = wrist_cap.read()
    if ret_w and wrist_frame is not None:
        wbox = find_red_cube_bbox(wrist_frame)
        if wbox is not None:
            wx1, wy1, wx2, wy2 = wbox
            cv2.rectangle(wrist_frame, (wx1, wy1), (wx2, wy2), (0, 255, 0), 3)
        cv2.imshow("Wrist Cam (pre-grasp confirmation)", wrist_frame)


def move_smoothly(target_pose, duration_sec=3.0, stall_check=False):
    # 2026-08-26: stall_check 기본값을 False로 바꿈 - 처음엔 모든 이동에
    # 켜뒀다가 실측해보니 hover 이동(큰 거리를 빠르게 움직이는 정상적인
    # 상황)에서 서보가 명령을 따라잡는 지연만으로도 오탐(lag=279틱)이 발생,
    # 큐브 근처도 못 가고 멈춰버림. 테이블 접촉이 실제로 우려되는 하강
    # 단계(아래 "2. 큐브 위치로 정밀 하강" 호출)에서만 stall_check=True로
    # 명시적으로 켬 - 나머지는 원래대로 막힘 감지 없이 진행.
    global emergency_stop
    if emergency_stop:
        return False
    current_pose = read_all_positions()
    steps = int(duration_sec * 30)
    dt = duration_sec / steps
    stall_count = 0
    for step in range(1, steps + 1):
        step_start = time.time()
        if check_emergency():
            return False
        alpha = (1.0 - np.cos(step / float(steps) * np.pi)) / 2.0
        interp = [int(current_pose[sid - 1] + alpha * (cmd_val - current_pose[sid - 1]))
                  for sid, cmd_val in enumerate(target_pose, start=1)]
        for sid, cmd in enumerate(interp, start=1):
            packetHandler.write2ByteTxRx(portHandler, sid, ADDR_STS_GOAL_POSITION, cmd)

        render_frames()  # update 6: 이동 중에도 라이브 화면 계속 갱신

        if stall_check and step % STALL_CHECK_EVERY == 0:
            actual = joints  # render_frames()가 방금 읽은 현재 관절값 재사용
            lag = max(abs(a - c) for a, c in zip(actual, interp))
            if lag > STALL_THRESHOLD_TICKS:
                stall_count += 1
            else:
                stall_count = 0
            if stall_count >= STALL_CONSECUTIVE:
                for sid, p in enumerate(actual, start=1):
                    packetHandler.write2ByteTxRx(portHandler, sid, ADDR_STS_GOAL_POSITION, p)
                print(f"  [!] 이동 중 막힘 감지 (lag={lag}틱) - 그 자리에서 멈춤 (테이블/장애물 접촉 가능성)")
                return False

        elapsed = time.time() - step_start
        time.sleep(max(0, dt - elapsed))
    return True


def has_enough_calib_points():
    return len(calib_pts_pixel) >= 3


def predict_joints_from_pixel(u, v):
    """update 18: 전역 affine 행렬(matrix_M) 하나로 화면 전체를 커버하는
    대신, 클릭한 픽셀에서 가장 가까운 LOCAL_FIT_K개 보정점만 골라 그
    자리에서 지역 affine을 즉석으로 다시 적합 - leave-one-out 교차검증으로
    정직하게 비교한 결과(스크래치패드에서 실측): 전역 156틱 -> 지역(K=5)
    135틱, 약 12% 개선. 2차(quadratic) 항을 추가하는 방법도 시도해봤는데
    점이 12개뿐이라 과적합만 심해져서(학습 데이터엔 더 잘 맞아 보이는데
    LOOCV로는 오히려 153.6->192.5로 더 나빠짐) 기각 - 데이터가 훨씬
    많아지기 전까진 시도 안 하는 게 맞음. 점이 LOCAL_FIT_K개보다 적으면
    있는 점 전부를 씀(그래도 최소 3개는 있어야 affine 자체가 성립)."""
    if not has_enough_calib_points():
        return None
    pixels_arr = np.array(calib_pts_pixel, dtype=np.float64)
    joints_arr = np.array(calib_pts_joints, dtype=np.float64)
    k = min(LOCAL_FIT_K, len(pixels_arr))
    dists = np.sqrt(((pixels_arr - np.array([u, v])) ** 2).sum(axis=1))
    idx = np.argsort(dists)[:k]
    A = np.hstack([pixels_arr[idx], np.ones((k, 1))])
    M, _, rank, _ = np.linalg.lstsq(A, joints_arr[idx], rcond=None)  # M: (3, 6)
    pred_j = np.array([float(u), float(v), 1.0]) @ M
    return [int(max(500, min(3500, val))) for val in pred_j]


def report_calib_accuracy():
    """update 18: [M] 키 재활용 - 예전엔 "행렬을 계산해서 저장"하는
    키였는데, 이제 예측이 매번 즉석으로(위 predict_joints_from_pixel)
    이뤄져서 미리 계산해 저장할 단일 행렬 자체가 없음. 대신 이 키는 지금
    가진 보정점들로 leave-one-out 교차검증을 돌려서 실제 예상 오차를
    보고만 함 - 아무것도 저장하지 않음, 순수 진단용."""
    n = len(calib_pts_pixel)
    if n < 4:
        print(f"[안내] 보정점이 {n}개뿐입니다 - 교차검증에는 최소 4개 필요합니다.")
        return
    pixels_arr = np.array(calib_pts_pixel, dtype=np.float64)
    joints_arr = np.array(calib_pts_joints, dtype=np.float64)
    k = min(LOCAL_FIT_K, n - 1)
    errs = []
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        train_px, train_j = pixels_arr[mask], joints_arr[mask]
        dists = np.sqrt(((train_px - pixels_arr[i]) ** 2).sum(axis=1))
        idx = np.argsort(dists)[:k]
        A = np.hstack([train_px[idx], np.ones((k, 1))])
        M, _, _, _ = np.linalg.lstsq(A, train_j[idx], rcond=None)
        pred = np.append(pixels_arr[i], 1.0) @ M
        errs.append(pred - joints_arr[i])
    rmse = float(np.sqrt(np.mean(np.array(errs) ** 2)))
    print(f"[교차검증] 보정점 {n}개, KNN(K={k}) 지역 적합 기준 예상 오차(LOOCV RMSE) = {rmse:.1f} ticks")


def verify_gripper_grasp():
    """자동 판정 없이 참고용 로그만 - update 5(2026-08-26): HSV 자동탐지를
    태스크에서 완전히 빼고 사람이 화면 클릭으로 직접 목표를 지정하는 방식으로
    바뀌면서(아래 do_pick/do_place, on_mouse_click 참고), "제대로 잡혔는지"도
    자동 판정(J6 임계값 - 이 그리퍼+4cm 큐브 조합에서 이미 신뢰 못 한다고
    확인됨 - 또는 Astra 재탐지) 대신 사람이 화면을 보고 직접 판단하게 함.
    이 함수는 그 판단에 참고할 J6 엔코더 값만 찍어줌."""
    time.sleep(0.4)
    j6_pos = read_all_positions()[5]
    print(f"  -> [참고] 현재 J6 엔코더: {j6_pos} (Empty~Hold 기준: {gripper_empty_val}~{gripper_holding_val})")


def close_gripper_with_contact_detect(target_cmd, max_wait_sec=2.0, min_wait_sec=0.25,
                                       load_threshold=350):
    """update 7: 고정 시간만큼 무작정 기다리는 대신, 실제로 힘이 걸린
    시점을 감지해서 곧장 리턴 - 위 GRIPPER_CLOSE_SPEED 설명 참고. 서보 자체
    과부하 보호가 접촉 후 약 2~4초 안에 힘을 빼버리는 걸 실측했기 때문에,
    그 짧은 고출력 구간을 "그냥 대기"에 낭비하지 않고 곧바로 들어올리기로
    넘어가기 위함.

    2026-08-27 update 7 버그 수정 (두 번째): 처음엔 "위치가 더 이상 안
    바뀜"(still_ticks/still_checks)을 접촉 신호로 썼는데, 실제 클릭 테스트
    로그에 "접촉 감지: 1.23초 (Load=184, Current=1)"이 찍힘 - diag 스크립트로
    실측한 진짜 접촉 시점의 Load(900대)/Current(350~400대)에 한참 못 미치는
    값. 즉 살짝 스쳐서 속도만 잠깐 줄어든 순간(진짜 힘이 걸리기 훨씬 전)을
    "접촉"으로 오판하고 있었던 것 - 위치 정지는 힘이 실리기 전에도 일시적으로
    일어날 수 있어 신뢰할 수 있는 신호가 아니었음. Present_Load를 직접 보고
    실제로 문턱값을 넘긴 시점을 접촉으로 판단하도록 바꿈.

    update 7 버그 수정 (세 번째): load_threshold=450/max_wait_sec=1.2로 바꾼
    다음 실제 재시도에서도 "1.22초 (Load=136)"으로 또 타임아웃 - 그런데
    사용자가 육안으로는 "큐브에 살짝 걸린 채로 올라가다 놓쳤다"고 확인함,
    즉 약하지만 진짜 접촉은 있었다는 뜻. diag의 깨끗한 단독 테스트(약
    1.0~1.1초에 로드가 900대까지 올라감)와 달리 실제 클릭 위치마다 예측
    높이가 조금씩 달라 접촉 타이밍에 변동이 있고, 1.2초는 그 변동을 감당하기
    빠듯했던 것으로 보임. max_wait_sec을 2.0초로 늘리고(그래도 과부하 보호가
    걸리기 시작하는 약 2~4.5초보다는 충분히 여유 있음) load_threshold를
    350으로 낮춰서(두 번의 실측 실패값 136/184보다는 확실히 위, 900대보다는
    한참 아래) 약한 접촉도 더 일찍/더 안정적으로 잡되 노이즈와는 구분되게 함.
    큐브 없이 허공을 닫는 경우처럼 로드가 끝까지 안 오르면 max_wait_sec
    타임아웃으로 그냥 진행(무한정 기다리지 않음) - 이 경우 로그에 낮은 Load
    값이 남아 원인 파악에 참고가 됨."""
    packetHandler.write2ByteTxRx(portHandler, 6, ADDR_STS_GOAL_SPEED, GRIPPER_CLOSE_SPEED)
    packetHandler.write2ByteTxRx(portHandler, 6, ADDR_STS_GOAL_POSITION, target_cmd)
    t0 = time.time()
    load, cur = 0, 0
    contact_ok = False
    while time.time() - t0 < max_wait_sec:
        render_frames()  # 대기 중에도 화면 계속 갱신 (update 6과 동일한 이유)
        elapsed = time.time() - t0
        load, comm, _ = packetHandler.read2ByteTxRx(portHandler, 6, ADDR_STS_PRESENT_LOAD)
        cur, _, _ = packetHandler.read2ByteTxRx(portHandler, 6, ADDR_STS_PRESENT_CURRENT)
        if elapsed >= min_wait_sec and comm == COMM_SUCCESS and load >= load_threshold:
            contact_ok = True
            break
        time.sleep(0.03)
    tag = "" if contact_ok else "  [!] 문턱값 못 넘김 - 약한 접촉이거나 허공을 닫았을 수 있음"
    print(f"     접촉 감지: {time.time() - t0:.2f}초 (Load={load}, Current={cur}){tag}")
    # 이후 이동(들어올리기 등)은 원래 속도(400)로 되돌려 놓음 - 그리퍼만
    # 빠른 속도는 닫는 순간에만 필요.
    packetHandler.write2ByteTxRx(portHandler, 6, ADDR_STS_GOAL_SPEED, 400)


def do_pick(target_u, target_v):
    """클릭 위치(Astra RGB 픽셀)로 이동해서 파지 시도. 성공했는지는 자동
    판정하지 않음 - 사람이 화면(특히 손목캠 창)을 보고 직접 확인한 뒤, 잡혔으면
    쓰레기통을 클릭, 안 잡혔으면 큐브를 다시 클릭하면 됨(on_mouse_click)."""
    grasp_j = predict_joints_from_pixel(target_u, target_v)
    if grasp_j is None:
        print("[오류] 보정 데이터가 없습니다 - [C]로 점을 모으고 [M]으로 행렬을 계산하세요.")
        return

    print("\n" + "=" * 55)
    print(f"  [*] 클릭 위치 ({target_u}, {target_v})로 파지 시작")
    print("=" * 55)

    set_torque(True)
    time.sleep(0.2)
    for sid in range(1, 7):
        packetHandler.write1ByteTxRx(portHandler, sid, ADDR_STS_GOAL_ACC, 30)
        packetHandler.write2ByteTxRx(portHandler, sid, ADDR_STS_GOAL_SPEED, 400)

    p1, p2, p3, p4, p5, p6 = grasp_j
    hover_j = [p1, int(max(500, p2 - 250)), int(max(500, p3 - 150)), p4, p5, 2500]
    print("  -> 1. 큐브 상공으로 접근...")
    if not move_smoothly(hover_j, duration_sec=3.0):
        return
    time.sleep(0.3)

    # update 13: 보정점이 가르쳐준 정확한 "터치" 높이 대신, 테이블 접촉을
    # 피하기 위해 살짝 위(GRASP_HEIGHT_MARGIN)에서 멈춤 - 위 상수 설명 참고.
    grasp_open = [p1, int(max(500, p2 - GRASP_HEIGHT_MARGIN[0])),
                  int(max(500, p3 - GRASP_HEIGHT_MARGIN[1])), p4, p5, 2500]
    print("  -> 2. 큐브 위치로 정밀 하강...")
    if not move_smoothly(grasp_open, duration_sec=3.5, stall_check=True):
        # 하강 중 막힘 감지 = 테이블이든 큐브든 뭔가에 닿았다는 뜻 - 완전
        # 실패가 아니라 "도착한 걸로 보고 그냥 닫아본다" 쪽이 맞음(예측
        # 높이가 살짝 낮아도 멈춘 자리가 대략 파지 지점에 가까움).
        print("     막힌 지점에서 그대로 파지를 시도합니다.")
    time.sleep(0.4)

    print("  -> 3. 그리퍼 닫기 (Grasp)...")
    target_close_cmd = int(min(1750, gripper_empty_val - 50))
    close_gripper_with_contact_detect(target_close_cmd)

    # update 10: 접촉 확인 직후, 실제로 멈춘 위치 기준 완화된 목표로 낮춤 -
    # target_close_cmd(아주 공격적인, 물리적으로 못 닿는 목표)를 들어올리는
    # 내내 그대로 유지하면 서보가 계속 최대 전류를 끄는 상태가 되어 과부하
    # 알람(LED 깜빡임)이 걸려 놓치는 게 실제로 재현됨 - 위 GRIP_HOLD_MARGIN_TICKS
    # 설명 참고.
    actual_close_pos, _, _ = packetHandler.read2ByteTxRx(portHandler, 6, ADDR_STS_PRESENT_POSITION)
    hold_target = max(target_close_cmd, actual_close_pos - GRIP_HOLD_MARGIN_TICKS)
    packetHandler.write2ByteTxRx(portHandler, 6, ADDR_STS_GOAL_POSITION, hold_target)

    lift_j = [p1, int(max(500, p2 - 250)), int(max(500, p3 - 150)), p4, p5, hold_target]
    print("  -> 4. 물건 들어올리기 (Lift)...")
    if not move_smoothly(lift_j, duration_sec=2.5):
        return

    verify_gripper_grasp()
    print("\n[*] 파지 시도 완료 - 화면(특히 손목캠)으로 직접 확인해주세요.")
    print("    잡혔으면 쓰레기통 위치를 클릭, 안 잡혔으면 큐브 위치를 다시 클릭하세요.\n")


def do_place(target_u, target_v):
    """이미 물건을 든 상태에서 클릭 위치(쓰레기통 등)로 이동해서 내려놓음."""
    place_j = predict_joints_from_pixel(target_u, target_v)
    if place_j is None:
        print("[오류] 보정 데이터가 없습니다 - [C]로 점을 모으고 [M]으로 행렬을 계산하세요.")
        return

    print("\n" + "=" * 55)
    print(f"  [*] 클릭 위치 ({target_u}, {target_v})로 이송")
    print("=" * 55)

    set_torque(True)
    p1, p2, p3, p4, p5, p6 = place_j

    # update 12: 두 가지 수정.
    # (1) 쥐는 힘 유지: 아래 hover/gentle 단계의 그리퍼 목표가 원래 고정값
    # 1750이었는데, do_pick에서 실제 정착한 파지 위치와 다를 수 있어
    # (update 10과 같은 문제) 옮기는 동안 힘이 풀릴 수 있었음 - 지금 실제
    # 그리퍼 위치(carry_grip)를 그대로 유지 목표로 씀.
    # (2) 안전 높이 확보: 사용자가 "쓰레기통에 넣을 때 높이가 너무 낮다"고
    # 지적 - do_pick이 든 자세에서 곧장 쓰레기통 쪽 hover 위치(원래 오프셋
    # -180/-100, do_pick의 lift 오프셋 -250/-150보다 얕았음)로 이동하다보니
    # 낮게 다녔던 것으로 보임. 이동 전 먼저 확실히 더 높이(현재 자세 기준,
    # xy/그리퍼는 그대로) 올라간 뒤 쓰레기통 쪽으로 이동하도록 단계 추가하고,
    # hover 오프셋도 do_pick의 lift와 같은 -250/-150으로 맞춤.
    current = read_all_positions()
    carry_grip = current[5]
    safe_high_j = [current[0], int(max(500, current[1] - 150)), int(max(500, current[2] - 80)),
                   current[3], current[4], carry_grip]
    print("  -> 0. 이동 전 안전 높이로 상승...")
    if not move_smoothly(safe_high_j, duration_sec=1.2):
        return
    time.sleep(0.2)

    hover_place_j = [p1, int(max(500, p2 - 250)), int(max(500, p3 - 150)), p4, p5, carry_grip]
    print("  -> 1. 목표 상공으로 안전 이동...")
    if not move_smoothly(hover_place_j, duration_sec=3.0):
        return
    time.sleep(0.3)

    # update 13: 완전히 안까지 내려가서 여는 대신, 쓰레기통 위 중간 높이
    # (BIN_RELEASE_HEIGHT_OFFSET, 위 설명 참고)까지만 내려가서 거기서 놓음 -
    # 사용자 요청: "15~20cm 위에서 그리퍼를 열어서 놓는 게 좋을거 같다."
    release_j = [p1, int(max(500, p2 - BIN_RELEASE_HEIGHT_OFFSET[0])),
                 int(max(500, p3 - BIN_RELEASE_HEIGHT_OFFSET[1])), p4, p5, carry_grip]
    print("  -> 2. 쓰레기통 위 놓는 높이로 하강...")
    if not move_smoothly(release_j, duration_sec=2.0, stall_check=True):
        print("     막힌 지점에서 그대로 놓습니다.")
    time.sleep(0.4)

    print("  -> 3. 그리퍼 열기 (Release)...")
    packetHandler.write2ByteTxRx(portHandler, 6, ADDR_STS_GOAL_POSITION, 2500)
    time.sleep(1.0)

    retract_j = [p1, int(max(500, p2 - 220)), int(max(500, p3 - 120)), p4, p5, 2500]
    print("  -> 4. 상공으로 안전 복귀...")
    move_smoothly(retract_j, duration_sec=2.0)
    time.sleep(0.3)

    target_home = home_pose if home_pose is not None else [2048, 2048, 2048, 2048, 2048, 2048]
    print("  -> 5. 홈 포지션으로 복귀...")
    move_smoothly(target_home, duration_sec=3.0)
    print("[*] 이송 완료! 다음 큐브를 클릭하면 다시 파지를 시작합니다.\n")


# update 5 (2026-08-26): 태스크 실행을 HSV 자동탐지에서 "Astra RGB 창을
# 클릭 -> 그 위치로 이동" 방식으로 전환. holding_cube는 자동 판정이 아니라
# 그냥 "마지막 동작이 pick이었는지 place였는지" 기록 - 클릭할 때마다 번갈아
# pick/place를 실행함. 사람이 화면 보고 실제로 잡혔는지 판단해서, 안
# 잡혔으면 큐브를 다시 클릭(그러면 do_pick이 다시 실행됨)하면 됨.
click_pixel = None
holding_cube = False
WINDOW_NAME = "SO-101 Red Cube Calibration & Pick (Astra RGB)"
DEPTH_WINDOW_NAME = "Astra Depth (reference only)"


def on_mouse_click(event, x, y, flags, param):
    global click_pixel
    if event == cv2.EVENT_LBUTTONDOWN:
        click_pixel = (x, y)


# update 9: 뎁스 창 위에 마우스를 올리면 그 픽셀의 원시 깊이값(mm)을 화면에
# 찍어줌 - "이 영역이 진짜 깊이값이 없는(0) 건지, 아니면 값은 있는데 그냥
# 어둡게 보이는 건지"를 직접 눈으로 확인하기 위함 (그레이스케일 렌더링만
# 봐서는 구분 안 되는 걸 사용자가 스크린샷으로 직접 지적함).
depth_hover_pixel = None


def on_depth_mouse_move(event, x, y, flags, param):
    global depth_hover_pixel
    if event == cv2.EVENT_MOUSEMOVE:
        depth_hover_pixel = (x, y)


cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, on_mouse_click)
cv2.namedWindow(DEPTH_WINDOW_NAME)
cv2.setMouseCallback(DEPTH_WINDOW_NAME, on_depth_mouse_move)


# update 14/15: 음성 명령 -> Gemini가 해석 -> pick/place 실행. 위 module
# docstring 설명 참고 - 백그라운드 스레드는 녹음/음성인식/LLM
# 호출만 하고 cv2는 절대 건드리지 않음(스레드 안전성). voice_pending_plan에
# 결과만 담아두면 메인 루프가 다음 틱에 알아서 집어서 실행함(click_pixel과
# 동일한 패턴).
voice_pending_plan = None
voice_busy = False
_genai_client = None  # 첫 사용 시점에 지연 생성 - API 키 없으면 여기서 에러 나는 게 자연스러움


def _get_genai_client():
    global _genai_client
    if _genai_client is None:
        _genai_client = genai.Client()  # GEMINI_API_KEY(우선순위 낮음) 또는 GOOGLE_API_KEY 환경변수 사용
    return _genai_client


def voice_command_worker():
    """[L]에서 스레드로 실행됨 - 녹음 -> 음성인식(한국어) -> Gemini 해석 ->
    voice_pending_plan에 결과 저장. 실패해도 예외가 스레드 밖으로 안 나가게
    전부 잡아서 사용자에게 보이는 메시지로 바꿔줌(백그라운드 스레드에서 발생한
    예외는 기본적으로 조용히 무시되니까)."""
    global voice_pending_plan, voice_busy
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name
        print(f"\n[음성] 녹음 중... ({VOICE_RECORD_SECONDS}초, 지금 말씀하세요)")
        rec = subprocess.run(
            ["arecord", "-D", VOICE_MIC_DEVICE, "-f", "cd", "-d", str(VOICE_RECORD_SECONDS), "-t", "wav", wav_path],
            capture_output=True, text=True, timeout=VOICE_RECORD_SECONDS + 5,
        )
        if rec.returncode != 0:
            print(f"[음성] 녹음 실패: {rec.stderr.strip()}")
            return

        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio, language=VOICE_STT_LANG)
        except sr.UnknownValueError:
            print("[음성] 알아듣지 못했습니다 - 다시 시도해주세요.")
            return
        except sr.RequestError as e:
            print(f"[음성] 음성인식 서비스 오류: {e}")
            return
        print(f"[음성] 인식된 문장: \"{text}\"")

        cube_status = f"픽셀 {current_target_center}에서 탐지됨" if current_target_center else "지금 탐지 안 됨"
        bin_status = f"픽셀 {current_bin_center}에서 탐지됨" if current_bin_center else "지금 탐지 안 됨"
        system_prompt = (
            "너는 SO-101 로봇팔 pick-and-place 시스템의 음성 명령 해석기다. "
            "가능한 행동은 두 가지뿐이다: "
            "'pick' - 지정한 픽셀 위치의 물체를 집는다. "
            "'place' - 이미 들고 있는 물체를 지정한 픽셀 위치에 놓는다. "
            f"현재 카메라 탐지 상태: 빨간 큐브는 {cube_status}, 쓰레기통은 {bin_status}. "
            "사용자의 한국어 음성 명령을 해석해서, 수행할 행동을 순서대로 담은 "
            "JSON 배열만 출력해라 - 다른 설명, 코드블록 표시, 텍스트는 절대 포함하지 마라. "
            '형식: [{"action": "pick", "pixel": [u, v]}, {"action": "place", "pixel": [u, v]}] '
            "명령이 가리키는 물체가 탐지 안 된 상태거나, 명령을 이 두 행동으로 표현할 수 없으면 "
            "빈 배열 []만 출력해라."
        )
        client = _get_genai_client()
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=text,
            config=genai_types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",  # JSON만 나오도록 강제 - 파싱 실패 여지 줄임
                max_output_tokens=1024,
            ),
        )
        reply_text = response.text or ""
        try:
            plan = json.loads(reply_text.strip())
        except json.JSONDecodeError:
            print(f"[음성] Gemini 응답을 JSON으로 해석 못함: {reply_text!r}")
            return

        if not plan:
            print("[음성] 실행할 동작이 없습니다 (명령을 못 알아들었거나 대상이 탐지 안 됨).")
            return
        print(f"[음성] 실행 계획: {plan}")
        voice_pending_plan = plan
    except FileNotFoundError:
        print("[음성] arecord를 찾을 수 없습니다 (alsa-utils 설치 필요).")
    except genai_errors.ClientError as e:
        print(f"[음성] Gemini API 인증/요청 오류 - GEMINI_API_KEY 환경변수를 확인해주세요: {e}")
    except genai_errors.APIError as e:
        print(f"[음성] Gemini API 오류: {e}")
    except Exception as e:  # noqa: BLE001 - 백그라운드 스레드라 여기서 반드시 잡아야 함
        print(f"[음성] 처리 중 오류: {e}")
    finally:
        voice_busy = False


try:
    while True:
        loop_start = time.time()
        # ThreadedOrbbecRGBDCamera.read()는 cv2.VideoCapture와 달리 논블로킹 -
        # 백그라운드 스레드가 아직 첫 프레임을 못 받아온 시작 직후 잠깐은
        # (ret=False, frame=None)이 정상이다 (그 클래스 자체 docstring 참고).
        # render_frames()는 그 경우 그냥 이번 틱은 그리기만 건너뛰고 계속
        # 진행함 - 카메라가 진짜 끊긴 것과 구분할 수 없으므로 계속 재시도.
        render_frames()

        # update 5: 클릭 하나 = pick/place 하나 (자동 HSV 탐지 없이 클릭한
        # 그 픽셀을 그대로 목표로 씀). 캘리브레이션([V]/[C]) 중에는 이 클릭이
        # 상관없는 pick 동작을 트리거하지 않도록, [V]로 픽셀을 스냅샷한
        # 상태(pending_pixel 있음)에서는 클릭을 무시.
        if click_pixel is not None:
            cu, cvv = click_pixel
            click_pixel = None
            if pending_pixel is not None:
                print("\n[안내] 지금 보정점 캡처 중(V 스냅샷 대기)이라 클릭은 무시됩니다 - [C]로 마저 완료하거나 계속 진행하세요.\n")
            elif not has_enough_calib_points():
                print("\n[경고] 보정 데이터가 없습니다 - [V]/[C]로 먼저 보정하세요 (최소 3개).\n")
            elif holding_cube:
                do_place(cu, cvv)
                holding_cube = False
            else:
                do_pick(cu, cvv)
                holding_cube = True

        # update 14: voice_command_worker()가 백그라운드 스레드에서 채워둔
        # 계획을 메인 루프(=cv2를 다루는 유일한 스레드)에서 실행 - click_pixel과
        # 완전히 동일한 패턴.
        if voice_pending_plan is not None:
            plan = voice_pending_plan
            voice_pending_plan = None
            for step in plan:
                action, pixel = step.get("action"), step.get("pixel")
                if not pixel or len(pixel) != 2:
                    print(f"[음성] 잘못된 스텝 무시: {step}")
                    continue
                pu, pv = int(pixel[0]), int(pixel[1])
                if not has_enough_calib_points():
                    print("\n[경고] 보정 데이터가 없습니다 - [V]/[C]로 먼저 보정하세요 (최소 3개).\n")
                elif action == "pick":
                    do_pick(pu, pv)
                    holding_cube = True
                elif action == "place":
                    do_place(pu, pv)
                    holding_cube = False
                else:
                    print(f"[음성] 알 수 없는 action 무시: {step}")

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            emergency_stop = True
            print("\n[!!! 비상정지 발동 !!!] 즉시 정지합니다.")
            cur = read_all_positions()
            for sid, p in enumerate(cur, start=1):
                packetHandler.write2ByteTxRx(portHandler, sid, ADDR_STS_GOAL_POSITION, p)

        elif key in (ord("g"), ord("G")):
            gripper_holding_val = current_j6
            grasp_threshold = int((gripper_holding_val + gripper_empty_val) / 2)
            save_calib(gripper_holding_val=gripper_holding_val, gripper_empty_val=gripper_empty_val,
                       grasp_threshold=grasp_threshold)
            print(f"\n[★ 큐브 물림 값 등록 완료] J6 Holding: {gripper_holding_val} -> 기준값(Threshold): {grasp_threshold}\n")

        elif key in (ord("e"), ord("E")):
            gripper_empty_val = current_j6
            grasp_threshold = int((gripper_holding_val + gripper_empty_val) / 2)
            save_calib(gripper_holding_val=gripper_holding_val, gripper_empty_val=gripper_empty_val,
                       grasp_threshold=grasp_threshold)
            print(f"\n[★ 빈손 닫힘 값 등록 완료] J6 Empty: {gripper_empty_val} -> 기준값(Threshold): {grasp_threshold}\n")

        elif key in (ord("h"), ord("H")):
            home_pose = list(joints)
            save_calib(home_pose=home_pose)
            print(f"\n[★ 홈 포지션 등록 완료] Home Pose: {home_pose}\n")

        elif key in (ord("v"), ord("V")):
            # 보정점 1단계 (update 4) - 팔이 큐브를 안 가리는 지금 이 순간의
            # 픽셀을 미리 찜해둠. 손이 잠깐 스쳐 지나가는 정도의 가림은 update 3의
            # 재시도 로직으로, 그리퍼 자체가 큐브 위에 있어서 못 보는 근본적인
            # 가림은 이 2단계 분리로 각각 대응.
            snap = current_target_center
            if snap is None:
                print("\n[안내] 지금 화면에서 목표물이 안 보입니다 - 손을 잠깐 비켜주세요 (최대 1.5초 재시도)...")
                for _ in range(15):
                    time.sleep(0.1)
                    ret_c, color_c, _ = astra.read()
                    if not ret_c or color_c is None:
                        continue
                    box_c = find_red_cube_bbox_depth_checked(color_c, astra.read_raw_depth_mm())
                    if box_c is not None:
                        bx1, by1, bx2, by2 = box_c
                        snap = ((bx1 + bx2) // 2, (by1 + by2) // 2)
                        break
            if snap is None:
                print("[안내] 여전히 목표물이 감지되지 않아 스냅샷할 수 없습니다.\n")
            else:
                pending_pixel = snap
                print(f"\n[★ 픽셀 스냅샷] pixel={pending_pixel} - 이제 그리퍼를 큐브로 옮기고 [C]를 눌러 완성하세요.\n")

        elif key in (ord("c"), ord("C")):
            # 보정점 2단계 - [V]로 미리 찜해둔 픽셀 + 지금 이 순간(그리퍼가
            # 실제로 큐브에 가 있는 상태)의 관절값을 짝지어 보정점 하나 완성.
            if pending_pixel is None:
                print("\n[안내] 먼저 [V]로 큐브 위치를 스냅샷해야 합니다 (팔이 큐브를 안 가린 상태에서).\n")
            else:
                calib_pts_pixel.append(pending_pixel)
                calib_pts_joints.append(list(joints))
                # 즉시 파일에 저장 - 코드 수정 때문에 스크립트를 재시작해도
                # 지금까지 모은 보정점이 안 없어지게.
                save_calib(calib_points=[
                    {"pixel": list(px), "joints": jt} for px, jt in zip(calib_pts_pixel, calib_pts_joints)
                ])
                print(f"\n[★ 보정점 #{len(calib_pts_pixel)} 추가] pixel={pending_pixel} joints={joints}\n")
                pending_pixel = None

        elif key in (ord("m"), ord("M")):
            # update 18: 더 이상 "행렬을 계산해서 저장"하는 키가 아님 -
            # report_calib_accuracy() 설명 참고. 순수 진단용, 아무것도 안 바뀜.
            report_calib_accuracy()

        elif key in (ord("x"), ord("X")):
            # 보정점이 이제 파일에 계속 남으니(위 [C] 핸들러 참고), 완전히
            # 새로 찍고 싶을 때를 위한 초기화 키. save_calib은 update-merge라
            # 파일을 직접 다시 써서 calib_points를 진짜로 없앰.
            calib_pts_pixel.clear()
            calib_pts_joints.clear()
            pending_pixel = None
            clean_data = {}
            if os.path.exists(CALIB_FILE):
                with open(CALIB_FILE, "r", encoding="utf-8") as f:
                    clean_data = json.load(f)
            clean_data.pop("calib_points", None)
            clean_data.pop("matrix_M", None)  # 예전 버전 파일에 남아있을 수 있는 잔재 정리
            with open(CALIB_FILE, "w", encoding="utf-8") as f:
                json.dump(clean_data, f, indent=2)
            print("\n[★ 보정점 전부 삭제됨] 처음부터 다시 [V]/[C]로 모아주세요.\n")

        elif key in (ord("r"), ord("R")):
            set_torque(False)
            emergency_stop = False
            print("[*] 토크 해제됨 (자유 모드).\n")

        elif key in (ord("l"), ord("L")):
            # update 14: 이미 녹음/처리 중이면 무시 - 스레드 두 개가 동시에
            # arecord/Gemini 호출하며 겹치지 않게.
            if voice_busy:
                print("\n[음성] 이미 처리 중입니다 - 잠시 기다려주세요.\n")
            else:
                voice_busy = True
                threading.Thread(target=voice_command_worker, daemon=True).start()

        elif key in (ord("q"), ord("Q")):
            break

        elapsed = time.time() - loop_start
        time.sleep(max(0, (1.0 / FPS) - elapsed))

finally:
    astra.release()
    wrist_cap.release()
    cv2.destroyAllWindows()
    set_torque(False)
    portHandler.closePort()
    print("[종료] 프로그램 정상 종료.")
