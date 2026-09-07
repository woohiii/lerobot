"""Bimanual click-to-grasp, IR-view variant of click_grasp_bimanual.py.

Same grasp/place/home logic (task_state_machine, gripper, kinematics) reused
unchanged. Only the display and click-target source differ: instead of the
Astra RGB window, this shows a 3-panel WRIST_LEFT | WRIST_RIGHT | ASTRA_IR
window. Only clicks landing in the ASTRA_IR panel pick a grasp target;
clicks on either wrist panel are ignored (they're for monitoring only).

# ponytail: homography.json was calibrated against Astra RGB pixels, not IR.
# IR and RGB are physically separate lenses on the Astra S (different
# baseline), so reusing the RGB homography for IR pixel coords carries a
# parallax offset - accepted per user's explicit choice (2026-09-07) over
# building a dedicated IR homography calibration, which is out of scope
# here. If click accuracy on IR turns out too far off in practice, the
# upgrade path is a calibrate_camera.py run against ASTRA_IR_FRAME_PATH to
# produce a separate ir_homography.json, passed as pixel_to_xy's
# `homography` arg instead of the default.

Needs camera_hub.py (both wrist cams) AND astra_s_ir_hub.py already running
and publishing - camera devices aren't opened here.

Run: uv run python3 custom_scripts/vision_pick_place/task_red_cube_to_bin_new_gripper/click_grasp_ir_wrist.py \\
    --left-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14029976-if00 --right-port /dev/so101_follower
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

import config
import gripper
import perception
import task_state_machine as tsm
from click_grasp_bimanual import handle_click
from kinematics import CollisionDetected, SOArm101
from lerobot.robots.bi_so_follower import BiSOFollower, BiSOFollowerConfig
from lerobot.robots.so_follower import SOFollowerConfig

WINDOW = "IR view: left-click=left arm, right-click=right arm - 'e'=e-stop, 'q'=quit"
WRIST_LEFT_FRAME_PATH = "/tmp/vsp_wrist_left.png"  # same value as config.LEFT_OVERRIDES["WRIST_FRAME_PATH"]
DISPLAY_H = 480  # common panel height for hconcat


def _panel(frame, label: str):
    """Resize to DISPLAY_H and label, for hconcat. None -> a black placeholder
    (camera not publishing yet) so the layout doesn't shift/crash."""
    if frame is None:
        vis = np.zeros((DISPLAY_H, 640, 3), dtype="uint8")
    else:
        h, w = frame.shape[:2]
        vis = cv2.resize(frame, (int(w * DISPLAY_H / h), DISPLAY_H))
    cv2.putText(vis, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return vis


def build_display(wrist_left, wrist_right, astra_ir):
    """Returns (combined_bgr, ir_panel_x_offset, ir_panel_native_frame) so a
    click on the combined image can be translated back to IR-native pixels."""
    panel_left = _panel(wrist_left, "WRIST LEFT")
    panel_right = _panel(wrist_right, "WRIST RIGHT")
    panel_ir = _panel(astra_ir, "ASTRA IR")
    ir_x_offset = panel_left.shape[1] + panel_right.shape[1]
    combined = cv2.hconcat([panel_left, panel_right, panel_ir])
    return combined, ir_x_offset, panel_ir.shape[1]


def click_to_ir_native(x: int, y: int, ir_x_offset: int, ir_panel_w: int, ir_native_frame) -> tuple[int, int] | None:
    """None if the click wasn't inside the IR panel, or the IR frame isn't
    currently available."""
    if ir_native_frame is None or not (ir_x_offset <= x < ir_x_offset + ir_panel_w):
        return None
    native_h, native_w = ir_native_frame.shape[:2]
    px = int((x - ir_x_offset) * native_w / ir_panel_w)
    py = int(y * native_h / DISPLAY_H)
    return px, py


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-port", required=True, help="왼쪽 팔 follower 포트")
    parser.add_argument("--right-port", required=True, help="오른쪽 팔 follower 포트")
    parser.add_argument("--allow-unverified-left", action="store_true",
                         help="왼팔 보정값이 미검증 placeholder여도 왼팔 클릭을 허용")
    parser.add_argument("--grasp-strategy", choices=["legacy", "il"], default="legacy",
                        help="양팔 모두에 적용할 파지 전략 (기본: %(default)r)")
    parser.add_argument("--policy-path", default=None, help="grasp-strategy=il일 때 필요한 학습된 ACT 체크포인트 경로")
    args = parser.parse_args()

    if args.grasp_strategy == "il" and not args.policy_path:
        parser.error("--grasp-strategy il requires --policy-path")

    bi_config = BiSOFollowerConfig(
        id="follower",
        left_arm_config=SOFollowerConfig(port=args.left_port, max_relative_target=config.MAX_RELATIVE_TARGET_DEG),
        right_arm_config=SOFollowerConfig(port=args.right_port, max_relative_target=config.MAX_RELATIVE_TARGET_DEG),
    )
    bi = BiSOFollower(bi_config)
    bi.connect(calibrate=False)
    print("[click_grasp_ir_wrist] 양팔 연결 성공")

    arm_left = SOArm101(robot=bi.left_arm)
    arm_right = SOArm101(robot=bi.right_arm)
    arm_left.connect()
    arm_right.connect()

    home_left = arm_left.get_joint_deg()
    home_xyz_left = tuple(arm_left.kin.forward_kinematics(home_left[: len(config.ARM_JOINTS)])[:3, 3])
    home_right = arm_right.get_joint_deg()
    home_xyz_right = tuple(arm_right.kin.forward_kinematics(home_right[: len(config.ARM_JOINTS)])[:3, 3])

    wrist_left_cap = perception.PublishedFrameSource(WRIST_LEFT_FRAME_PATH)
    wrist_right_cap = perception.PublishedFrameSource(config.WRIST_FRAME_PATH)
    astra_ir_cap = perception.PublishedFrameSource(config.ASTRA_IR_FRAME_PATH)

    click_queue: list[tuple[str, int, int]] = []
    layout = {"ir_x_offset": 0, "ir_panel_w": 0, "ir_native": None}

    def on_mouse(event, x, y, flags, userdata):
        if event not in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            return
        native = click_to_ir_native(x, y, layout["ir_x_offset"], layout["ir_panel_w"], layout["ir_native"])
        if native is None:
            print("[click_grasp_ir_wrist] 손목캠 화면은 파지 목표 클릭용이 아닙니다 - ASTRA IR 패널을 클릭하세요")
            return
        side = "left" if event == cv2.EVENT_LBUTTONDOWN else "right"
        click_queue.append((side, *native))

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)

    # ponytail: `return` inside a try still runs `finally` in Python - an
    # emergency stop must NOT be followed by the finally block's home-return
    # move, so track it explicitly and gate the home-return on this flag
    # rather than trying to short-circuit out of the try/finally.
    emergency_stopped = False
    try:
        while True:
            _, wl = wrist_left_cap.read()
            _, wr = wrist_right_cap.read()
            ok_ir, ir = astra_ir_cap.read()
            combined, ir_x_offset, ir_panel_w = build_display(wl, wr, ir)
            layout["ir_x_offset"] = ir_x_offset
            layout["ir_panel_w"] = ir_panel_w
            layout["ir_native"] = ir if ok_ir else None
            cv2.imshow(WINDOW, combined)

            # Popping and handling a click here blocks the video feed for the
            # duration of the grasp attempt - same already-blocking design as
            # click_grasp_bimanual.py, accepted tradeoff, not fixed here.
            if click_queue:
                side, x, y = click_queue.pop(0)
                handle_click(side, x, y, arm_left, arm_right, args.allow_unverified_left,
                             home_xyz_left, home_xyz_right, args.grasp_strategy, args.policy_path, astra_ir_cap)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("e"):
                arm_left.release_torque()
                arm_right.release_torque()
                print("[click_grasp_ir_wrist] 비상 정지 - 양팔 토크 해제됨")
                emergency_stopped = True
                break
            if key in (ord("q"), 27):
                break
    finally:
        cv2.destroyAllWindows()
        if emergency_stopped:
            print("[click_grasp_ir_wrist] 비상 정지 상태 유지 - 홈 복귀 시도하지 않음 (양팔 토크 해제된 채로 둠)")
        else:
            print(f"[click_grasp_ir_wrist] 왼팔 홈 복귀: {home_xyz_left}")
            try:
                arm_left.move_to_xyz_converge(home_xyz_left, tolerance_m=0.015, max_iters=20)
            except CollisionDetected as e:
                print(f"[click_grasp_ir_wrist] 왼팔 홈 복귀 중 충돌 감지: {e}")
            except Exception as e:
                print(f"[click_grasp_ir_wrist] 왼팔 홈 복귀 실패: {e}")
            print(f"[click_grasp_ir_wrist] 오른팔 홈 복귀: {home_xyz_right}")
            try:
                arm_right.move_to_xyz_converge(home_xyz_right, tolerance_m=0.015, max_iters=20)
            except CollisionDetected as e:
                print(f"[click_grasp_ir_wrist] 오른팔 홈 복귀 중 충돌 감지: {e}")
            except Exception as e:
                print(f"[click_grasp_ir_wrist] 오른팔 홈 복귀 실패: {e}")
        bi.disconnect()


if __name__ == "__main__":
    main()
