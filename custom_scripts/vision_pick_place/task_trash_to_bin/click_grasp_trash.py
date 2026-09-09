"""Single-arm trash-to-bin click grasp, 2026-09-08.

Modeled on ../task_red_cube_to_bin_new_gripper/click_grasp_bimanual.py, but
for the calibrated right arm only. Targeting is color-agnostic and click-only:
there is no generic object detector, so no wrist-camera fine_servo is used.
The --dry-hover flag is a Stage-2 verification path that the original does
not have; it reaches the clicked hover pose and returns home without grasping.

Run: uv run python3 custom_scripts/vision_pick_place/task_trash_to_bin/click_grasp_trash.py \\
    --port /dev/so101_follower
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

import config
import gripper
import il_grasp_skill
import perception
import task_state_machine as tsm
from kinematics import CollisionDetected, SOArm101
from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

WINDOW = "click=grasp target - 'e'=e-stop, 'q'=quit"


def handle_click(
    x: float,
    y: float,
    arm: SOArm101,
    home_xyz,
    grasp_strategy: str = "legacy",
    policy_path: str | None = None,
    cap=None,
    dry_hover: bool = False,
    source_click_px: tuple[float, float] | None = None,
) -> None:
    xy = perception.pixel_to_xy(x, y)
    if xy is None:
        print("[click_grasp_trash] 호모그래피 없음 (homography.json 확인 필요)")
        return
    if not perception.is_xy_within_safe_workspace(*xy):
        print(f"[click_grasp_trash] ({xy[0]:.3f}, {xy[1]:.3f})는 보정된 안전 작업영역 밖입니다 - 이동 취소")
        return

    target = (xy[0], xy[1], config.SEARCH_HOVER_XYZ[2])
    print(f"[click_grasp_trash] 클릭 위치로 이동 ({xy[0]:.3f}, {xy[1]:.3f})")
    try:
        arm.move_to_xyz_converge(target, tolerance_m=0.015, max_iters=20)
    except CollisionDetected as e:
        print(f"[click_grasp_trash] 이동 중 충돌 감지 ({e}) - 취소")
        return

    if dry_hover:
        print("[click_grasp_trash] dry-hover reached, returning home, no grasp attempted")
        try:
            arm.move_to_xyz_converge(home_xyz, tolerance_m=0.015, max_iters=20)
        except CollisionDetected as e:
            print(f"[click_grasp_trash] 홈 복귀 중 충돌 감지: {e}")
        return

    # There is no generic per-object detector for arbitrary clicked trash, so
    # free-space hover remains IK and the legacy path skips wrist-cam fine_servo.
    grasped = (
        il_grasp_skill.run_grasp_skill(arm, cap, policy_path)
        if grasp_strategy == "il"
        else tsm.descend_and_grasp(arm, click_px=source_click_px or (x, y))
    )
    if grasped:
        print("[click_grasp_trash] 파지 성공 - 상승")
        arm.move_z(config.LIFT_M, steps=20, step_delay_s=0.05)

        if config.BIN_POSE_XYZ is None:
            print(
                "[click_grasp_trash] BIN_POSE_XYZ 미측정 - measure_bin_pose.py로 측정 후 "
                "config.py에 반영하세요. 잡은 채로 제자리 유지합니다."
            )
            return

        print("[click_grasp_trash] 쓰레기통으로 이동 후 놓습니다")
        try:
            arm.move_to_xyz_converge(config.BIN_POSE_XYZ, tolerance_m=0.015, max_iters=20)
        except CollisionDetected as e:
            print(f"[click_grasp_trash] 쓰레기통 이동 중 충돌 감지 ({e}) - 놓지 않고 정지")
            return
        gripper.open_gripper(arm)
        print("[click_grasp_trash] 투기 완료 - 홈으로 복귀합니다")
        try:
            arm.move_to_xyz_converge(home_xyz, tolerance_m=0.015, max_iters=20)
        except CollisionDetected as e:
            print(f"[click_grasp_trash] 홈 복귀 중 충돌 감지: {e}")
        return

    print("[click_grasp_trash] 파지 실패 - 홈으로 복귀합니다")
    try:
        arm.move_to_xyz_converge(home_xyz, tolerance_m=0.015, max_iters=20)
    except CollisionDetected as e:
        print(f"[click_grasp_trash] 홈 복귀 중 충돌 감지: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="오른쪽 팔 follower 포트")
    parser.add_argument(
        "--grasp-strategy", choices=["legacy", "il"], default="legacy", help="파지 전략 (기본: %(default)r)"
    )
    parser.add_argument("--policy-path", default=None, help="grasp-strategy=il일 때 필요한 학습된 ACT 체크포인트 경로")
    parser.add_argument(
        "--dry-hover", action="store_true", help="Stage-2 검증: 클릭 호버까지만 이동 후 파지 없이 홈으로 복귀"
    )
    args = parser.parse_args()

    if args.grasp_strategy == "il" and not args.policy_path:
        parser.error("--grasp-strategy il requires --policy-path")

    robot = SOFollower(
        SOFollowerRobotConfig(port=args.port, id="follower", max_relative_target=config.MAX_RELATIVE_TARGET_DEG)
    )
    robot.connect(calibrate=False)
    print("[click_grasp_trash] 오른팔 연결 성공")

    arm = SOArm101(robot=robot)
    arm.connect()
    home_pose = arm.get_joint_deg()
    home_xyz = tuple(arm.kin.forward_kinematics(home_pose[: len(config.ARM_JOINTS)])[:3, 3])

    click_queue: list[tuple[int, int, tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int, int, int]]] = []
    click_context = ((1, 1), (1, 1), (1, 1), (0, 0, 1, 1))

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            click_queue.append((x, y, *click_context))

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, on_mouse)
    cap = perception.PublishedFrameSource(config.ASTRA_RGB_FRAME_PATH)

    emergency_stopped = False
    try:
        while True:
            ok, frame = cap.read()
            if ok:
                depth_mm = perception.load_fresh_depth()
                rgb_panel = frame.copy()
                valid_mask = perception.depth_valid_mask(depth_mm, rgb_panel.shape)
                if valid_mask is not None and not np.all(valid_mask):
                    invalid = ~valid_mask
                    rgb_panel[invalid] = (rgb_panel[invalid] * 0.35).astype(np.uint8)
                    yy, xx = np.indices(rgb_panel.shape[:2])
                    hatch = invalid & (((xx + yy) // 8) % 2 == 0)
                    rgb_panel[hatch] = (0, 0, 255)  # BGR red diagonal hatch over unsafe (no-depth) area
                    cv2.putText(rgb_panel, "NO DEPTH: do not click shaded area", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                elif depth_mm is None:
                    cv2.putText(rgb_panel, "NO DEPTH: waiting for depth stream", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                polygon = perception.reach_boundary_polygon_px(rgb_panel.shape)
                if polygon is not None:
                    safe = np.zeros(rgb_panel.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(safe, [polygon], 1)
                    unreachable = safe == 0
                    rgb_panel[unreachable] = (rgb_panel[unreachable] * 0.35).astype(np.uint8)
                    yy, xx = np.indices(rgb_panel.shape[:2])
                    hatch = unreachable & (((xx + yy) // 8) % 2 == 0)
                    rgb_panel[hatch] = (0, 0, 255)  # BGR red diagonal hatch: outside arm's reach arc
                if depth_mm is None:
                    depth_panel = np.zeros_like(rgb_panel)
                    cv2.putText(depth_panel, "NO DEPTH", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    depth_vis = cv2.normalize(depth_mm, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    depth_panel = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)
                    if depth_panel.shape[:2] != rgb_panel.shape[:2]:
                        depth_panel = cv2.resize(depth_panel, (rgb_panel.shape[1], rgb_panel.shape[0]))
                rgb_display = cv2.resize(
                    rgb_panel, None, fx=perception.RGBD_PREVIEW_SCALE, fy=perception.RGBD_PREVIEW_SCALE, interpolation=cv2.INTER_NEAREST
                )
                depth_display = cv2.resize(
                    depth_panel, None, fx=perception.RGBD_PREVIEW_SCALE, fy=perception.RGBD_PREVIEW_SCALE, interpolation=cv2.INTER_NEAREST
                )
                display = np.hstack((rgb_display, depth_display))
                cv2.imshow(WINDOW, display)
                try:
                    window_rect = cv2.getWindowImageRect(WINDOW)
                except cv2.error:
                    window_rect = (0, 0, display.shape[1], display.shape[0])
                if window_rect[2] <= 0 or window_rect[3] <= 0:
                    window_rect = (0, 0, display.shape[1], display.shape[0])
                click_context = (
                    (display.shape[1], display.shape[0]),
                    (rgb_display.shape[1], rgb_display.shape[0]),
                    (frame.shape[1], frame.shape[0]),
                    window_rect,
                )
            if click_queue:
                raw_x, raw_y, display_size, rgb_display_size, frame_size, window_rect = click_queue.pop(0)
                panel_px = (raw_x * display_size[0] / window_rect[2], raw_y * display_size[1] / window_rect[3])
                if not (0 <= panel_px[0] < rgb_display_size[0] and 0 <= panel_px[1] < rgb_display_size[1]):
                    print("[click_grasp_trash] depth 패널 클릭은 무시합니다")
                    continue
                display_to_rgb = (frame_size[0] / rgb_display_size[0], frame_size[1] / rgb_display_size[1])
                source_click_px = (panel_px[0] * display_to_rgb[0], panel_px[1] * display_to_rgb[1])
                x, y = perception.rgb_px_to_homography_px(*source_click_px, (frame_size[1], frame_size[0]))
                print(
                    "[click_grasp_trash] click debug "
                    f"raw=({raw_x}, {raw_y}) display={display_size} rgb_display={rgb_display_size} frame={frame_size} "
                    f"window_rect={window_rect} display_to_rgb={display_to_rgb} "
                    f"pixel_to_xy_input=({x:.1f}, {y:.1f})"
                )
                handle_click(
                    x,
                    y,
                    arm,
                    home_xyz,
                    args.grasp_strategy,
                    args.policy_path,
                    cap,
                    args.dry_hover,
                    source_click_px=source_click_px,
                )
            key = cv2.waitKey(30) & 0xFF
            if key == ord("e"):
                arm.release_torque()
                print("[click_grasp_trash] 비상 정지 - 오른팔 토크 해제됨")
                emergency_stopped = True
                break
            if key in (ord("q"), 27):
                break
    finally:
        cv2.destroyAllWindows()
        if emergency_stopped:
            print("[click_grasp_trash] 비상 정지 상태 유지 - 홈 복귀 시도하지 않음 (토크 해제된 채로 둠)")
        else:
            print(f"[click_grasp_trash] 홈 복귀: {home_xyz}")
            try:
                arm.move_to_xyz_converge(home_xyz, tolerance_m=0.015, max_iters=20)
            except CollisionDetected as e:
                print(f"[click_grasp_trash] 홈 복귀 중 충돌 감지: {e}")
            except Exception as e:
                print(f"[click_grasp_trash] 홈 복귀 실패: {e}")
        robot.disconnect()


if __name__ == "__main__":
    main()
