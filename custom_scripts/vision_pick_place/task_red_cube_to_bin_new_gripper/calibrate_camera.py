"""Camera-to-robot homography (re)calibration via touch points - adapted
from the older custom_scripts/vision_pick_place/calibrate_camera.py, with
two changes this session's real hardware use surfaced:

1. Uses kinematics.SOArm101 (this task's protected class - firmware Max_
   Torque_Limit/Protection_Current/Overload_Torque caps + stall detection on
   all 5 arm joints via connect()'s own _protect_arm_motors, config.
   FOLLOWER_PORT) instead of the old script's less-protected RobotController/
   hardcoded /dev/ttyACM0 - per the user's 2026-09-01 "모터 고장 안나게" ask.
2. Enter-key pacing ("place the cube, press Enter") doesn't work when this
   process's stdin isn't a real terminal - e.g. launched by an agent, same
   `< /dev/null` problem the old script's own docstring already documents
   for an earlier fixed-sleep version, just one layer further removed (an
   agent relaying "press Enter" can't actually press it either). Swapped
   Enter for a MOUSE CLICK on the preview window - the same pacing mechanism
   click_pick_place_zeroshot.py already uses for exactly this "a human's
   real-world action, relayed through an agent" problem.

Triggered by a real, confirmed-broken homography this session: the stored
calibration's 4 pixel_points cluster in a tiny region of the current live
Astra frame instead of spanning the table, meaning the camera moved since
the last calibration - see this session's log for the visual check that
found it.

Flow per point: arm moves to a known (x, y, hover height) -> preview window
shows "place the red cube directly under the gripper tip, click when ready"
-> click -> captures the cube's camera-pixel position. After all points,
fits a homography (pixel -> robot-frame x,y) and overwrites homography.json.
'q'/ESC on any point's click prompt skips that point (not the whole run).

Run: uv run python3 custom_scripts/vision_pick_place/task_red_cube_to_bin_new_gripper/calibrate_camera.py
"""

from __future__ import annotations

import json
import time

import cv2
import numpy as np

import config
import perception
from kinematics import SOArm101

# Same rectangle the old script used - within the arm's confirmed-reachable
# area, unaffected by the camera-position issue this recalibration fixes
# (reachability is a robot-kinematics property, not a camera one).
CALIB_POINTS_XY = [
    (0.18, -0.08),
    (0.18, 0.08),
    (0.28, -0.08),
    (0.28, 0.08),
    (0.23, 0.0),  # center, extra point for a more robust fit
]

# 2026-09-01: 8cm (previous try) -> 5cm per the user's follow-up ask.
# config.TABLE_Z is still the reference (real measured gripper-tip contact
# height for the current gripper).
CALIB_HOVER_Z = config.TABLE_Z + 0.05

WINDOW = "calibration - place the red cube under the gripper tip, click when ready ('q'=skip point)"


def wait_for_ready_click(cap: perception.PublishedFrameSource, point_label: str) -> bool:
    """True on click (operator signals "cube placed"), False on 'q'/ESC (skip)."""
    clicked = []

    def on_click(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked.append((x, y))

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_click)
    try:
        while True:
            ok, frame = cap.read()
            if ok:
                display = frame.copy()
                cv2.putText(display, point_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(display, "click when the cube is placed ('q'=skip)", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                cv2.imshow(WINDOW, display)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                return False
            if clicked:
                return True
    finally:
        cv2.destroyWindow(WINDOW)


def capture_cube_pixel(cap: perception.PublishedFrameSource, tries: int = 20) -> tuple[float, float] | None:
    for _ in range(tries):
        ret, frame = cap.read()
        if not ret or frame is None or perception.is_frame_corrupted(frame):
            time.sleep(0.05)
            continue
        det = perception.detect_red_cube(frame)
        if det is not None:
            return (det.cx, det.cy)
        time.sleep(0.05)
    return None


def main() -> None:
    cap = perception.PublishedFrameSource(config.ASTRA_RGB_FRAME_PATH)
    if not cap.isOpened():
        print(
            "[calibrate] Astra RGB 프레임이 없습니다. 먼저 astra_s_live.py를 실행해주세요:\n"
            "  source ~/lerobot_song_venv/bin/activate && "
            "python custom_scripts/vision_pick_place/astra_s_live.py"
        )
        return

    arm = SOArm101()
    arm.connect()
    print("[calibrate] 연결 성공.")

    pixel_pts: list[tuple[float, float]] = []
    robot_pts: list[tuple[float, float]] = []

    try:
        for i, (x, y) in enumerate(CALIB_POINTS_XY):
            target = (x, y, CALIB_HOVER_Z)
            print(f"\n[{i + 1}/{len(CALIB_POINTS_XY)}] target={target}")
            reached = arm.move_to_xyz_converge(target, tolerance_m=0.02, max_iters=15)
            print(f"   실제 도달 위치: {reached}")
            if np.linalg.norm(np.array(target) - reached) > 0.035:
                print("   [건너뜀] 목표에 충분히 도달하지 못했습니다 (도달 범위 밖일 수 있음).")
                continue

            ready = wait_for_ready_click(cap, f"[{i + 1}/{len(CALIB_POINTS_XY)}] target xy=({x:.2f},{y:.2f})")
            if not ready:
                print("   [건너뜀] 사용자가 건너뛰었습니다.")
                continue

            pixel = capture_cube_pixel(cap)
            if pixel is None:
                print("   [실패] 큐브를 카메라에서 못 찾았습니다. 이 점은 건너뜁니다.")
                continue

            print(f"   픽셀 좌표: {pixel}")
            pixel_pts.append(pixel)
            robot_pts.append((x, y))

        if len(pixel_pts) < 4:
            print(f"\n[중단] 유효한 포인트가 {len(pixel_pts)}개뿐입니다 (호모그래피에는 최소 4개 필요). 다시 시도해주세요.")
            return

        src = np.array(pixel_pts, dtype=np.float32)
        dst = np.array(robot_pts, dtype=np.float32)
        H, _mask = cv2.findHomography(src, dst, method=0)
        print("\n호모그래피 행렬:\n", H)

        errors = []
        for (px, py), (rx, ry) in zip(pixel_pts, robot_pts):
            pt = np.array([px, py, 1.0])
            mapped = H @ pt
            mapped /= mapped[2]
            err = np.hypot(mapped[0] - rx, mapped[1] - ry)
            errors.append(err)
        print(f"재투영 오차(m): {errors} (평균 {np.mean(errors):.4f})")

        with open(config.HOMOGRAPHY_PATH, "w") as f:
            json.dump(
                {
                    "homography": H.tolist(),
                    "table_z": CALIB_HOVER_Z,
                    "pixel_points": pixel_pts,
                    "robot_points": robot_pts,
                    "mean_reprojection_error_m": float(np.mean(errors)),
                },
                f,
                indent=2,
            )
        print(f"\n저장 완료: {config.HOMOGRAPHY_PATH}")
    finally:
        cv2.destroyAllWindows()
        cap.release()
        arm.disconnect()


if __name__ == "__main__":
    main()
