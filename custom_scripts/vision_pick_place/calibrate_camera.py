"""Camera-to-robot calibration via touch points: the gripper moves to N known
(x, y) spots at a fixed table height; at each one, place the red cube directly
under the gripper tip and press Enter, and the camera's cube-pixel position is
paired with the robot-frame (x, y) and recorded. After all points, fits a
homography mapping camera pixel -> robot-frame (x, y) on the table plane, and
saves it to homography.json for pick_place.py to use.

2026-08-26: originally a fixed WAIT_PER_POINT_S sleep instead of an Enter
prompt, on the assumption the operator would be watching the terminal
directly - broke down badly when relayed through an agent instead (all 5
points' waits elapsed before the relay could tell the operator "now", so 4/5
points captured nothing). Enter-gated like manual_grasp_calibration.py so the
person actually placing the cube controls the pacing directly, no matter who
(or what) is driving the terminal.
"""

import json
import time

import cv2
import numpy as np

from camera_utils import ASTRA_RGB_FRAME_PATH, PublishedFrameSource
from cube_detector import detect_red_cube, is_frame_corrupted
from robot_control import RobotController
TABLE_Z = 0.003  # 2026-08-26: was 0.045, physically ~45mm above the real table -
# see the dated comment on this same constant in visual_servo_pick_place.py

# Rectangle of touch points within the arm's confirmed-reachable area (from earlier
# preview_move/IK tests around (0.25, 0, 0.05)). Visually unverified until run.
CALIB_POINTS_XY = [
    (0.18, -0.08),
    (0.18, 0.08),
    (0.28, -0.08),
    (0.28, 0.08),
    (0.23, 0.0),  # center, extra point for a more robust fit
]

OUT_PATH = "homography.json"


def capture_cube_pixel(cap: PublishedFrameSource, tries: int = 20) -> tuple[float, float] | None:
    for _ in range(tries):
        ret, frame = cap.read()
        if not ret or frame is None or is_frame_corrupted(frame):
            time.sleep(0.05)
            continue
        det = detect_red_cube(frame)
        if det is not None:
            return (det.cx, det.cy)
        time.sleep(0.05)
    return None


def main():
    # 2026-08-26: follower 전용 USB 어댑터 보드가 고장나 leader 보드를 대신 꽂아 씀 -
    # visual_servo_pick_place.py의 같은 날짜 주석 참고. 또한 RGB 소스도 이날 Astra Pro
    # Plus에서 Astra S로 바뀌어 astra_s_live.py가 발행하는 프레임을 읽는다 (camera_hub.py
    # 가 아님 - 그쪽은 지금 손목캠만 발행 중).
    rc = RobotController(port="/dev/ttyACM0")
    rc.connect()
    cap = PublishedFrameSource(ASTRA_RGB_FRAME_PATH)
    if not cap.isOpened():
        print(
            "[calibrate] Astra RGB 프레임이 없습니다. 먼저 다른 터미널에서 astra_s_live.py를 "
            "~/lerobot_song_venv로 실행해주세요."
        )
        rc.disconnect()
        return

    pixel_pts = []
    robot_pts = []

    try:
        for i, (x, y) in enumerate(CALIB_POINTS_XY):
            target = (x, y, TABLE_Z)
            plan = rc.preview_move(target)
            print(f"\n[{i+1}/{len(CALIB_POINTS_XY)}] target={target} max_delta={plan['max_abs_delta_deg']:.1f}deg")
            # No outright rejection here: the robot currently starts folded/tucked
            # (not the outstretched pose earlier IK tests were run from), so even
            # ordinary reachable table points can show a large one-shot delta.
            # move_to_xyz_converge is what actually executes the move, in small
            # (lerobot-clamped) increments each retry - that's the real safety
            # boundary, not this preview number.
            # tolerance loosened from an initial 0.008: a live test converged to a
            # stable ~0.017m residual (likely a joint limit near that particular
            # target, not a bug - it plateaus rather than drifting) and then just
            # spent the rest of its iteration budget oscillating in place. 0.02 is
            # still fine for touch-point calibration against a 4cm cube.
            reached = rc.move_to_xyz_converge(target, tolerance_m=0.02, max_iters=15)
            print(f"   실제 도달 위치: {reached}")
            if np.linalg.norm(np.array(target) - reached) > 0.035:
                print("   [건너뜀] 목표에 충분히 도달하지 못했습니다 (도달 범위 밖일 수 있음).")
                continue

            input("   빨간 큐브를 그리퍼 바로 아래(테이블 위)에 놓아주세요 - 준비되면 Enter... ")

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
        H, mask = cv2.findHomography(src, dst, method=0)
        print("\n호모그래피 행렬:\n", H)

        # reprojection error check
        errors = []
        for (px, py), (rx, ry) in zip(pixel_pts, robot_pts):
            pt = np.array([px, py, 1.0])
            mapped = H @ pt
            mapped /= mapped[2]
            err = np.hypot(mapped[0] - rx, mapped[1] - ry)
            errors.append(err)
        print(f"재투영 오차(m): {errors} (평균 {np.mean(errors):.4f})")

        with open(OUT_PATH, "w") as f:
            json.dump(
                {
                    "homography": H.tolist(),
                    "table_z": TABLE_Z,
                    "pixel_points": pixel_pts,
                    "robot_points": robot_pts,
                    "mean_reprojection_error_m": float(np.mean(errors)),
                },
                f,
                indent=2,
            )
        print(f"\n저장 완료: {OUT_PATH}")
    finally:
        cap.release()
        rc.disconnect()


if __name__ == "__main__":
    main()
