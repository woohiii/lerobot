"""오른팔 작업영역용 Astra RGB-로봇 xy 호모그래피 수동 보정 도구.

각 목표점으로 안전하게 이동한 뒤, 화면에서 그리퍼 팁을 직접 클릭한다.
유효 점이 4개 이상이면 공유 homography.json을 새 보정값으로 교체한다.
"""

from __future__ import annotations

import argparse
import json

import cv2
import numpy as np

import config
import perception
from kinematics import SOArm101

CALIB_POINTS_XY = [
    (0.20, 0.00),
    (0.23, 0.00),
    (0.25, 0.00),
    (0.27, 0.00),
    (0.23, -0.08),
    (0.23, 0.08),
    (0.25, -0.08),
    (0.25, 0.08),
    (0.20, -0.10),
    (0.20, 0.10),
]
CALIB_HOVER_Z = config.TABLE_Z + 0.05
WINDOW = "Astra RGB homography calibration - click gripper tip ('q'=skip)"


def capture_gripper_tip_pixel(cap: perception.PublishedFrameSource, point_label: str) -> tuple[float, float] | None:
    click_queue: list[tuple[int, int, tuple[int, int], tuple[int, int, int, int]]] = []
    click_context = ((1, 1), (0, 0, 1, 1))

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            click_queue.append((x, y, *click_context))

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, on_mouse)
    try:
        while True:
            ok, frame = cap.read()
            if ok:
                display = frame.copy()
                cv2.putText(display, point_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                cv2.putText(display, "click gripper tip ('q'=skip)", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.imshow(WINDOW, display)
                try:
                    window_rect = cv2.getWindowImageRect(WINDOW)
                except cv2.error:
                    window_rect = (0, 0, display.shape[1], display.shape[0])
                if window_rect[2] <= 0 or window_rect[3] <= 0:
                    window_rect = (0, 0, display.shape[1], display.shape[0])
                click_context = ((frame.shape[1], frame.shape[0]), window_rect)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                return None
            if click_queue:
                raw_x, raw_y, frame_size, window_rect = click_queue.pop(0)
                source_px = (raw_x * frame_size[0] / window_rect[2], raw_y * frame_size[1] / window_rect[3])
                return perception.rgb_px_to_homography_px(*source_px, (frame_size[1], frame_size[0]))
    finally:
        cv2.destroyWindow(WINDOW)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual", action="store_true", help="각 목표점에서 화면의 그리퍼 팁을 직접 클릭")
    args = parser.parse_args()
    if not args.manual:
        parser.error("이 작업은 --manual 방식만 지원합니다")

    cap = perception.PublishedFrameSource(config.ASTRA_RGB_FRAME_PATH)
    if not cap.isOpened():
        print("[calibrate] Astra RGB 프레임이 없습니다. 먼저 astra_s_stream_supervisor.py --mode rgbd를 실행하세요.")
        return

    arm = SOArm101(port=config.FOLLOWER_PORT)
    arm.connect()
    print("[calibrate] 오른팔 연결 성공")
    pixel_pts: list[tuple[float, float]] = []
    robot_pts: list[tuple[float, float]] = []
    try:
        for index, (x, y) in enumerate(CALIB_POINTS_XY, start=1):
            target = (x, y, CALIB_HOVER_Z)
            print(f"\n[{index}/{len(CALIB_POINTS_XY)}] 목표={target}")
            reached = arm.move_to_xyz_converge(target, tolerance_m=0.02, max_iters=15)
            print(f"   실제 도달 위치: {reached}")
            if np.linalg.norm(np.array(target) - reached) > 0.035:
                print("   [건너뜀] 목표에 충분히 도달하지 못했습니다.")
                continue
            pixel = capture_gripper_tip_pixel(cap, f"[{index}/{len(CALIB_POINTS_XY)}] target xy=({x:.2f}, {y:.2f})")
            if pixel is None:
                print("   [건너뜀] 사용자가 건너뛰었습니다.")
                continue
            print(f"   호모그래피 기준 픽셀: {pixel}")
            pixel_pts.append(pixel)
            robot_pts.append((x, y))

        if len(pixel_pts) < 4:
            print(f"\n[중단] 유효한 점이 {len(pixel_pts)}개입니다. 호모그래피에는 최소 4개가 필요합니다.")
            return

        # RANSAC (not the plain least-squares method=0): a single misclick among
        # many taught points otherwise drags the whole fit toward the outlier.
        homography, inlier_mask = cv2.findHomography(
            np.array(pixel_pts, dtype=np.float32), np.array(robot_pts, dtype=np.float32), method=cv2.RANSAC, ransacReprojThreshold=0.02
        )
        if homography is None:
            print("[중단] 호모그래피 계산에 실패했습니다.")
            return
        if inlier_mask is not None and not bool(np.all(inlier_mask)):
            outliers = [i + 1 for i, ok in enumerate(inlier_mask.ravel()) if not ok]
            print(f"[calibrate] RANSAC이 이상치로 제외한 점: {outliers}")
        errors = []
        for (px, py), (rx, ry) in zip(pixel_pts, robot_pts):
            mapped = homography @ np.array([px, py, 1.0])
            mapped /= mapped[2]
            errors.append(float(np.hypot(mapped[0] - rx, mapped[1] - ry)))
        # RANSAC-excluded points are known-bad clicks the fit didn't use, so
        # averaging their (large, by definition) error back in would misreport
        # the homography's real accuracy - report inliers only.
        inlier_errors = errors if inlier_mask is None else [e for e, ok in zip(errors, inlier_mask.ravel()) if ok]
        with open(config.HOMOGRAPHY_PATH, "w") as f:
            json.dump(
                {
                    "homography": homography.tolist(),
                    "table_z": CALIB_HOVER_Z,
                    "pixel_points": pixel_pts,
                    "robot_points": robot_pts,
                    "mean_reprojection_error_m": float(np.mean(inlier_errors)),
                },
                f,
                indent=2,
            )
        print(f"\n[calibrate] 저장 완료: {config.HOMOGRAPHY_PATH}")
        print(f"[calibrate] 재투영 오차(m) 전체: {errors}")
        print(f"[calibrate] 인라이어 기준 평균={np.mean(inlier_errors):.4f} (n={len(inlier_errors)}/{len(errors)})")
    finally:
        cv2.destroyAllWindows()
        cap.release()
        arm.disconnect()


if __name__ == "__main__":
    main()
