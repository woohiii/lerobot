"""GraspNet 카메라 좌표계를 SO-101 베이스 좌표계로 보정하는 수동 도구."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


OUTPUT_PATH = Path(__file__).resolve().parent / "camera_extrinsic.json"
CALIBRATION_POINTS_XYZ = (
    (0.20, -0.10, 0.08),
    (0.20, 0.10, 0.08),
    (0.35, -0.10, 0.08),
    (0.35, 0.10, 0.08),
    (0.24, -0.06, 0.14),
    (0.24, 0.06, 0.14),
    (0.32, -0.06, 0.18),
    (0.32, 0.06, 0.18),
)
WINDOW = "GraspNet 3D extrinsic calibration - click gripper tip ('q'=skip)"


def estimate_rigid_transform(
    camera_points: np.ndarray, robot_points: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """카메라 점을 로봇 점으로 보내는 회전, 이동, 평균 오차를 계산한다."""
    source = np.asarray(camera_points, dtype=np.float64)
    target = np.asarray(robot_points, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("camera_points와 robot_points는 같은 Nx3 배열이어야 합니다")
    if len(source) < 3:
        raise ValueError("강체변환에는 최소 3개의 점이 필요합니다")
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    centered_source = source - source_center
    centered_target = target - target_center
    covariance = centered_source.T @ centered_target
    left, _singular_values, right_transposed = np.linalg.svd(covariance)
    rotation = right_transposed.T @ left.T
    if np.linalg.det(rotation) < 0:
        right_transposed[-1, :] *= -1.0
        rotation = right_transposed.T @ left.T
    translation = target_center - rotation @ source_center
    projected = (rotation @ source.T).T + translation
    errors = np.linalg.norm(projected - target, axis=1)
    return rotation, translation, float(np.mean(errors))


def depth_pixel_to_camera_xyz_m(
    pixel_xy: tuple[float, float], depth_mm: np.ndarray, intrinsics: dict[str, float], rgb_shape: tuple[int, ...]
) -> tuple[float, float, float] | None:
    """RGB 클릭과 depth 배열을 pinhole 카메라 좌표계 meter로 변환한다."""
    rgb_height, rgb_width = rgb_shape[:2]
    depth_height, depth_width = depth_mm.shape
    depth_x = int(round(pixel_xy[0] * depth_width / rgb_width))
    depth_y = int(round(pixel_xy[1] * depth_height / rgb_height))
    depth_x = min(max(depth_x, 0), depth_width - 1)
    depth_y = min(max(depth_y, 0), depth_height - 1)
    value_mm = float(depth_mm[depth_y, depth_x])
    if not np.isfinite(value_mm) or value_mm <= 0:
        return None
    x = (pixel_xy[0] - intrinsics["cx"]) * (value_mm / 1000.0) / intrinsics["fx"]
    y = (pixel_xy[1] - intrinsics["cy"]) * (value_mm / 1000.0) / intrinsics["fy"]
    return float(x), float(y), value_mm / 1000.0


def _capture_gripper_tip_pixel(cap, point_label: str) -> tuple[float, float] | None:
    """published RGB 창에서 그리퍼 팁 클릭을 한 번 수집한다."""
    click_queue: list[tuple[int, int, tuple[int, int], tuple[int, int, int, int]]] = []
    click_context = ((1, 1), (0, 0, 1, 1))

    def on_mouse(event, x, y, _flags, _userdata):
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
                return (
                    raw_x * frame_size[0] / window_rect[2],
                    raw_y * frame_size[1] / window_rect[3],
                )
    finally:
        cv2.destroyWindow(WINDOW)


def _load_intrinsics() -> dict[str, float]:
    """Phase 2-1 bridge의 intrinsic 기본값을 그대로 재사용한다."""
    from graspnet_bridge import infer_published_grasps

    defaults = infer_published_grasps.__kwdefaults__ or {}
    return {key: float(defaults[key]) for key in ("fx", "fy", "cx", "cy")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual", action="store_true", help="각 목표점에서 그리퍼 팁을 직접 클릭")
    args = parser.parse_args()
    if not args.manual:
        parser.error("이 작업은 --manual 방식만 지원합니다")

    import config
    import perception
    from kinematics import SOArm101

    cap = perception.PublishedFrameSource(config.ASTRA_RGB_FRAME_PATH)
    depth_mm = perception.load_fresh_depth(config.ASTRA_DEPTH_MM_PATH)
    if not cap.isOpened() or depth_mm is None:
        print("[calibrate_graspnet_extrinsic] Astra RGB/depth published frame이 없습니다.")
        print("[calibrate_graspnet_extrinsic] 먼저 astra_s_stream_supervisor.py --mode rgbd를 실행하세요.")
        return

    intrinsics = _load_intrinsics()
    arm = SOArm101(port=config.FOLLOWER_PORT)
    arm.connect()
    print("[calibrate_graspnet_extrinsic] 오른팔 연결 성공")
    camera_points: list[tuple[float, float, float]] = []
    robot_points: list[tuple[float, float, float]] = []
    try:
        for index, target in enumerate(CALIBRATION_POINTS_XYZ, start=1):
            print(f"\n[{index}/{len(CALIBRATION_POINTS_XYZ)}] 목표 xyz={target}")
            reached = np.asarray(arm.move_to_xyz_converge(target, tolerance_m=0.015, max_iters=20), dtype=float)
            print(f"   실제 도달 위치: {reached.tolist()}")
            if reached.shape != (3,) or np.linalg.norm(reached - np.asarray(target)) > 0.035:
                print("   [건너뜀] 목표에 충분히 도달하지 못했습니다.")
                continue
            pixel = _capture_gripper_tip_pixel(cap, f"[{index}/{len(CALIBRATION_POINTS_XYZ)}] xyz={target}")
            if pixel is None:
                print("   [건너뜀] 사용자가 건너뛰었습니다.")
                continue
            ok, frame = cap.read()
            depth_mm = perception.load_fresh_depth(config.ASTRA_DEPTH_MM_PATH)
            if not ok or frame is None or depth_mm is None:
                print("   [건너뜀] 클릭 직후 RGB/depth 프레임이 없습니다.")
                continue
            camera_point = depth_pixel_to_camera_xyz_m(pixel, depth_mm, intrinsics, frame.shape)
            if camera_point is None:
                print("   [건너뜀] 클릭 위치의 depth가 유효하지 않습니다.")
                continue
            camera_points.append(camera_point)
            robot_points.append(tuple(float(value) for value in reached))
            print(f"   camera xyz(m)={camera_point}, robot xyz(m)={tuple(reached)}")

        if len(camera_points) < 6:
            print(f"\n[중단] 유효한 점이 {len(camera_points)}개입니다. 최소 6개가 필요합니다.")
            return
        rotation, translation, mean_error = estimate_rigid_transform(
            np.asarray(camera_points), np.asarray(robot_points)
        )
        payload = {
            "rotation_matrix_3x3": rotation.tolist(),
            "translation_m": translation.tolist(),
            "camera_points": camera_points,
            "robot_points": robot_points,
            "mean_reprojection_error_m": mean_error,
        }
        with open(OUTPUT_PATH, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
        print(f"\n[calibrate_graspnet_extrinsic] 저장 완료: {OUTPUT_PATH}")
        print(f"[calibrate_graspnet_extrinsic] 평균 재투영 오차(m): {mean_error:.6f}")
    finally:
        cv2.destroyAllWindows()
        cap.release()
        arm.disconnect()


if __name__ == "__main__":
    main()
