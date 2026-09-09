"""Interactively collect camera/base correspondences without automatic motion."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def auto_select_depth_pixel(depth: np.ndarray) -> tuple[int, int]:
    """Select the center of the nearest valid depth cluster (usually the gripper)."""
    values = np.asarray(depth)
    valid = values[(values > 0) & np.isfinite(values)]
    if len(valid) == 0:
        raise RuntimeError("no valid depth pixels")
    threshold = float(np.percentile(valid, 1.0))
    rows, cols = np.where((values > 0) & (values <= threshold))
    if len(cols) == 0:
        raise RuntimeError("no valid depth cluster")
    center = np.array([float(np.mean(cols)), float(np.mean(rows))])
    distances = (cols - center[0]) ** 2 + (rows - center[1]) ** 2
    selected = int(np.argmin(distances))
    return int(cols[selected]), int(rows[selected])


def nearest_valid_depth_pixel(depth: np.ndarray, u: int, v: int, radius: int = 6) -> tuple[int, int]:
    """Find the nearest nonzero depth sample around a clicked TCP pixel."""
    y0, y1 = max(0, v - radius), min(depth.shape[0], v + radius + 1)
    x0, x1 = max(0, u - radius), min(depth.shape[1], u + radius + 1)
    window = depth[y0:y1, x0:x1]
    rows, cols = np.where(window > 0)
    if len(cols) == 0:
        raise RuntimeError("선택한 TCP 주변에 유효한 depth가 없습니다")
    distances = (cols + x0 - u) ** 2 + (rows + y0 - v) ** 2
    selected = int(np.argmin(distances))
    return int(cols[selected] + x0), int(rows[selected] + y0)


def find_nearest_valid_depth_pixel(
    depth: np.ndarray,
    u: int,
    v: int,
    *,
    radius: int = 6,
    max_radius: int = 48,
) -> tuple[int, int] | None:
    """Search progressively farther when the clicked TCP pixel is occluded."""
    search_radius = max(1, radius)
    while search_radius <= max_radius:
        try:
            return nearest_valid_depth_pixel(depth, u, v, radius=search_radius)
        except RuntimeError:
            search_radius *= 2
    return None


def resolve_depth_pixel(
    depth: np.ndarray,
    u: int,
    v: int,
    *,
    allow_fallback: bool = False,
) -> tuple[int, int] | None:
    """Resolve a selected pixel, optionally allowing an offset depth sample."""
    if depth[v, u] > 0:
        return u, v
    if not allow_fallback:
        return None
    return find_nearest_valid_depth_pixel(depth, u, v)


def build_target_depth_pixels(
    count: int, width: int = 320, height: int = 240, *, enabled: bool = True
) -> list[tuple[int, int]]:
    """Build evenly spaced on-screen guide points for any requested count."""
    if not enabled or count < 1:
        return []
    columns = min(4, count)
    rows = math.ceil(count / columns)
    xs = np.linspace(width * 0.2, width * 0.8, columns)
    ys = np.linspace(height * 0.2, height * 0.8, rows)
    return [(int(round(x)), int(round(y))) for y in ys for x in xs][:count]


def validate_joint_points(joint_points: np.ndarray, expected_count: int | None = None) -> np.ndarray:
    """Validate the per-sample joint-angle matrix before saving it."""
    values = np.asarray(joint_points, dtype=float)
    if values.ndim != 2 or values.shape[1] < 5:
        raise ValueError("joint points must be a finite N x 5+ matrix")
    if expected_count is not None and values.shape[0] != expected_count:
        raise ValueError("joint points row count does not match calibration points")
    if not np.isfinite(values).all():
        raise ValueError("joint points must be finite")
    return values


def prepare_manual_collection_arm(arm) -> None:
    """Release follower torque before hand-guided correspondence capture."""
    arm.release_torque()


def is_tcp_at_table_height(tcp_z: float, *, table_z: float, tolerance: float) -> bool:
    """Return whether a TCP sample is close enough to the calibrated table plane."""
    return np.isfinite(tcp_z) and np.isfinite(table_z) and np.isfinite(tolerance) and tolerance > 0 and abs(tcp_z - table_z) <= tolerance


def is_sample_near_home(
    tcp_xyz, joints_deg, home_xyz, home_joints_deg, *, max_xy_radius: float, wrist_tolerance: float
) -> bool:
    """Keep samples near home while preserving wrist flex/roll angles."""
    xyz = np.asarray(tcp_xyz, dtype=float)
    joints = np.asarray(joints_deg, dtype=float)
    home_xyz = np.asarray(home_xyz, dtype=float)
    home_joints = np.asarray(home_joints_deg, dtype=float)
    return (
        xyz.shape == (3,) and joints.shape[0] >= 5 and home_xyz.shape == (3,) and home_joints.shape[0] >= 5
        and np.isfinite(xyz).all() and np.isfinite(joints).all() and np.isfinite(home_xyz).all()
        and np.isfinite(home_joints).all() and np.isfinite(max_xy_radius) and max_xy_radius > 0
        and np.isfinite(wrist_tolerance) and wrist_tolerance > 0
        and float(np.linalg.norm(xyz[:2] - home_xyz[:2])) <= max_xy_radius
        and bool(np.all(np.abs(joints[3:5] - home_joints[3:5]) <= wrist_tolerance))
    )


def is_sample_within_anchor_envelope(
    tcp_xyz,
    anchor_xyz,
    *,
    max_base_radius_m: float,
) -> bool:
    """Keep samples within a measured maximum base-frame radius."""
    xyz = np.asarray(tcp_xyz, dtype=float)
    anchor = np.asarray(anchor_xyz, dtype=float)
    if (
        xyz.shape != (3,)
        or anchor.shape != (3,)
        or not np.isfinite(xyz).all()
        or not np.isfinite(anchor).all()
        or not np.isfinite(max_base_radius_m)
        or max_base_radius_m <= 0
    ):
        return False
    return float(np.linalg.norm(xyz[:2])) <= max_base_radius_m


def is_vertical_grasp_pose(tool_pose: np.ndarray, *, max_tilt_deg: float) -> bool:
    """Accept only a downward-facing gripper tool axis near the table normal."""
    pose = np.asarray(tool_pose, dtype=float)
    if pose.shape != (4, 4) or not np.isfinite(pose).all() or not np.isfinite(max_tilt_deg) or not 0 < max_tilt_deg < 90:
        return False
    approach_axis = pose[:3, 2]
    norm = float(np.linalg.norm(approach_axis))
    if norm == 0:
        return False
    return float(np.dot(approach_axis / norm, np.array([0.0, 0.0, -1.0]))) >= float(np.cos(np.deg2rad(max_tilt_deg)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="SO-101 follower serial port")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--points", type=int, default=6)
    parser.add_argument("--table-z", type=float, help="table plane Z in robot-base metres; default is the startup anchor TCP Z")
    parser.add_argument("--table-z-tolerance", type=float, default=0.015, help="allowed TCP height error in metres")
    parser.add_argument("--max-base-radius-m", type=float, help="maximum allowed TCP XY radius; default is startup anchor radius")
    parser.add_argument("--max-vertical-tilt-deg", type=float, default=20.0, help="maximum gripper tilt from downward vertical")
    parser.add_argument(
        "--manual-pixel",
        action="store_true",
        help="ask for the raw-depth u,v at each point instead of choosing the nearest surface",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="show Astra RGB and click TCP pixels; press Space to capture, q to quit",
    )
    parser.add_argument(
        "--allow-depth-fallback",
        action="store_true",
        help="use a nearby depth pixel when the selected TCP pixel is zero (less accurate)",
    )
    args = parser.parse_args(argv)
    if args.points < 3:
        parser.error("at least three points are required")
    if not np.isfinite(args.table_z) if args.table_z is not None else False:
        parser.error("--table-z must be finite")
    if not np.isfinite(args.table_z_tolerance) or args.table_z_tolerance <= 0:
        parser.error("--table-z-tolerance must be positive and finite")
    if args.max_base_radius_m is not None and (not np.isfinite(args.max_base_radius_m) or args.max_base_radius_m <= 0):
        parser.error("--max-base-radius-m must be positive and finite")
    if not np.isfinite(args.max_vertical_tilt_deg) or not 0 < args.max_vertical_tilt_deg < 90:
        parser.error("--max-vertical-tilt-deg must be between 0 and 90")

    vision_dir = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(vision_dir))
    from orbbec_color_camera import ThreadedOrbbecRGBDCamera
    from primesense import openni2

    from .left_arm import LeftSOArm101

    camera = ThreadedOrbbecRGBDCamera(width=640, height=480, fps=30)
    arm = LeftSOArm101(port=args.port)
    arm_connected = False
    camera_points: list[tuple[float, float, float]] = []
    base_points: list[tuple[float, float, float]] = []
    joint_points: list[np.ndarray] = []
    captured_pixels: list[tuple[int, int]] = []
    preview_window_created = False
    try:
        if not camera.isOpened():
            raise RuntimeError("Astra S did not open")
        arm.connect()
        arm_connected = True
        prepare_manual_collection_arm(arm)
        anchor_joints = np.asarray(arm.get_joint_deg(), dtype=float)
        anchor_xyz = arm.kin.forward_kinematics(anchor_joints[:5])[:3, 3]
        table_z = float(anchor_xyz[2]) if args.table_z is None else args.table_z
        max_base_radius = float(np.linalg.norm(anchor_xyz[:2])) if args.max_base_radius_m is None else args.max_base_radius_m
        print(
            f"anchor TCP={np.round(anchor_xyz, 4)}, table_z={table_z:.4f}m, "
            f"max base radius={max_base_radius:.4f}m, vertical tilt≤{args.max_vertical_tilt_deg:.1f}°"
        )
        print("자동 이동하지 않습니다. 그리퍼 끝단을 기준점에 수동으로 위치시키세요.")
        selected_pixel: list[tuple[int, int] | None] = [None]
        if args.preview:
            def on_click(event, x, y, _flags, _userdata):
                if event == cv2.EVENT_LBUTTONDOWN:
                    selected_pixel[0] = (x, y)

            cv2.namedWindow("handeye RGB")
            cv2.setMouseCallback("handeye RGB", on_click)
            preview_window_created = True
        index = 0
        while index < args.points:
            if args.preview:
                print(f"Point {index + 1}/{args.points}: TCP를 맞추고 클릭 후 Space")
                while True:
                    ret, color, _ = camera.read()
                    if not ret or color is None:
                        cv2.waitKey(30)
                        continue
                    display = color.copy()
                    for point in captured_pixels:
                        px = int(point[0] * color.shape[1] / 320)
                        py = int(point[1] * color.shape[0] / 240)
                        cv2.circle(display, (px, py), 6, (0, 255, 0), 2)
                    if selected_pixel[0] is not None:
                        cv2.circle(display, selected_pixel[0], 8, (0, 0, 255), 2)
                    cv2.putText(
                        display,
                        "move TCP on its reachable arc; click actual TCP, SPACE=capture, Q=quit",
                        (10, 25),
                        0,
                        0.65,
                        (0, 255, 255),
                        2,
                    )
                    cv2.imshow("handeye RGB", display)
                    key = cv2.waitKey(30) & 0xFF
                    if key == ord("q"):
                        raise KeyboardInterrupt
                    if key == ord(" ") and selected_pixel[0] is not None:
                        break
                u = int(selected_pixel[0][0] * 320 / color.shape[1])
                v = int(selected_pixel[0][1] * 240 / color.shape[0])
                selected_pixel[0] = None
            else:
                input(f"Point {index + 1}/{args.points}: 위치를 고정한 뒤 Enter...")
            depth = None
            for _ in range(30):
                depth = camera.read_raw_depth_mm()
                if depth is not None and np.any(depth > 0):
                    break
                time.sleep(0.1)
            if depth is None:
                raise RuntimeError("Astra depth frame unavailable")
            if args.preview:
                pass
            elif args.manual_pixel:
                pixel = input("그리퍼 TCP 중심의 depth 픽셀을 u,v 형식으로 입력: ")
                try:
                    u, v = (int(value.strip()) for value in pixel.split(","))
                except ValueError as exc:
                    raise RuntimeError("픽셀 형식은 u,v 이어야 합니다") from exc
                if not (0 <= u < depth.shape[1] and 0 <= v < depth.shape[0]):
                    raise RuntimeError("입력 픽셀이 depth 프레임 범위를 벗어났습니다")
            else:
                u, v = auto_select_depth_pixel(depth)
            depth_mm = float(depth[round(v), round(u)])
            if depth_mm <= 0:
                resolved = resolve_depth_pixel(depth, u, v, allow_fallback=args.allow_depth_fallback)
                if resolved is None:
                    print(
                        "선택한 TCP 픽셀의 depth가 0입니다. "
                        "같은 보정점에서 depth가 보이는 픽셀을 다시 선택하세요."
                    )
                    continue
                u, v = resolved
                depth_mm = float(depth[v, u])
                print(f"TCP 픽셀 depth=0, 인접 유효 픽셀로 보정: u={u}, v={v}")
            captured_pixels.append((u, v))
            print(f"자동 선택 depth 픽셀: u={u}, v={v}, depth={depth_mm:.0f}mm")
            xyz_mm = openni2.convert_depth_to_world(camera.depth_stream, u, v, depth_mm)
            camera_points.append(tuple(float(value) / 1000.0 for value in xyz_mm))
            joints = np.asarray(arm.get_joint_deg(), dtype=float)
            tool_pose = arm.kin.forward_kinematics(joints[:5])
            base_xyz = tool_pose[:3, 3]
            if not is_tcp_at_table_height(float(base_xyz[2]), table_z=table_z, tolerance=args.table_z_tolerance):
                print(
                    f"TCP Z={base_xyz[2]:.4f}m is outside table height "
                    f"{table_z:.4f}±{args.table_z_tolerance:.4f}m; point rejected"
                )
                captured_pixels.pop()
                camera_points.pop()
                continue
            if not is_sample_within_anchor_envelope(
                base_xyz, anchor_xyz, max_base_radius_m=max_base_radius,
            ):
                print(
                    "sample exceeds maximum base radius; point rejected"
                )
                captured_pixels.pop()
                camera_points.pop()
                continue
            if not is_vertical_grasp_pose(tool_pose, max_tilt_deg=args.max_vertical_tilt_deg):
                print("gripper is not vertically downward within tilt limit; point rejected")
                captured_pixels.pop()
                camera_points.pop()
                continue
            base_points.append(tuple(float(value) for value in base_xyz))
            joint_points.append(joints)
            print(f"  camera={camera_points[-1]} base={base_points[-1]}")
            index += 1
        args.output_dir.mkdir(parents=True, exist_ok=True)
        np.save(args.output_dir / "camera_points.npy", np.asarray(camera_points))
        np.save(args.output_dir / "base_points.npy", np.asarray(base_points))
        joints_array = validate_joint_points(np.asarray(joint_points), expected_count=args.points)
        np.save(args.output_dir / "joint_points.npy", joints_array)
        print(f"saved {args.output_dir / 'camera_points.npy'}")
        print(f"saved {args.output_dir / 'base_points.npy'}")
        print(f"saved {args.output_dir / 'joint_points.npy'}")
    finally:
        if arm_connected:
            arm.disconnect()
        camera.release()
        if preview_window_created:
            cv2.destroyWindow("handeye RGB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
