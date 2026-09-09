"""Low-speed, hover-only validation for a provisional hand-eye calibration."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def validate_calibration_rms(rms_error_m: float, *, max_rms_m: float) -> None:
    if not np.isfinite(rms_error_m) or rms_error_m > max_rms_m:
        raise ValueError(f"RMS {rms_error_m:.6f} m exceeds hover limit {max_rms_m:.6f} m")


def validate_hold_mode(*, execute: bool, hold_position: bool) -> None:
    if hold_position and not execute:
        raise ValueError("--hold-position requires --execute")


def compute_hover_target(
    camera_point_xyz: tuple[float, float, float],
    camera_to_base: np.ndarray,
    *,
    hover_offset_m: float,
) -> tuple[float, float, float]:
    point = np.asarray([*camera_point_xyz, 1.0], dtype=float)
    transform = np.asarray(camera_to_base, dtype=float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("camera_to_base must be a finite 4x4 matrix")
    if not np.isfinite(point).all() or not np.isfinite(hover_offset_m) or hover_offset_m <= 0:
        raise ValueError("camera point and hover offset must be finite; offset must be positive")
    base = transform @ point
    if not np.allclose(base[3], 1.0, atol=1e-5):
        raise ValueError("camera_to_base returned an invalid homogeneous point")
    base[2] += hover_offset_m
    return tuple(float(value) for value in base[:3])


def _check_workspace(point: tuple[float, float, float], workspace: list[float] | None) -> None:
    if workspace is None:
        return
    if len(workspace) != 6 or not np.isfinite(workspace).all():
        raise ValueError("workspace must be six finite bounds: xmin xmax ymin ymax zmin zmax")
    x, y, z = point
    xmin, xmax, ymin, ymax, zmin, zmax = workspace
    if not (xmin <= x <= xmax and ymin <= y <= ymax and zmin <= z <= zmax):
        raise ValueError(f"hover target {point} is outside workspace {workspace}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--camera-point", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    parser.add_argument("--port", required=True)
    parser.add_argument("--hover-offset-m", type=float, default=0.10)
    parser.add_argument("--max-rms-m", type=float, default=0.03)
    parser.add_argument(
        "--workspace",
        type=float,
        nargs=6,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
    )
    parser.add_argument(
        "--execute", action="store_true", help="move the arm to hover after MOVE_HOVER confirmation"
    )
    parser.add_argument(
        "--hold-position",
        action="store_true",
        help="keep the connection and torque enabled at hover until Ctrl+C",
    )
    args = parser.parse_args(argv)
    try:
        validate_hold_mode(execute=args.execute, hold_position=args.hold_position)
        payload = json.loads(args.calibration.read_text(encoding="utf-8"))
        rms = float(payload["rms_error_m"])
        validate_calibration_rms(rms, max_rms_m=args.max_rms_m)
        hover = compute_hover_target(
            tuple(args.camera_point), payload["camera_to_base"], hover_offset_m=args.hover_offset_m
        )
        _check_workspace(hover, args.workspace)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    print(f"calibration RMS: {rms:.6f} m")
    print(f"camera point: {tuple(args.camera_point)}")
    print(f"hover base target: {hover}")
    if not args.execute:
        print("dry hover plan: no hardware command was issued")
        return 0
    if input("Type MOVE_HOVER to move only to the hover point: ").strip() != "MOVE_HOVER":
        print("aborted: confirmation did not match")
        return 2

    vision_dir = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(vision_dir / "task_trash_to_bin"))
    sys.path.insert(0, str(vision_dir))
    from task_trash_to_bin.kinematics import SOArm101

    arm = SOArm101(port=args.port)
    connected = False
    holding = False
    try:
        arm.connect()
        connected = True
        print(f"current base TCP: {tuple(float(v) for v in arm.gripper_xyz())}")
        arm.move_to_xyz(hover, steps=40, step_delay_s=0.10, enforce_cap=False, stall_check=True)
        print(f"reached hover TCP: {tuple(float(v) for v in arm.gripper_xyz())}")
        if args.hold_position:
            holding = True
            print("holding hover position with torque enabled; press Ctrl+C to stop")
            while True:
                time.sleep(1.0)
    except KeyboardInterrupt:
        print("hover hold interrupted")
    finally:
        if connected and not holding:
            arm.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
