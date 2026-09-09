"""Jointly calibrate Astra camera-to-base pose and SO-101 TCP offset."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .calibration import CalibrationError, JointTcpCalibrationResult, solve_joint_tcp_transform


def save_joint_calibration(
    result: JointTcpCalibrationResult,
    output: Path,
    *,
    max_rms_m: float,
) -> None:
    if not np.isfinite(max_rms_m) or max_rms_m <= 0:
        raise ValueError("--max-rms-m must be positive and finite")
    if result.rms_error_m > max_rms_m:
        raise ValueError(f"RMS error {result.rms_error_m:.6f} m exceeds threshold {max_rms_m:.6f} m")
    payload = {
        "calibration_type": "joint_tcp",
        "camera_to_base": result.transform.tolist(),
        "tcp_offset_m": result.tcp_offset_m.tolist(),
        "rms_error_m": result.rms_error_m,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("camera_points", type=Path)
    parser.add_argument("joint_points", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-rms-m", type=float, default=0.005)
    args = parser.parse_args(argv)
    try:
        from .left_arm import build_left_kinematics

        result = solve_joint_tcp_transform(
            np.load(args.camera_points),
            np.load(args.joint_points),
            build_left_kinematics(),
        )
        save_joint_calibration(result, args.output, max_rms_m=args.max_rms_m)
    except (CalibrationError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"saved {args.output} (RMS={result.rms_error_m:.6f} m, TCP offset={result.tcp_offset_m.tolist()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
