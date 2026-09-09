"""CLI for solving and saving the Astra-to-SO-101 hand-eye transform."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .calibration import CalibrationError, solve_rigid_transform


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("camera_points", type=Path, help="N x 3 camera-frame .npy file")
    parser.add_argument("base_points", type=Path, help="N x 3 base-frame .npy file")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-rms-m", type=float, default=0.005)
    args = parser.parse_args(argv)
    try:
        if not np.isfinite(args.max_rms_m) or args.max_rms_m <= 0:
            raise CalibrationError("--max-rms-m must be positive and finite")
        result = solve_rigid_transform(np.load(args.camera_points), np.load(args.base_points))
        if result.rms_error_m > args.max_rms_m:
            raise CalibrationError(
                f"RMS error {result.rms_error_m:.6f} m exceeds threshold {args.max_rms_m:.6f} m"
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {"camera_to_base": result.transform.tolist(), "rms_error_m": result.rms_error_m},
                indent=2,
            )
            + "\n"
        )
    except (CalibrationError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"saved {args.output} (RMS={result.rms_error_m:.6f} m)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
