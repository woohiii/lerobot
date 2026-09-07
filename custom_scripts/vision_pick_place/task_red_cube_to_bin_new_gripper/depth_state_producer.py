"""Live depth_0..depth_15 producer for run_towel_policy_bimanual.py's
--depth-state-path.

Training's depth_0..depth_15 (see so101-bimanual-teleop/depth_to_state_feature.py)
came from `decode_video_frames(..., is_depth=True)` on the recorded
`observation.images.astra_depth` video, block-mean-pooled 4x4. That video was
losslessly encoded (`DepthEncoderConfig.extra_options = {"x265-params":
"lossless=1"}`) from `quantize_depth(raw_mm_frame)` using the untouched
`depth_encoder_defaults()` (record_bimanual_with_depth.py never overrides
`depth_encoder`) - so decode reproduces `quantize_depth`'s output exactly,
and this script only needs to replicate `quantize_depth` + the same
block-mean, applied directly to the live raw-mm array instead of a video
round-trip.

The recording pipeline's raw depth input was itself
`PublishedDepthSource(ASTRA_DEPTH_MM_PATH).read()` (see
record_bimanual_with_depth.py's add_depth_to_robot) - the exact same
publish path used live - so no unit conversion is needed either.

Run this alongside run_towel_policy_bimanual.py, pointing
--depth-state-path at this script's --out-path:
    uv run python depth_state_producer.py --out-path /tmp/vsp_depth_state.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # vision_pick_place/ for camera_utils

from camera_utils import ASTRA_DEPTH_MM_PATH, PublishedDepthSource  # noqa: E402
from lerobot.configs.video import depth_encoder_defaults  # noqa: E402
from lerobot.datasets.depth_utils import quantize_depth  # noqa: E402

GRID = 4  # must match depth_to_state_feature.py's GRID


def compute_depth_state(depth_mm: np.ndarray, grid: int = GRID) -> dict[str, float]:
    """Reproduce training's depth_0..depth_{grid*grid-1} from a raw mm frame."""
    cfg = depth_encoder_defaults()
    quantized = quantize_depth(
        depth_mm,
        depth_min=cfg.depth_min,
        depth_max=cfg.depth_max,
        shift=cfg.shift,
        use_log=cfg.use_log,
        video_backend=None,  # return the uint16 code array directly, skip av.VideoFrame packing
    )
    h, w = quantized.shape
    assert h % grid == 0 and w % grid == 0, f"depth frame {h}x{w} not divisible by grid={grid}"
    bh, bw = h // grid, w // grid
    block_mean = quantized.astype(np.float32).reshape(grid, bh, grid, bw).mean(axis=(1, 3)).reshape(-1)
    return {f"depth_{i}": float(v) for i, v in enumerate(block_mean)}


def atomic_write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data))
    os.replace(tmp, path)


def self_check() -> None:
    rng = np.random.default_rng(0)
    depth_mm = rng.integers(300, 900, size=(240, 320), dtype=np.uint16)
    state = compute_depth_state(depth_mm)
    assert set(state) == {f"depth_{i}" for i in range(16)}
    values = np.array(list(state.values()))
    assert np.isfinite(values).all()
    # DEPTH_QMAX=4095 caps any single quantized pixel; a 4x4 block-mean over
    # this sensor's near-range (300-900mm) inputs stays well under that and
    # comfortably inside the ~0-708.775 range measured on real training data.
    assert (values >= 0).all() and (values < 4095).all(), values
    assert (values < 900).all(), f"expected near-range block means, got {values}"

    # Cross-check against a manual, unvectorized block-mean on the same
    # quantized array (independent of compute_depth_state's reshape order).
    from lerobot.configs.video import depth_encoder_defaults as _defaults
    from lerobot.datasets.depth_utils import quantize_depth as _quantize

    cfg = _defaults()
    quantized = _quantize(
        depth_mm, depth_min=cfg.depth_min, depth_max=cfg.depth_max, shift=cfg.shift, use_log=cfg.use_log,
        video_backend=None,
    ).astype(np.float32)
    bh, bw = 240 // GRID, 320 // GRID
    for gy in range(GRID):
        for gx in range(GRID):
            block = quantized[gy * bh : (gy + 1) * bh, gx * bw : (gx + 1) * bw]
            expected = float(block.mean())
            got = state[f"depth_{gy * GRID + gx}"]
            assert abs(expected - got) < 1e-3, (gy, gx, expected, got)
    print("PASS: compute_depth_state matches manual per-block mean, values in expected range")
    print(f"sample depth state: {state}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-path", type=Path, default=Path("/tmp/vsp_depth_state.json"))
    parser.add_argument("--depth-mm-path", default=ASTRA_DEPTH_MM_PATH)
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()

    if args.self_check:
        self_check()
        return 0

    source = PublishedDepthSource(args.depth_mm_path)
    period = 1.0 / args.hz
    print(f"[depth_state_producer] reading {args.depth_mm_path} -> writing {args.out_path} at {args.hz}Hz")
    while True:
        depth_mm = source.read()
        if depth_mm is None:
            print("[depth_state_producer] no fresh depth frame; is astra_s_live.py running?")
        else:
            atomic_write_json(args.out_path, compute_depth_state(depth_mm))
        time.sleep(period)


if __name__ == "__main__":
    raise SystemExit(main())
