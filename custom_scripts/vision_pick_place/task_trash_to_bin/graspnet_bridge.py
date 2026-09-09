"""Published Astra RGB-D 파일을 별도 GraspNet venv로 전달하는 얇은 브릿지."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path


GRASPNET_PYTHON = Path("/home/youngchan/third_party/graspnet-venv/bin/python")
GRASPNET_SCRIPT = Path("/home/youngchan/third_party/graspnet-baseline/graspnet_infer.py")


def infer_published_grasps(
    rgb_path: str | Path,
    depth_mm_path: str | Path,
    *,
    intrinsic_json: str | Path | None = None,
    top_k: int = 20,
    fx: float = 285.0,
    fy: float = 285.0,
    cx: float = 160.0,
    cy: float = 120.0,
) -> list[dict[str, object]]:
    """published RGB/depth 파일을 추론 subprocess에 넘기고 grasp 목록을 반환한다."""
    with tempfile.TemporaryDirectory(prefix="graspnet_bridge_") as temporary_dir:
        output_path = Path(temporary_dir) / "grasps.json"
        command = [
            str(GRASPNET_PYTHON),
            str(GRASPNET_SCRIPT),
            "--rgb",
            str(rgb_path),
            "--depth-mm",
            str(depth_mm_path),
            "--top-k",
            str(top_k),
            "--output",
            str(output_path),
        ]
        if intrinsic_json is not None:
            command.extend(("--intrinsic-json", str(intrinsic_json)))
        else:
            command.extend(("--fx", str(fx), "--fy", str(fy), "--cx", str(cx), "--cy", str(cy)))
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        if completed.stdout:
            print(f"[graspnet_bridge] {completed.stdout.strip()}")
        with open(output_path, encoding="utf-8") as stream:
            payload = json.load(stream)
    return list(payload["grasps"])


if __name__ == "__main__":
    import config

    # 이 fallback은 Astra 실측 intrinsic이 아니며, 나중에 반드시 실측 보정해야 한다.
    candidates = infer_published_grasps(config.ASTRA_RGB_FRAME_PATH, config.ASTRA_DEPTH_MM_PATH)
    scores = [float(candidate["score"]) for candidate in candidates]
    print(f"[graspnet_bridge] 후보 {len(candidates)}개, score 범위={min(scores):.6f}..{max(scores):.6f}")
