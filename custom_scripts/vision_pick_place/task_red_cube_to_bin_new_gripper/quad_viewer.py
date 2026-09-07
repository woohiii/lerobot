"""Read-only 2x2 live viewer for the Astra and both wrist-camera publishes.

This script never opens a camera device; astra_s_live.py and camera_hub.py
already own those devices and atomically publish the frames this viewer reads.
Run it with `uv run python3 quad_viewer.py`. GUI-capable OpenCV is required
for the cv2 window.
"""

from __future__ import annotations

import os
import time

import cv2
import numpy as np

import config
from perception import PublishedFrameSource


# Matches zeroshot_viewer.py / astra_s_live.py's displayed depth range.
DEPTH_MIN_MM = 350
DEPTH_MAX_MM = 800


def load_depth_vis(
    depth_path: str = config.ASTRA_DEPTH_MM_PATH,
    stale_timeout_s: float = config.FRAME_STALE_TIMEOUT_S,
):
    """Return the existing Astra mm-depth visualization, or None if unavailable."""
    if not os.path.exists(depth_path) or (time.time() - os.path.getmtime(depth_path)) >= stale_timeout_s:
        return None
    try:
        depth_mm = np.load(depth_path)
    except (OSError, ValueError):
        return None
    clipped = np.clip(depth_mm, DEPTH_MIN_MM, DEPTH_MAX_MM).astype(np.float32)
    norm = ((clipped - DEPTH_MIN_MM) / (DEPTH_MAX_MM - DEPTH_MIN_MM) * 255.0).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_JET)


def _panel(frame: np.ndarray | None, label: str) -> np.ndarray:
    if frame is None:
        panel = np.zeros((config.FRAME_H, config.FRAME_W, 3), dtype=np.uint8)
        cv2.putText(panel, "waiting...", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        panel = cv2.resize(frame, (config.FRAME_W, config.FRAME_H))
    cv2.putText(panel, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return panel


def main() -> None:
    astra_rgb = PublishedFrameSource(config.ASTRA_RGB_FRAME_PATH)
    astra_ir = PublishedFrameSource(config.ASTRA_IR_FRAME_PATH)
    wrist_right = PublishedFrameSource(config.WRIST_FRAME_PATH)
    wrist_left = PublishedFrameSource(config.LEFT_OVERRIDES["WRIST_FRAME_PATH"])

    try:
        while True:
            _, astra_rgb_frame = astra_rgb.read()
            _, wrist_right_frame = wrist_right.read()
            _, wrist_left_frame = wrist_left.read()
            depth_frame = load_depth_vis()
            # IR is the supported Astra fallback when depth cannot start or
            # stalls.  It is intentionally not run together with depth: this
            # model's structured-light depth owns the IR sensor.
            secondary_frame = depth_frame
            secondary_label = "Astra Depth"
            if secondary_frame is None:
                _, secondary_frame = astra_ir.read()
                secondary_label = "Astra IR (depth fallback)"

            top = cv2.hconcat([_panel(astra_rgb_frame, "Astra RGB"), _panel(secondary_frame, secondary_label)])
            bottom = cv2.hconcat([_panel(wrist_right_frame, "Wrist Right"), _panel(wrist_left_frame, "Wrist Left")])
            cv2.imshow("Astra | Depth | Wrist", cv2.vconcat([top, bottom]))

            key = cv2.waitKey(30) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        astra_rgb.release()
        astra_ir.release()
        wrist_right.release()
        wrist_left.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
