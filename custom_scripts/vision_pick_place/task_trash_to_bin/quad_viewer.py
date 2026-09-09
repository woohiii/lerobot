"""Read-only 2x2 viewer for Astra RGB/depth and both wrist-camera publishes.

This script never opens a camera device.  The shared Astra and camera hubs own
the devices and atomically publish their latest frames; this viewer only reads
those published files.  Run from this directory with ``python3 quad_viewer.py``.
Press q or Esc to close the viewer.
"""

from __future__ import annotations

import os
import time

import cv2
import numpy as np

import config
from perception import PublishedFrameSource


# camera_hub.py publishes the second (left) wrist camera at this distinct
# shared path.  This right-arm-only task config intentionally has no left-arm
# override, unlike the bimanual sibling task.
WRIST_LEFT_FRAME_PATH = "/tmp/vsp_wrist_left.png"

# Match the proven sibling viewer's useful tabletop depth visualization range.
DEPTH_MIN_MM = 350
DEPTH_MAX_MM = 800


def load_depth_vis(
    depth_path: str = config.ASTRA_DEPTH_MM_PATH,
    stale_timeout_s: float = config.FRAME_STALE_TIMEOUT_S,
) -> np.ndarray | None:
    """Return a colorized fresh Astra millimetre-depth frame, if available."""
    if not os.path.exists(depth_path) or (time.time() - os.path.getmtime(depth_path)) >= stale_timeout_s:
        return None
    try:
        depth_mm = np.load(depth_path)
    except (OSError, ValueError):
        return None

    clipped = np.clip(depth_mm, DEPTH_MIN_MM, DEPTH_MAX_MM).astype(np.float32)
    normalized = ((clipped - DEPTH_MIN_MM) / (DEPTH_MAX_MM - DEPTH_MIN_MM) * 255.0).astype(np.uint8)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_JET)


def _panel(frame: np.ndarray | None, label: str) -> np.ndarray:
    """Resize a frame into one labelled grid cell, or draw its waiting state."""
    if frame is None:
        panel = np.zeros((config.FRAME_H, config.FRAME_W, 3), dtype=np.uint8)
        cv2.putText(panel, "waiting...", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        panel = cv2.resize(frame, (config.FRAME_W, config.FRAME_H))
    cv2.putText(panel, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return panel


def _published_files_available() -> bool:
    """Avoid opening a blank GUI when the required camera hubs are not running."""
    required_paths = (
        config.ASTRA_RGB_FRAME_PATH,
        config.ASTRA_DEPTH_MM_PATH,
        config.WRIST_FRAME_PATH,
        WRIST_LEFT_FRAME_PATH,
    )
    missing_paths = [path for path in required_paths if not os.path.exists(path)]
    if missing_paths:
        print("파일 없음 - 카메라 허브 필요:")
        for path in missing_paths:
            print(f"  {path}")
        return False
    return True


def main() -> None:
    if not _published_files_available():
        return

    astra_rgb = PublishedFrameSource(config.ASTRA_RGB_FRAME_PATH)
    astra_ir = PublishedFrameSource(config.ASTRA_IR_FRAME_PATH)
    wrist_right = PublishedFrameSource(config.WRIST_FRAME_PATH)
    wrist_left = PublishedFrameSource(WRIST_LEFT_FRAME_PATH)

    try:
        while True:
            _, astra_rgb_frame = astra_rgb.read()
            _, wrist_right_frame = wrist_right.read()
            _, wrist_left_frame = wrist_left.read()

            depth_frame = load_depth_vis()
            depth_label = "Astra Depth"
            # Astra structured-light depth and IR are mutually exclusive;
            # show the configured IR publish as the proven fallback.
            if depth_frame is None:
                _, depth_frame = astra_ir.read()
                depth_label = "Astra IR (depth fallback)"

            top = cv2.hconcat([_panel(astra_rgb_frame, "Astra RGB"), _panel(depth_frame, depth_label)])
            bottom = cv2.hconcat(
                [_panel(wrist_right_frame, "Wrist Right"), _panel(wrist_left_frame, "Wrist Left")]
            )
            cv2.imshow("Astra | Depth | Wrist", cv2.vconcat([top, bottom]))

            key = cv2.waitKey(30) & 0xFF
            if key == ord("q") or key == 27:
                break
    except KeyboardInterrupt:
        # Allow Ctrl+C (and bounded smoke tests) to use the same clean-up
        # path as q/Esc without emitting a traceback.
        pass
    finally:
        astra_rgb.release()
        astra_ir.release()
        wrist_right.release()
        wrist_left.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
