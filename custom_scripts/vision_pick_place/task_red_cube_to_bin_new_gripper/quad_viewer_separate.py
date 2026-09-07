"""Read-only live viewer with four independent camera windows.

This script never opens a camera device. astra_s_live.py and camera_hub.py
own the devices and atomically publish the files read here. Press q or Esc in
any viewer window to close this viewer only.

Run from this directory with `uv run python3 quad_viewer_separate.py`.
"""

from __future__ import annotations

import cv2

import config
from perception import PublishedFrameSource
from quad_viewer import _panel, load_depth_vis


WINDOWS = (
    ("Astra RGB", (0, 0)),
    ("Astra Depth", (660, 0)),
    ("Wrist Right", (0, 530)),
    ("Wrist Left", (660, 530)),
)


def main() -> None:
    astra_rgb = PublishedFrameSource(config.ASTRA_RGB_FRAME_PATH)
    astra_ir = PublishedFrameSource(config.ASTRA_IR_FRAME_PATH)
    wrist_right = PublishedFrameSource(config.WRIST_FRAME_PATH)
    wrist_left = PublishedFrameSource(config.LEFT_OVERRIDES["WRIST_FRAME_PATH"])

    for title, position in WINDOWS:
        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow(title, *position)

    try:
        while True:
            _, astra_rgb_frame = astra_rgb.read()
            _, wrist_right_frame = wrist_right.read()
            _, wrist_left_frame = wrist_left.read()

            cv2.imshow("Astra RGB", _panel(astra_rgb_frame, "Astra RGB"))
            depth_frame = load_depth_vis()
            if depth_frame is None:
                _, depth_frame = astra_ir.read()
                label = "Astra IR (depth fallback)"
            else:
                label = "Astra Depth"
            cv2.imshow("Astra Depth", _panel(depth_frame, label))
            cv2.imshow("Wrist Right", _panel(wrist_right_frame, "Wrist Right"))
            cv2.imshow("Wrist Left", _panel(wrist_left_frame, "Wrist Left"))

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
