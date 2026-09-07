"""Record the visible jaw-center pixel for wrist-camera visual servoing.

Camera only: this script opens /dev/video4, never opens a robot serial port,
and never sends a motor command.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import cv2

DEVICE_INDEX = 4  # USB 2.0 PC Cam video node; /dev/video5 is its metadata node.
WIDTH, HEIGHT, FPS = 640, 480, 30
WINDOW = "Wrist jaw-center calibration - camera only"
OUTPUT = Path(__file__).resolve().parent / "wrist_jaw_center.json"


def main() -> int:
    """Show the wrist stream and save one user-selected jaw-center pixel."""
    camera = cv2.VideoCapture(DEVICE_INDEX, cv2.CAP_V4L2)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    camera.set(cv2.CAP_PROP_FPS, FPS)
    if not camera.isOpened():
        raise RuntimeError(f"Cannot open wrist camera /dev/video{DEVICE_INDEX}")
    clicked: list[tuple[int, int]] = []

    def on_click(event, x, y, _flags, _userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked[:] = [(x, y)]

    try:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, 960, 720)
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                continue
            cv2.imshow(WINDOW, frame)
            cv2.waitKey(1)  # force native Qt window creation before mouse callback
            cv2.setMouseCallback(WINDOW, on_click)
            canvas = frame.copy()
            if clicked:
                cv2.drawMarker(canvas, clicked[0], (0, 255, 255), cv2.MARKER_CROSS, 22, 2)
            cv2.putText(canvas, "Click center between jaw tips; SPACE saves. q/ESC cancels.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 2)
            cv2.imshow(WINDOW, canvas)
            key = cv2.waitKey(25) & 0xFF
            if key in (27, ord("q")):
                return 0
            if key == ord(" ") and clicked:
                OUTPUT.write_text(json.dumps({
                    "created_utc": datetime.now(UTC).isoformat(),
                    "mode": "CAMERA_ONLY_NO_ROBOT_PORT",
                    "device": "/dev/video4",
                    "resolution": [WIDTH, HEIGHT],
                    "jaw_center_px": list(clicked[0]),
                }, indent=2) + "\n")
                print(f"[saved] {OUTPUT}: jaw_center_px={clicked[0]}")
                return 0
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
