"""Camera-only preview of wrist visual-servo error; no robot actuation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

SAFE_DIR = Path(__file__).resolve().parent
TASK_DIR = SAFE_DIR.parent / "task_red_cube_to_bin_new_gripper"
sys.path.insert(0, str(TASK_DIR))

from perception import detect_black_bin, detect_red_cube  # noqa: E402, I001


WINDOW = "Wrist visual-servo preview - camera only"
DEVICE_INDEX = 4


def main() -> int:
    """Overlay detection-to-jaw pixel error for manual visual validation."""
    jaw_center = tuple(json.loads((SAFE_DIR / "wrist_jaw_center.json").read_text())["jaw_center_px"])
    camera = cv2.VideoCapture(DEVICE_INDEX, cv2.CAP_V4L2)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not camera.isOpened():
        raise RuntimeError("Cannot open /dev/video4")
    target_name, detector = "red cube", detect_red_cube
    try:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, 960, 720)
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                continue
            det = detector(frame)
            canvas = frame.copy()
            cv2.drawMarker(canvas, jaw_center, (0, 255, 255), cv2.MARKER_CROSS, 22, 2)
            if det is not None:
                point = (round(det.cx), round(det.cy))
                dx, dy = point[0] - jaw_center[0], point[1] - jaw_center[1]
                cv2.circle(canvas, point, 7, (0, 255, 0), -1)
                cv2.line(canvas, jaw_center, point, (0, 255, 0), 2)
                cv2.putText(canvas, f"error: dx={dx}px dy={dy}px", (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 2)
            else:
                cv2.putText(canvas, "target not detected", (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 255), 2)
            cv2.putText(canvas, f"target={target_name}; 1=red cube, 2=black box, q/ESC=close", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2)
            cv2.imshow(WINDOW, canvas)
            key = cv2.waitKey(25) & 0xFF
            if key in (27, ord("q")):
                return 0
            if key == ord("1"):
                target_name, detector = "red cube", detect_red_cube
            if key == ord("2"):
                target_name, detector = "black box", detect_black_bin
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
