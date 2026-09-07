"""Live Astra S IR publisher, for the click-on-IR pick-and-place task.

Same role as astra_s_live.py (RGB+Depth publisher) but for the IR stream
instead: the user asked to click the object on Astra's *infrared* view, not
RGB, so the click pipeline needs an IR-pixel <-> robot-xy homography, which
in turn needs a live published IR frame to calibrate and click against.

IR-only, no depth: confirmed live on this Astra S unit that depth.start()
raises ONI_STATUS_ERROR whenever the IR stream is already running (this
device's structured-light depth needs exclusive use of the IR sensor - the
two cannot run together). So there is no per-object height estimate here;
descent falls back to the existing fixed-TABLE_Z + contact/collision-
detection safety net that the rest of this codebase already treats as the
real backstop regardless of whether a depth estimate is available.

Only ONE process may hold the Astra S device open at a time - same single-
owner constraint as astra_s_live.py/camera_hub.py - so this replaces
astra_s_live.py as the Astra owner (stop that script before running this
one).

Run standalone in ~/lerobot_song_venv (needs GUI opencv for imshow unless
headless). ASTRA_IR_HUB_HEADLESS=1 skips the window, same convention as
astra_s_live.py's ASTRA_LIVE_HEADLESS.
"""

import contextlib
import os
import time

import cv2
import numpy as np
from orbbec_color_camera import DEFAULT_OPENNI2_REDIST_DIR
from primesense import openni2

from camera_utils import ASTRA_IR_FRAME_PATH

IR_W, IR_H = 640, 480  # the only mode confirmed live to actually start() on this unit


def normalized_gray(raw: np.ndarray) -> np.ndarray:
    """Per-frame robust contrast stretch, same convention as the other IR viewers."""
    valid = raw[raw > 0]
    if not valid.size:
        return np.zeros(raw.shape, dtype=np.uint8)
    lo, hi = np.percentile(valid, (2, 98))
    return np.clip((raw.astype(np.float32) - lo) * 255.0 / max(hi - lo, 1.0), 0, 255).astype(np.uint8)


def atomic_write(path: str, frame) -> None:
    root, ext = os.path.splitext(path)
    tmp = f"{root}.tmp{ext}"
    cv2.imwrite(tmp, frame)
    os.replace(tmp, path)


def main() -> None:
    headless = os.environ.get("ASTRA_IR_HUB_HEADLESS") == "1"

    openni2.initialize(str(DEFAULT_OPENNI2_REDIST_DIR))
    device = openni2.Device.open_any()
    ir = device.create_ir_stream()
    if ir is None:
        print("[astra_s_ir_hub] Astra S IR 스트림을 열 수 없습니다.")
        device.close()
        return

    try:
        with contextlib.suppress(Exception):
            ir.configure_mode(IR_W, IR_H, 30, openni2.PIXEL_FORMAT_GRAY16)
        with contextlib.suppress(Exception):
            ir.set_mirroring_enabled(False)
        ir.start()

        if not headless:
            cv2.namedWindow("Astra S - IR", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Astra S - IR", 960, 720)
            print("[astra_s_ir_hub] 실행 중 - 'q' 또는 ESC로 종료.")
        else:
            print("[astra_s_ir_hub] 실행 중 (headless, ASTRA_IR_HUB_HEADLESS=1) - Ctrl-C로 종료.")

        while True:
            ir_frame = ir.read_frame()
            ir_raw = np.frombuffer(bytes(ir_frame.get_buffer_as_uint16()), dtype=np.uint16).reshape(ir_frame.height, ir_frame.width)
            ir_gray = normalized_gray(ir_raw)

            atomic_write(ASTRA_IR_FRAME_PATH, ir_gray)  # publish native-res 8-bit IR, not display-resized

            if headless:
                time.sleep(0.01)
                continue

            vis = ir_gray.copy()
            cv2.putText(vis, "ASTRA IR", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 255, 2)
            cv2.imshow("Astra S - IR", vis)
            if cv2.waitKey(15) & 0xFF in (27, ord("q")):
                break
    finally:
        with contextlib.suppress(Exception):
            ir.stop()
        device.close()
        if not headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
