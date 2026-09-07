"""Two-way live view: Astra S depth and follower wrist camera.

Astra S cannot stream IR and Depth at the same time on this driver (IR
read_frame() blocks forever once Depth is also started) -- run
astra_s_ir_live.py separately (and close it first) if you need the IR view.
"""

import contextlib

import cv2
import numpy as np
from orbbec_color_camera import DEFAULT_OPENNI2_REDIST_DIR
from primesense import openni2


def depth_view(depth_mm: np.ndarray) -> np.ndarray:
    """Render 350--800 mm depth as a color panel."""
    clipped = np.clip(depth_mm, 350, 800).astype(np.float32)
    scaled = ((clipped - 350) * 255.0 / 450.0).astype(np.uint8)
    image = cv2.applyColorMap(scaled, cv2.COLORMAP_JET)
    image[depth_mm == 0] = (0, 0, 0)
    return image


def panel(image: np.ndarray, label: str) -> np.ndarray:
    """Resize a panel to VGA display and add an unobtrusive title."""
    image = cv2.resize(image, (640, 480))
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    return image


def main() -> None:
    """Own Astra for depth and open the separate wrist UVC camera."""
    openni2.initialize(str(DEFAULT_OPENNI2_REDIST_DIR))
    device = openni2.Device.open_any()
    depth, wrist = device.create_depth_stream(), cv2.VideoCapture(4, cv2.CAP_V4L2)
    if depth is None or not wrist.isOpened():
        raise RuntimeError("depth/wrist camera could not all be opened")
    try:
        with contextlib.suppress(Exception):
            depth.configure_mode(320, 240, 30, openni2.PIXEL_FORMAT_DEPTH_1_MM)
        with contextlib.suppress(Exception):
            depth.set_mirroring_enabled(False)
        depth.start()
        cv2.namedWindow("Astra Depth | Wrist", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Astra Depth | Wrist", 960, 480)
        print("[two-view] DEPTH | WRIST 실행 중 - q 또는 ESC로 종료")
        while True:
            depth_frame = depth.read_frame()
            ok, wrist_frame = wrist.read()
            if not ok:
                continue
            depth_raw = np.frombuffer(bytes(depth_frame.get_buffer_as_uint16()), dtype=np.uint16).reshape(depth_frame.height, depth_frame.width)
            combined = cv2.hconcat([panel(depth_view(depth_raw), "ASTRA DEPTH"), panel(wrist_frame, "FOLLOWER WRIST")])
            cv2.imshow("Astra Depth | Wrist", combined)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        with contextlib.suppress(Exception):
            depth.stop()
        device.close()
        wrist.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
