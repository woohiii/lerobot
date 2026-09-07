"""Three-way live view: Astra S infrared, depth, and follower wrist camera."""

import contextlib

import cv2
import numpy as np
from orbbec_color_camera import DEFAULT_OPENNI2_REDIST_DIR
from primesense import openni2


def normalized_gray(raw: np.ndarray) -> np.ndarray:
    """Render a 16-bit sensor plane with per-frame robust contrast."""
    valid = raw[raw > 0]
    if not valid.size:
        return np.zeros(raw.shape, dtype=np.uint8)
    lo, hi = np.percentile(valid, (2, 98))
    return np.clip((raw.astype(np.float32) - lo) * 255.0 / max(hi - lo, 1.0), 0, 255).astype(np.uint8)


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
    """Own Astra once for IR+depth and open the separate wrist UVC camera."""
    openni2.initialize(str(DEFAULT_OPENNI2_REDIST_DIR))
    device = openni2.Device.open_any()
    ir, depth, wrist = device.create_ir_stream(), device.create_depth_stream(), cv2.VideoCapture(4, cv2.CAP_V4L2)
    if ir is None or depth is None or not wrist.isOpened():
        raise RuntimeError("IR/depth/wrist camera could not all be opened")
    try:
        with contextlib.suppress(Exception):
            ir.configure_mode(320, 240, 30, openni2.PIXEL_FORMAT_GRAY16)
        with contextlib.suppress(Exception):
            depth.configure_mode(320, 240, 30, openni2.PIXEL_FORMAT_DEPTH_1_MM)
        with contextlib.suppress(Exception):
            ir.set_mirroring_enabled(False)
            depth.set_mirroring_enabled(False)
        ir.start()
        depth.start()
        cv2.namedWindow("Astra IR | Astra Depth | Wrist", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Astra IR | Astra Depth | Wrist", 1440, 480)
        print("[three-view] IR | DEPTH | WRIST 실행 중 - q 또는 ESC로 종료")
        while True:
            ir_frame, depth_frame = ir.read_frame(), depth.read_frame()
            ok, wrist_frame = wrist.read()
            if not ok:
                continue
            ir_raw = np.frombuffer(bytes(ir_frame.get_buffer_as_uint16()), dtype=np.uint16).reshape(ir_frame.height, ir_frame.width)
            depth_raw = np.frombuffer(bytes(depth_frame.get_buffer_as_uint16()), dtype=np.uint16).reshape(depth_frame.height, depth_frame.width)
            combined = cv2.hconcat([panel(normalized_gray(ir_raw), "ASTRA INFRARED"), panel(depth_view(depth_raw), "ASTRA DEPTH"), panel(wrist_frame, "FOLLOWER WRIST")])
            cv2.imshow("Astra IR | Astra Depth | Wrist", combined)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        with contextlib.suppress(Exception):
            ir.stop()
            depth.stop()
        device.close()
        wrist.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
