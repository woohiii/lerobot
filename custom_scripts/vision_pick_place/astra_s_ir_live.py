"""Live infrared preview for the existing Orbbec Astra S OpenNI2 device."""

import contextlib

import cv2
import numpy as np
from orbbec_color_camera import DEFAULT_OPENNI2_REDIST_DIR
from primesense import openni2


def main():
    """Open only Astra's IR stream and render its 16-bit signal as grayscale."""
    openni2.initialize(str(DEFAULT_OPENNI2_REDIST_DIR))
    device = openni2.Device.open_any()
    stream = device.create_ir_stream()
    if stream is None:
        raise RuntimeError("Astra S IR stream is unavailable on this device/driver")
    try:
        with contextlib.suppress(Exception):
            stream.configure_mode(640, 480, 30, openni2.PIXEL_FORMAT_GRAY16)
        stream.set_mirroring_enabled(False)
        stream.start()
        cv2.namedWindow("Astra S - Infrared", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Astra S - Infrared", 960, 720)
        print("[astra_s_ir_live] IR 화면 실행 중 - q 또는 ESC로 종료")
        while True:
            frame = stream.read_frame()
            raw = np.frombuffer(bytes(frame.get_buffer_as_uint16()), dtype=np.uint16).reshape(frame.height, frame.width)
            valid = raw[raw > 0]
            if valid.size:
                lo, hi = np.percentile(valid, (2, 98))
                image = np.clip((raw.astype(np.float32) - lo) * 255.0 / max(hi - lo, 1.0), 0, 255).astype(np.uint8)
            else:
                image = np.zeros(raw.shape, dtype=np.uint8)
            cv2.putText(image, "ASTRA S INFRARED", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 255, 2)
            cv2.imshow("Astra S - Infrared", image)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        stream.stop()
        device.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
