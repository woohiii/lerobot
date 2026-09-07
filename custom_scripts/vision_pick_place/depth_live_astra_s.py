"""Live depth viewer for the Astra S (2bc5:0402), via OpenNI2 - NOT the Astra
Pro Plus this repo mostly targets. The Astra S is a single vendor-specific
device (no /dev/videoN node at all, cv2.VideoCapture can't open it), so it
needs the OpenNI2 bundle in openni2_redist/ (see orbbec_color_camera.py's
docstring for why the system libopenni2 doesn't work: no Orbbec driver).

Run standalone in ~/lerobot_song_venv (needs GUI opencv for imshow).

Depth read happens on its own thread so a slow/blocked read_frame() call
never freezes the displayed window - the main loop just polls whatever the
thread last decoded, same pattern as camera_hub.py's CameraWorker. (Yesterday's
depth_live.cpp version - for the Astra Pro Plus's OrbbecSDK vendor stream -
froze under concurrent RGB+wrist+robot USB load; that specific contention
doesn't apply here since the Astra S is the only thing using its own device,
but the threaded-read pattern is kept anyway so a live window is guaranteed.)

Colorizes to the observed per-frame min/max range (not a fixed span) - same
reasoning as depth_live.cpp: a fixed 0-5.12m range looks all-blue on a close
tabletop scene.
"""

import threading
import time

import cv2
import numpy as np
from primesense import openni2

from orbbec_color_camera import DEFAULT_OPENNI2_REDIST_DIR, _ensure_openni_initialized


class DepthWorker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.device = None
        self.stream = None
        self._lock = threading.Lock()
        self.vis = None
        self.stop_flag = False
        self.open_error = None
        self._opened_event = threading.Event()

    def run(self):
        try:
            _ensure_openni_initialized(DEFAULT_OPENNI2_REDIST_DIR)
            self.device = openni2.Device.open_any()
            self.stream = self.device.create_depth_stream()
            if self.stream is None:
                raise RuntimeError("create_depth_stream() -> None")
            try:
                self.stream.configure_mode(640, 480, 30, openni2.PIXEL_FORMAT_DEPTH_1_MM)
            except Exception:
                pass
            try:
                self.stream.set_mirroring_enabled(False)
            except Exception:
                pass
            self.stream.start()
        except Exception as e:
            self.open_error = e
            self._opened_event.set()
            return
        self._opened_event.set()

        while not self.stop_flag:
            try:
                oni_frame = self.stream.read_frame()
            except Exception:
                continue
            h, w = oni_frame.height, oni_frame.width
            raw = bytes(oni_frame.get_buffer_as_uint16())
            depth_mm = np.frombuffer(raw, dtype=np.uint16).reshape((h, w))

            valid = depth_mm > 0
            if valid.any():
                vmin = int(depth_mm[valid].min())
                vmax = int(depth_mm[valid].max())
            else:
                vmin, vmax = 0, 0

            if vmax > vmin:
                clipped = np.clip(depth_mm, vmin, vmax).astype(np.float32)
                norm = ((clipped - vmin) * (255.0 / (vmax - vmin))).astype(np.uint8)
            else:
                norm = np.zeros_like(depth_mm, dtype=np.uint8)
            colorized = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
            colorized[~valid] = (0, 0, 0)
            cv2.putText(
                colorized, f"range: {vmin}-{vmax}mm", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
            )
            with self._lock:
                self.vis = colorized

    def isOpened(self):
        self._opened_event.wait(timeout=5.0)
        return self.open_error is None and self.stream is not None

    def latest(self):
        with self._lock:
            return None if self.vis is None else self.vis.copy()

    def stop(self):
        self.stop_flag = True


def main():
    worker = DepthWorker()
    worker.start()
    if not worker.isOpened():
        print(f"[depth_live_astra_s] 뎁스 스트림을 열 수 없습니다: {worker.open_error}")
        worker.stop()
        return

    cv2.namedWindow("Astra S - Depth (auto-ranged)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Astra S - Depth (auto-ranged)", 640, 480)
    print("[depth_live_astra_s] 실행 중 - 'q' 또는 ESC로 종료.")
    try:
        while True:
            frame = worker.latest()
            if frame is not None:
                cv2.imshow("Astra S - Depth (auto-ranged)", frame)
            key = cv2.waitKey(15) & 0xFF
            if key == ord("q") or key == 27:
                break
            if frame is None:
                time.sleep(0.01)
    finally:
        worker.stop()
        worker.join(timeout=2)
        if worker.stream is not None:
            try:
                worker.stream.stop()
            except Exception:
                pass
        if worker.device is not None:
            try:
                worker.device.close()
            except Exception:
                pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
