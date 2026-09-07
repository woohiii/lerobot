#!/usr/bin/env python3
"""Depth-first Astra S publisher for ``astra_s_stream_supervisor.py``.

Unlike ``astra_s_live.py``, depth is read and written by this process's main
thread.  Thus a native OpenNI2 hang stops the file mtime instead of repeatedly
publishing the last cached array.  RGB is deliberately best-effort and lives
in a separate thread: a color hang cannot prevent fresh depth publication.
"""

from __future__ import annotations

import contextlib
import os
import threading

import cv2
import numpy as np
from primesense import openni2

from camera_utils import ASTRA_DEPTH_MM_PATH, ASTRA_RGB_FRAME_PATH
from orbbec_color_camera import DEFAULT_OPENNI2_REDIST_DIR

DEPTH_W, DEPTH_H = 320, 240
COLOR_W, COLOR_H = 320, 240


def _write_npy(path: str, frame: np.ndarray) -> None:
    tmp = f"{path}.tmp.npy"
    np.save(tmp, frame)
    os.replace(tmp, path)


def _write_png(path: str, frame: np.ndarray) -> None:
    root, ext = os.path.splitext(path)
    tmp = f"{root}.tmp{ext}"
    cv2.imwrite(tmp, frame)
    os.replace(tmp, path)


class _ColorCache:
    def __init__(self, device) -> None:
        self._device, self._lock, self._running = device, threading.Lock(), True
        self._frame: np.ndarray | None = None
        self._id = 0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            stream = self._device.create_color_stream()
            if stream is None:
                return
            with contextlib.suppress(Exception):
                stream.configure_mode(COLOR_W, COLOR_H, 30, openni2.PIXEL_FORMAT_RGB888)
            with contextlib.suppress(Exception):
                stream.set_mirroring_enabled(False)
            stream.start()
            while self._running:
                frame = stream.read_frame()
                rgb = np.frombuffer(bytes(frame.get_buffer_as_uint8()), dtype=np.uint8).reshape(frame.height, frame.width, 3)
                with self._lock:
                    self._frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    self._id += 1
        except Exception as exc:
            print(f"[astra_s_depth_hub] RGB disabled: {exc}", flush=True)
        finally:
            with contextlib.suppress(Exception):
                stream.stop()  # type: ignore[possibly-undefined]

    def latest_after(self, previous_id: int) -> tuple[np.ndarray | None, int]:
        with self._lock:
            if self._frame is None or self._id == previous_id:
                return None, previous_id
            return self._frame.copy(), self._id

    def stop(self) -> None:
        self._running = False


def main() -> None:
    openni2.initialize(str(DEFAULT_OPENNI2_REDIST_DIR))
    device = openni2.Device.open_any()
    depth = device.create_depth_stream()
    if depth is None:
        raise RuntimeError("Astra S depth stream is unavailable")
    color = _ColorCache(device)
    last_rgb_id = 0
    try:
        with contextlib.suppress(Exception):
            depth.configure_mode(DEPTH_W, DEPTH_H, 30, openni2.PIXEL_FORMAT_DEPTH_1_MM)
        with contextlib.suppress(Exception):
            depth.set_mirroring_enabled(False)
        depth.start()
        print("[astra_s_depth_hub] publishing depth; RGB is best-effort", flush=True)
        while True:
            frame = depth.read_frame()  # intentionally main-thread: parent detects a native wedge
            depth_mm = np.frombuffer(bytes(frame.get_buffer_as_uint16()), dtype=np.uint16).reshape(frame.height, frame.width)
            _write_npy(ASTRA_DEPTH_MM_PATH, depth_mm)
            rgb, last_rgb_id = color.latest_after(last_rgb_id)
            if rgb is not None:
                _write_png(ASTRA_RGB_FRAME_PATH, rgb)
    finally:
        color.stop()
        with contextlib.suppress(Exception):
            depth.stop()
        device.close()


if __name__ == "__main__":
    main()
