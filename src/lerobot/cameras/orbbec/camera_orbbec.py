"""OpenNI2-backed Orbbec Astra S camera for LeRobot recording."""

import logging
import threading
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError

from ..camera import Camera
from ..configs import ColorMode
from .configuration_orbbec import OrbbecCameraConfig

logger = logging.getLogger(__name__)


class OrbbecCamera(Camera):
    """Read Astra S RGB and depth in a background thread.

    Exactly one process may own an Astra S. Close ``astra_s_live.py`` and its
    viewers before connecting this camera through ``lerobot-record``.
    """

    def __init__(self, config: OrbbecCameraConfig):
        super().__init__(config)
        self.config = config
        self.use_rgb = config.use_rgb
        self.use_depth = config.use_depth
        self.color_mode = config.color_mode
        self.preview = config.preview
        self._device: Any | None = None
        self._color_stream: Any | None = None
        self._depth_stream: Any | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._frame_event = threading.Event()
        self._lock = threading.Lock()
        self._color: NDArray[np.uint8] | None = None
        self._depth: NDArray[np.uint16] | None = None
        self._timestamp: float | None = None
        self._error: Exception | None = None

    @property
    def _redist_dir(self) -> Path:
        if self.config.openni2_redist_dir is not None:
            return self.config.openni2_redist_dir
        return Path(__file__).resolve().parents[4] / "custom_scripts/vision_pick_place/openni2_redist"

    @property
    def is_connected(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and self._error is None

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        return [{"name": "Orbbec Astra S", "type": "Orbbec", "id": "openni2"}]

    @check_if_already_connected
    def connect(self, warmup: bool = True) -> None:
        try:
            from primesense import openni2
        except ImportError as exc:
            raise ImportError("Install the `primesense` Python package to use an Orbbec Astra S.") from exc

        redist_dir = self._redist_dir
        if not (redist_dir / "libOpenNI2.so").is_file():
            raise FileNotFoundError(f"Orbbec OpenNI2 SDK not found: {redist_dir}")
        openni2.initialize(str(redist_dir))
        try:
            self._device = openni2.Device.open_any()
            self._color_stream = self._device.create_color_stream() if self.use_rgb else None
            self._depth_stream = self._device.create_depth_stream() if self.use_depth else None
            if self.use_rgb and self._color_stream is None:
                raise ConnectionError("Astra S did not expose an RGB stream.")
            if self.use_depth and self._depth_stream is None:
                raise ConnectionError("Astra S did not expose a depth stream.")
            if self._color_stream is not None:
                self._color_stream.configure_mode(
                    self.config.width, self.config.height, self.config.fps, openni2.PIXEL_FORMAT_RGB888
                )
                self._color_stream.set_mirroring_enabled(False)
            if self._depth_stream is not None:
                self._depth_stream.configure_mode(
                    self.config.depth_width,
                    self.config.depth_height,
                    self.config.fps,
                    openni2.PIXEL_FORMAT_DEPTH_1_MM,
                )
                self._depth_stream.set_mirroring_enabled(False)
            if self.use_rgb and self.use_depth:
                try:
                    self._device.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)
                    self._device.set_depth_color_sync_enabled(True)
                except Exception:
                    logger.warning(
                        "Astra S depth-to-color registration is unavailable; recording both streams anyway."
                    )
            if self._color_stream is not None:
                self._color_stream.start()
            if self._depth_stream is not None:
                self._depth_stream.start()
        except Exception:
            self._close_device()
            raise

        self._stop_event.clear()
        self._frame_event.clear()
        self._error = None
        self._thread = threading.Thread(target=self._capture_loop, name="orbbec-astra-s", daemon=True)
        self._thread.start()
        if warmup and not self._frame_event.wait(timeout=max(5.0, self.config.warmup_s + 1.0)):
            self.disconnect()
            raise ConnectionError("Astra S did not provide frames during warmup.")
        if self._error is not None:
            error = self._error
            self.disconnect()
            raise ConnectionError("Astra S frame capture failed during warmup.") from error
        if self.preview:
            cv2.namedWindow("Astra S RGB (recording)", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Astra S depth mm (recording)", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Astra S RGB (recording)", 640, 480)
            cv2.resizeWindow("Astra S depth mm (recording)", 640, 480)
        logger.info(
            "Orbbec Astra S connected (%sx%s RGB, %sx%s depth).",
            self.width,
            self.height,
            self.width,
            self.height,
        )

    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                color = None
                depth = None
                if self._color_stream is not None:
                    frame = self._color_stream.read_frame()
                    raw = bytes(frame.get_buffer_as_uint8())
                    rgb = np.frombuffer(raw, dtype=np.uint8).reshape(frame.height, frame.width, 3)
                    color = (
                        rgb.copy()
                        if self.color_mode is ColorMode.RGB
                        else cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    )
                if self._depth_stream is not None:
                    frame = self._depth_stream.read_frame()
                    raw = bytes(frame.get_buffer_as_uint16())
                    depth = np.frombuffer(raw, dtype=np.uint16).reshape(frame.height, frame.width).copy()
                    if depth.shape != (self.height, self.width):
                        depth = cv2.resize(depth, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
                    depth = depth[..., np.newaxis]
                with self._lock:
                    if color is not None:
                        self._color = color
                    if depth is not None:
                        self._depth = depth
                    self._timestamp = time.monotonic()
                self._frame_event.set()
            except Exception as exc:
                if not self._stop_event.is_set():
                    self._error = exc
                return

    def _latest(self, depth: bool, max_age_ms: int) -> NDArray[Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self.__class__.__name__} is not connected.")
        with self._lock:
            frame = self._depth if depth else self._color
            timestamp = self._timestamp
            if frame is None or timestamp is None:
                raise RuntimeError("Astra S has not captured a frame yet.")
            if (time.monotonic() - timestamp) * 1000 > max_age_ms:
                raise TimeoutError("Latest Astra S frame is stale.")
            result = frame.copy()
        if self.preview:
            self._show_preview(result, depth=depth)
        return result

    def _show_preview(self, frame: NDArray[Any], depth: bool) -> None:
        if depth:
            depth_mm = frame[..., 0]
            normalized = np.clip(depth_mm.astype(np.float32), 350, 800)
            normalized = ((normalized - 350) * 255 / 450).astype(np.uint8)
            normalized[depth_mm == 0] = 0
            image = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
            image[depth_mm == 0] = 0
            cv2.imshow("Astra S depth mm (recording)", image)
        else:
            image = frame if self.color_mode is ColorMode.BGR else cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("Astra S RGB (recording)", image)
        cv2.waitKey(1)

    @check_if_not_connected
    def read(self) -> NDArray[Any]:
        self._frame_event.wait(timeout=10)
        return self._latest(depth=False, max_age_ms=10_000)

    @check_if_not_connected
    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        if not self._frame_event.wait(timeout=timeout_ms / 1000):
            raise TimeoutError("Timed out waiting for an Astra S RGB frame.")
        return self._latest(depth=False, max_age_ms=max(500, int(timeout_ms) + 100))

    @check_if_not_connected
    def read_latest(self, max_age_ms: int = 500) -> NDArray[Any]:
        return self._latest(depth=False, max_age_ms=max_age_ms)

    @check_if_not_connected
    def read_depth(self) -> NDArray[np.uint16]:
        self._frame_event.wait(timeout=10)
        return self._latest(depth=True, max_age_ms=10_000)

    @check_if_not_connected
    def async_read_depth(self, timeout_ms: float = 200) -> NDArray[np.uint16]:
        if not self._frame_event.wait(timeout=timeout_ms / 1000):
            raise TimeoutError("Timed out waiting for an Astra S depth frame.")
        return self._latest(depth=True, max_age_ms=max(500, int(timeout_ms) + 100))

    @check_if_not_connected
    def read_latest_depth(self, max_age_ms: int = 500) -> NDArray[np.uint16]:
        return self._latest(depth=True, max_age_ms=max_age_ms)

    def _close_device(self) -> None:
        for stream in (self._color_stream, self._depth_stream):
            if stream is not None:
                with suppress(Exception):
                    stream.stop()
        if self._device is not None:
            with suppress(Exception):
                self._device.close()
        self._color_stream = None
        self._depth_stream = None
        self._device = None

    def disconnect(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None
        self._close_device()
        if self.preview:
            cv2.destroyWindow("Astra S RGB (recording)")
            cv2.destroyWindow("Astra S depth mm (recording)")
