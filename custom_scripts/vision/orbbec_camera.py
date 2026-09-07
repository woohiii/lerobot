"""
Astra S (Orbbec) color camera helper via OpenNI2.

Astra S is a PrimeSense(PS1080)-based structured-light camera. It is NOT a
UVC webcam (no /dev/videoX node) - it must be accessed through OpenNI2.

Requires:
  - system packages: libopenni2-0, libopenni-sensor-primesense0 (already installed)
  - pip package `openni` inside the active venv/conda env
  - a udev rule granting the user rw access to the USB device (see
    ../../docs or the 99-orbbec-astra.rules file next to this script)
"""
import atexit
import time
import numpy as np

# The apt-packaged libopenni2-0 / libopenni-sensor-primesense0 (PS1080 driver)
# does NOT recognize Orbbec-branded devices (vendor id 2bc5) - it only knows
# the older PrimeSense/ASUS Xtion ids, so Device.open_any() silently finds
# zero devices even with correct USB permissions.
# Orbbec's own OpenNI2 SDK ships a replacement driver (liborbbec.so +
# orbbec.ini) that recognizes 2bc5/0402 (Astra S) and friends - use that.
import os
OPENNI2_REDIST_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "OpenNI2_SDK", "sdk_extracted", "sdk", "libs",
)

_initialized = False


def _ensure_openni2():
    global _initialized
    from openni import openni2
    if not _initialized:
        openni2.initialize(OPENNI2_REDIST_DIR)
        _initialized = True
    return openni2


class OrbbecColorCamera:
    """Minimal RGB-only reader for the Astra S color stream."""

    def __init__(self):
        # NOTE: do NOT retry Device.open_any() in a loop within one process -
        # after a failed/timed-out open the openni ctypes bindings are left in
        # a corrupt state (observed: "double free or corruption" on 2nd try).
        # If the open is flaky, retry at the process level instead (see
        # open_with_retries() below / red_cube_detect.py's --retries flag).
        openni2 = _ensure_openni2()
        self._openni2 = openni2
        t0 = time.time()
        try:
            self.device = openni2.Device.open_any()
        except Exception as e:
            raise RuntimeError(
                f"Could not open the Orbbec device via OpenNI2 ({time.time() - t0:.1f}s).\n"
                "  - Is it plugged in? (check: lsusb | grep -i orbbec)\n"
                "  - Do you have permission? (check: ls -la /dev/bus/usb/00X/00Y ; "
                "should be mode 666 - see the udev rule)\n"
                "  - usbmon trace shows intermittent silent packet loss during the\n"
                "    handshake (some control transfers get no response at all and\n"
                "    time out after 5s) - this points to a flaky USB link "
                "(cable/port), not a driver/permission bug.\n"
                f"Original error: {e}"
            ) from e
        self.color_stream = self.device.create_color_stream()
        if self.color_stream is None:
            raise RuntimeError("Device has no color sensor?")
        self.color_stream.start()
        atexit.register(self.close)

    def read_bgr(self):
        """Return one frame as an OpenCV-style HxWx3 uint8 BGR array."""
        frame = self.color_stream.read_frame()
        buf = frame.get_buffer_as_uint8()
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(frame.height, frame.width, 3)
        # OpenNI2 color stream default pixel format is RGB888 -> flip to BGR for OpenCV
        return arr[:, :, ::-1].copy()

    def close(self):
        try:
            self.color_stream.stop()
        except Exception:
            pass
        try:
            self.device.close()
        except Exception:
            pass
