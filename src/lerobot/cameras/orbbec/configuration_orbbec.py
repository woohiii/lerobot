"""Configuration for an Orbbec Astra S camera accessed through OpenNI2."""

from dataclasses import dataclass
from pathlib import Path

from ..configs import CameraConfig, ColorMode


@CameraConfig.register_subclass("orbbec")
@dataclass
class OrbbecCameraConfig(CameraConfig):
    """Capture synchronized RGB and metric depth from an Orbbec Astra S.

    The Astra S exposes no ``/dev/video*`` device. It must be opened through
    OpenNI2 and the Orbbec SDK's bundled driver. Depth is stored as ``uint16``
    millimetres and is nearest-neighbour resized to the RGB resolution so the
    LeRobot ``<camera>`` / ``<camera>_depth`` features have matching shapes.
    """

    fps: int = 30
    width: int = 640
    height: int = 480
    depth_width: int = 320
    depth_height: int = 240
    use_rgb: bool = True
    use_depth: bool = True
    color_mode: ColorMode = ColorMode.RGB
    warmup_s: float = 1.0
    preview: bool = False
    openni2_redist_dir: Path | None = None

    def __post_init__(self) -> None:
        self.color_mode = ColorMode(self.color_mode)
        if not self.use_rgb and not self.use_depth:
            raise ValueError("At least one of `use_rgb` or `use_depth` must be enabled.")
        if min(self.fps, self.width, self.height, self.depth_width, self.depth_height) <= 0:
            raise ValueError("Camera dimensions and fps must be positive.")
