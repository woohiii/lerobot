"""Production seams for Gemini, Astra RGB-D, and the existing SO-101 safety stack."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .models import CameraIntrinsics, ObjectRoi, RgbdFrame


class ObjectNotDetectedError(RuntimeError):
    """Raised when Gemini cannot identify the requested object."""


@dataclass
class GeminiRoiDetector:
    """Adapt the existing Gemini detector to the new half-open ROI contract."""

    detector: Callable[[np.ndarray, str], Any] | None = None

    def detect(self, bgr_image: np.ndarray, prompt: str) -> ObjectRoi:
        detector = self.detector or self._existing_detector
        result = detector(bgr_image, prompt)
        if result is None:
            raise ObjectNotDetectedError(f"Gemini found no object for {prompt!r}")
        bbox = result.bbox if hasattr(result, "bbox") else result
        if len(bbox) != 4:
            raise ValueError("Gemini detector must return x, y, width, height")
        x, y, width, height = (int(round(float(value))) for value in bbox)
        return ObjectRoi(x, y, x + width, y + height)

    @staticmethod
    def _existing_detector(image: np.ndarray, prompt: str) -> Any:
        from custom_scripts.vision_pick_place.task_red_cube_to_bin.perception_zeroshot import detect_zeroshot

        return detect_zeroshot(image, prompt)


@dataclass
class RgbdCameraAdapter:
    """Read one registered RGB-D frame from an existing Astra camera object."""

    camera: Any
    intrinsics: CameraIntrinsics
    registration_verified: bool = False

    def read(self) -> RgbdFrame:
        if not self.registration_verified:
            raise RuntimeError("Astra RGB-depth registration is not verified")
        raw = self.camera.read()
        if isinstance(raw, tuple):
            _, bgr, _ = raw
        else:
            bgr = raw
        depth_mm = self.camera.read_raw_depth_mm()
        if bgr is None or depth_mm is None:
            raise RuntimeError("Astra returned an incomplete RGB-D frame")
        rgb = cv2.cvtColor(np.asarray(bgr), cv2.COLOR_BGR2RGB)
        depth = np.asarray(depth_mm)
        if depth.ndim != 2:
            raise RuntimeError("Astra RGB and depth frames are not aligned")
        if rgb.shape[:2] != depth.shape:
            depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        return RgbdFrame(rgb=rgb, depth=depth.astype(np.uint16, copy=False), intrinsics=self.intrinsics)


class SO101ArmAdapter:
    """Map the generic executor calls onto the existing protected SO-101 arm."""

    def __init__(self, arm: Any):
        self.arm = arm
        self._validated_retreat_waypoint: tuple[float, float, float] | None = None

    def move_to(self, point: tuple[float, float, float]) -> None:
        self.arm.move_to_xyz_converge(point)

    def open_gripper(self) -> None:
        self.arm.send_joint_deg(self._with_gripper(100.0))

    def close_gripper(self) -> None:
        self.arm.send_joint_deg(self._with_gripper(0.0))

    def retreat(self) -> None:
        if self._validated_retreat_waypoint is None:
            raise RuntimeError("no validated retreat waypoint is available")
        self.arm.move_to_xyz_converge(self._validated_retreat_waypoint)

    def set_validated_retreat_waypoint(self, waypoint: tuple[float, float, float]) -> None:
        """Store the retreat waypoint produced by left-arm preflight."""
        point = tuple(float(value) for value in waypoint)
        if len(point) != 3 or not np.isfinite(point).all():
            raise ValueError("validated retreat waypoint must contain three finite coordinates")
        self._validated_retreat_waypoint = point

    def verify_grasp(self) -> bool:
        """Conservative local check: gripper state is readable and closed."""
        joints = np.asarray(self.arm.get_joint_deg(), dtype=float)
        return bool(joints.size >= 6 and np.isfinite(joints).all() and joints[-1] < 95.0)

    def verify_pose(self) -> bool:
        """Ensure the arm still reports a finite six-value state after lifting."""
        joints = np.asarray(self.arm.get_joint_deg(), dtype=float)
        return bool(joints.size >= 6 and np.isfinite(joints).all())

    def _with_gripper(self, value: float) -> np.ndarray:
        joints = np.asarray(self.arm.get_joint_deg(), dtype=float).copy()
        joints[-1] = value
        return joints
