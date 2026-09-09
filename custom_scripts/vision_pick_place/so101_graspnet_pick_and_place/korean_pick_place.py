"""Preview-first Korean RGB-D pick-and-place command for the left SO-101."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .left_pick_executor import (
    PickPlaceWaypoints,
    build_and_validate_waypoints,
    build_waypoints,
    execute_validated_pick_place,
)
from .local_vlm import KoreanTaskVlm, OllamaVlmClient
from .models import CameraIntrinsics, RgbdFrame, TaskCommand
from .stable_localization import StablePose, localize_stable_roi

_MAX_RMS_M = 0.03
_MIN_VALID_DEPTH_RATIO = 0.5
_MAX_SPREAD_M = 0.015
_TABLE_Z_M = -0.04
_HOVER_HEIGHT_M = 0.08
_LIFT_HEIGHT_M = 0.10
_MAX_JOINT_DELTA_DEG = 10.0
_CONFIRMATION = "EXECUTE_PICK_PLACE"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--command",
        help="Korean pick-and-place command; if omitted, enter it after the camera preview appears",
    )
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--port", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preview", action="store_true")
    mode.add_argument("--execute", action="store_true")
    return parser


def _load_calibration(path: Path) -> tuple[np.ndarray, CameraIntrinsics | None]:
    document = json.loads(path.read_text())
    rms = document.get("rms_error_m")
    transform = np.asarray(document.get("camera_to_base"), dtype=float)
    if not isinstance(rms, (int, float)) or not np.isfinite(rms):
        raise ValueError("calibration needs a finite rms_error_m")
    if float(rms) > _MAX_RMS_M:
        raise ValueError(f"calibration RMS {rms:.6f} m exceeds {_MAX_RMS_M:.3f} m")
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("calibration camera_to_base must be a finite 4x4 matrix")
    intrinsics_document = document.get("camera_intrinsics")
    if intrinsics_document is None:
        return transform, None
    if not isinstance(intrinsics_document, dict):
        raise ValueError("calibration camera_intrinsics must be an object")
    try:
        intrinsics = CameraIntrinsics(**intrinsics_document)
    except TypeError as exc:
        raise ValueError("calibration camera_intrinsics is invalid") from exc
    return transform, intrinsics


def _collect_frames(camera: Any) -> tuple[RgbdFrame, RgbdFrame, RgbdFrame]:
    frames: list[RgbdFrame] = []
    deadline = time.monotonic() + 1.0
    while len(frames) < 3:
        frame = camera.read()
        if not isinstance(frame, RgbdFrame):
            raise ValueError("camera factory must provide read() -> RgbdFrame")
        if frame.capture_id is not None and not any(frame.capture_id == previous.capture_id for previous in frames):
            frames.append(frame)
            continue
        if time.monotonic() >= deadline:
            raise ValueError("timed out waiting for three distinct atomic RGB-D capture samples")
        time.sleep(0.02)
    return tuple(frames)  # type: ignore[return-value]


def _draw_preview(rgb: np.ndarray, depth: np.ndarray, command: TaskCommand, source: StablePose, destination: StablePose) -> np.ndarray:
    preview = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for item, pose, color in ((command.source, source, (0, 255, 0)), (command.destination, destination, (255, 0, 0))):
        x0, y0, x1, y1 = item.box_xyxy
        cv2.rectangle(preview, (x0, y0), (x1, y1), color, 2)
        xyz = ", ".join(f"{value:.3f}" for value in pose.base_xyz)
        cv2.putText(preview, f"{item.label}: {xyz} m", (x0, max(16, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    clipped = np.clip(depth, 300, 1500).astype(np.float32)
    depth_vis = cv2.applyColorMap(((clipped - 300) / 1200 * 255).astype(np.uint8), cv2.COLORMAP_JET)
    depth_vis[depth == 0] = 0
    depth_vis = cv2.resize(depth_vis, (preview.shape[1], preview.shape[0]))
    return np.hstack((preview, depth_vis))


def _json_plan(command: TaskCommand, source: StablePose, destination: StablePose, waypoints: PickPlaceWaypoints) -> str:
    def values(point: Sequence[float]) -> list[float]:
        return [float(value) for value in point]

    return json.dumps(
        {
            "dry plan": True,
            "source": {"label": command.source.label, "confidence": command.source.confidence, "camera_xyz": values(source.camera_xyz), "base_xyz": values(source.base_xyz), "stability_spread_m": source.max_spread_m},
            "destination": {"label": command.destination.label, "confidence": command.destination.confidence, "camera_xyz": values(destination.camera_xyz), "base_xyz": values(destination.base_xyz), "stability_spread_m": destination.max_spread_m},
            "source_camera_xyz": values(source.camera_xyz),
            "destination_base_xyz": values(destination.base_xyz),
            "waypoints": {name: values(getattr(waypoints, name)) for name in waypoints.__dataclass_fields__},
        },
        ensure_ascii=False,
        allow_nan=False,
    )


def _display(preview: np.ndarray, *, seconds: float = 3.0) -> None:
    if os.environ.get("DISPLAY"):
        cv2.imshow("Korean pick-place dry plan", preview)
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if cv2.waitKey(30) >= 0:
                break
        cv2.destroyWindow("Korean pick-place dry plan")


def _show_camera_preview(camera: Any, initial: RgbdFrame, *, seconds: float = 5.0) -> None:
    """Show live RGB-D frames briefly before asking for the user's command."""
    if not os.environ.get("DISPLAY"):
        print("camera preview unavailable (DISPLAY is not set)")
        return
    deadline = time.monotonic() + seconds
    frame = initial
    while time.monotonic() < deadline:
        rgb = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
        clipped = np.clip(frame.depth, 300, 1500).astype(np.float32)
        depth = cv2.applyColorMap(((clipped - 300) / 1200 * 255).astype(np.uint8), cv2.COLORMAP_JET)
        depth[frame.depth == 0] = 0
        cv2.imshow("Astra S RGB", rgb)
        cv2.imshow("Astra S Depth", depth)
        cv2.waitKey(1)
        try:
            candidate = camera.read()
            if isinstance(candidate, RgbdFrame):
                frame = candidate
        except Exception:
            pass


class _AstraRgbdCamera:
    """Registered Astra S frames using OpenNI SDK-owned depth calibration."""

    def __init__(self, intrinsics: CameraIntrinsics | None = None) -> None:
        from primesense import openni2

        from custom_scripts.vision_pick_place.orbbec_color_camera import ThreadedOrbbecRGBDCamera

        self._camera = ThreadedOrbbecRGBDCamera(width=640, height=480, fps=30)
        self._intrinsics = intrinsics
        if not self._camera.isOpened():
            self._camera.release()
            raise RuntimeError("Astra S did not open")
        device = self._camera.device
        if device is None or device.get_image_registration_mode() != openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR:
            self._camera.release()
            raise RuntimeError("Astra RGB-depth registration is not active; refusing metric localization")

    def read(self) -> RgbdFrame:
        # OpenNI streams can report opened before their first paired frames are
        # available.  Wait briefly instead of failing on that normal startup
        # race.
        deadline = time.monotonic() + 2.0
        while True:
            ready, bgr, depth, sequence, _timestamp_s = self._camera.read_rgbd_snapshot()
            if ready and bgr is not None and depth is not None and sequence is not None:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError("Astra S returned an incomplete RGB-D frame")
            time.sleep(0.05)
        rgb = cv2.cvtColor(np.asarray(bgr), cv2.COLOR_BGR2RGB)
        if depth.shape != rgb.shape[:2]:
            depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        provider = self._camera.sdk_xyz_provider()
        return RgbdFrame(
            rgb=rgb,
            depth=np.asarray(depth, dtype=np.uint16),
            intrinsics=self._intrinsics,
            xyz_provider=provider,
            capture_id=int(sequence),
        )

    def close(self) -> None:
        """Release the Astra streams after preview/execution setup."""
        self._camera.release()


def _default_camera_factory(intrinsics: CameraIntrinsics | None) -> _AstraRgbdCamera:
    return _AstraRgbdCamera(intrinsics)


def _default_arm_factory(*, port: str) -> Any:
    from .left_arm import LeftSOArm101

    return LeftSOArm101(port=port)


def main(
    argv: list[str] | None = None,
    *,
    camera_factory: Callable[[], Any] | None = None,
    vlm_factory: Callable[[], KoreanTaskVlm] | None = None,
    arm_factory: Callable[..., Any] | None = None,
) -> int:
    """Print a safe plan first; hardware is only constructed after confirmation."""
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        camera_to_base, intrinsics = _load_calibration(args.calibration)
        camera = camera_factory() if camera_factory is not None else _default_camera_factory(intrinsics)
        frames = _collect_frames(camera)
        if args.command is None:
            _show_camera_preview(camera, frames[-1])
            print("카메라 화면을 확인한 뒤 한국어 명령을 입력하세요.")
            args.command = input("명령: ").strip()
            if not args.command:
                raise ValueError("command must not be empty")
        vlm = (vlm_factory or OllamaVlmClient)()
        print("[1/4] RGB 화면에서 물체 탐지 중...", flush=True)
        detections = vlm.detect_objects(frames[0].rgb, "Detect visible pickable objects and destination trays only.")
        print(f"[2/4] 탐지 완료: {len(detections)}개 물체", flush=True)
        print("[3/4] 한국어 명령 해석 중...", flush=True)
        command = vlm.parse_korean_task(args.command, detections)
        print("[4/4] RGB-D 3차원 좌표 안정화 중...", flush=True)
        source = localize_stable_roi(
            frames,
            command.source.roi,
            camera_to_base,
            min_valid_depth_ratio=_MIN_VALID_DEPTH_RATIO,
            max_spread_m=_MAX_SPREAD_M,
        )
        destination = localize_stable_roi(
            frames,
            command.destination.roi,
            camera_to_base,
            min_valid_depth_ratio=_MIN_VALID_DEPTH_RATIO,
            max_spread_m=_MAX_SPREAD_M,
        )
        waypoints = build_waypoints(
            source.base_xyz,
            destination.base_xyz,
            hover_height_m=_HOVER_HEIGHT_M,
            lift_height_m=_LIFT_HEIGHT_M,
        )
        _display(_draw_preview(frames[0].rgb, frames[0].depth, command, source, destination))
        print(_json_plan(command, source, destination, waypoints))
        try:
            if args.preview:
                return 0
            if input(f"Type {_CONFIRMATION} to send this plan to the arm: ") != _CONFIRMATION:
                print("execution cancelled: exact confirmation was not provided")
                return 2
            arm = (arm_factory or _default_arm_factory)(port=args.port)
            arm.connect()
            try:
                validated = build_and_validate_waypoints(
                    arm,
                    source.base_xyz,
                    destination.base_xyz,
                    table_z=_TABLE_Z_M,
                    hover_height_m=_HOVER_HEIGHT_M,
                    lift_height_m=_LIFT_HEIGHT_M,
                    max_joint_delta_deg=_MAX_JOINT_DELTA_DEG,
                )
                execute_validated_pick_place(arm, validated)
            finally:
                arm.disconnect()
        finally:
            close = getattr(camera, "close", None)
            if callable(close):
                close()
        return 0
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
