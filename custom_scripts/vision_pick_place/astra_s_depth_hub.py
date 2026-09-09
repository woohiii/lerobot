#!/usr/bin/env python3
"""Depth-first Astra S publisher for ``astra_s_stream_supervisor.py``.

Unlike ``astra_s_live.py``, depth is read and written by this process's main
thread.  Thus a native OpenNI2 hang stops the file mtime instead of repeatedly
publishing the last cached array.  RGB is deliberately best-effort and lives
in a separate thread: a color hang cannot prevent fresh depth publication.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading

import cv2
import numpy as np
from primesense import openni2

from camera_utils import ASTRA_DEPTH_MM_PATH, ASTRA_RGB_FRAME_PATH
from orbbec_color_camera import DEFAULT_OPENNI2_REDIST_DIR
from task_trash_to_bin import config as trash_config
from task_trash_to_bin.perception import RGBD_PREVIEW_SCALE

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


def _write_json(path: str, data: dict[str, bool]) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)


def _configure_registration(device) -> bool:
    mode = openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR
    try:
        supported = device.is_image_registration_mode_supported(mode)
    except Exception as exc:
        print(f"[astra_s_depth_hub] depth-to-color registration 지원 여부 확인 실패: {exc}", flush=True)
        return False
    if not supported:
        print("[astra_s_depth_hub] depth-to-color registration 미지원 - FOV fallback 사용", flush=True)
        return False
    try:
        device.set_image_registration_mode(mode)
        actual_mode = device.get_image_registration_mode()
        active = actual_mode == mode
    except Exception as exc:
        print(f"[astra_s_depth_hub] depth-to-color registration 설정 실패: {exc}", flush=True)
        return False
    print(
        f"[astra_s_depth_hub] depth-to-color registration {'활성' if active else '비활성'} "
        f"(요청={mode}, 실제={actual_mode})",
        flush=True,
    )
    if active:
        print(
            "[astra_s_depth_hub] registration은 좌표 정합만 수행합니다. RGB FOV가 더 넓으면 가장자리 depth=0은 하드웨어상 정상입니다.",
            flush=True,
        )
    try:
        device.set_depth_color_sync_enabled(True)
        print("[astra_s_depth_hub] depth/color 동기화 활성 요청 완료", flush=True)
    except Exception as exc:
        print(f"[astra_s_depth_hub] depth/color 동기화 설정 실패: {exc}", flush=True)
    return active


def _log_stream_mode(label: str, stream) -> None:
    try:
        mode = stream.get_video_mode()
        print(
            f"[astra_s_depth_hub] {label} 실제 모드: {mode.resolutionX}x{mode.resolutionY}@{mode.fps}",
            flush=True,
        )
    except Exception as exc:
        print(f"[astra_s_depth_hub] {label} 실제 모드 확인 실패: {exc}", flush=True)


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
            try:
                stream.configure_mode(COLOR_W, COLOR_H, 30, openni2.PIXEL_FORMAT_RGB888)
            except Exception as exc:
                print(f"[astra_s_depth_hub] RGB 모드 요청 실패: {exc}", flush=True)
            with contextlib.suppress(Exception):
                stream.set_mirroring_enabled(False)
            stream.start()
            _log_stream_mode("RGB", stream)
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
    # 단독 카메라 디버깅 때만 hub 미리보기 창을 연다.
    hub_preview = os.environ.get("ASTRA_HUB_PREVIEW") == "1"
    openni2.initialize(str(DEFAULT_OPENNI2_REDIST_DIR))
    device = openni2.Device.open_any()
    depth = device.create_depth_stream()
    if depth is None:
        raise RuntimeError("Astra S depth stream is unavailable")
    registration_active = _configure_registration(device)
    _write_json(trash_config.ASTRA_DEPTH_REGISTRATION_STATUS_PATH, {"registered": registration_active})
    color = _ColorCache(device)
    last_rgb_id = 0
    latest_rgb: np.ndarray | None = None
    try:
        try:
            depth.configure_mode(DEPTH_W, DEPTH_H, 30, openni2.PIXEL_FORMAT_DEPTH_1_MM)
        except Exception as exc:
            print(f"[astra_s_depth_hub] depth 모드 요청 실패: {exc}", flush=True)
        with contextlib.suppress(Exception):
            depth.set_mirroring_enabled(False)
        depth.start()
        _log_stream_mode("depth", depth)
        if hub_preview:
            print("[astra_s_depth_hub] depth publish 및 RGB/depth 미리보기 중 - 'q' 또는 ESC로 종료", flush=True)
        else:
            print("[astra_s_depth_hub] depth publish 중 (미리보기 비활성, ASTRA_HUB_PREVIEW=1로 활성화)", flush=True)
        while True:
            frame = depth.read_frame()  # intentionally main-thread: parent detects a native wedge
            depth_mm = np.frombuffer(bytes(frame.get_buffer_as_uint16()), dtype=np.uint16).reshape(frame.height, frame.width)
            _write_npy(ASTRA_DEPTH_MM_PATH, depth_mm)
            rgb, last_rgb_id = color.latest_after(last_rgb_id)
            if rgb is not None:
                latest_rgb = rgb
                _write_png(ASTRA_RGB_FRAME_PATH, rgb)
            if hub_preview and latest_rgb is not None:
                depth_vis = cv2.normalize(depth_mm, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)
                if depth_color.shape[:2] != latest_rgb.shape[:2]:
                    depth_color = cv2.resize(depth_color, (latest_rgb.shape[1], latest_rgb.shape[0]))
                rgb_display = cv2.resize(
                    latest_rgb, None, fx=RGBD_PREVIEW_SCALE, fy=RGBD_PREVIEW_SCALE, interpolation=cv2.INTER_NEAREST
                )
                depth_display = cv2.resize(
                    depth_color,
                    None,
                    fx=RGBD_PREVIEW_SCALE,
                    fy=RGBD_PREVIEW_SCALE,
                    interpolation=cv2.INTER_NEAREST,
                )
                cv2.imshow("Astra S RGB + Depth", np.hstack((rgb_display, depth_display)))
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
    finally:
        color.stop()
        with contextlib.suppress(Exception):
            depth.stop()
        device.close()
        if hub_preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
