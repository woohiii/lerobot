"""Click a visible object in Astra RGB and print its camera XYZ point."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def select_valid_depth(depth: np.ndarray, u: int, v: int) -> tuple[int, int, float]:
    if not (0 <= u < depth.shape[1] and 0 <= v < depth.shape[0]):
        raise ValueError("pixel is outside depth frame")
    value = float(depth[v, u])
    if value <= 0:
        raise ValueError("selected pixel has zero depth")
    return u, v, value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/object_point"))
    args = parser.parse_args(argv)
    vision_dir = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(vision_dir / "task_trash_to_bin"))
    sys.path.insert(0, str(vision_dir))
    from orbbec_color_camera import ThreadedOrbbecRGBDCamera
    from primesense import openni2

    camera = ThreadedOrbbecRGBDCamera(width=640, height=480, fps=30)
    selected: list[tuple[int, int] | None] = [None]
    window_created = False
    try:
        if not camera.isOpened():
            raise RuntimeError("Astra S did not open")

        def on_click(event, x, y, _flags, _userdata):
            if event == cv2.EVENT_LBUTTONDOWN:
                selected[0] = (x, y)

        cv2.namedWindow("object RGB")
        cv2.setMouseCallback("object RGB", on_click)
        window_created = True
        print("박스 중심을 클릭하고 Space를 누르세요. 종료는 q입니다.")
        while True:
            ret, color, _ = camera.read()
            if not ret or color is None:
                time.sleep(0.05)
                continue
            display = color.copy()
            if selected[0] is not None:
                cv2.circle(display, selected[0], 8, (0, 0, 255), 2)
            cv2.putText(display, "click box; SPACE=capture; Q=quit", (10, 25), 0, 0.7, (0, 255, 255), 2)
            cv2.imshow("object RGB", display)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                return 2
            if key != ord(" ") or selected[0] is None:
                continue
            depth = camera.read_raw_depth_mm()
            if depth is None:
                print("depth frame unavailable; retry")
                continue
            u = int(selected[0][0] * depth.shape[1] / color.shape[1])
            v = int(selected[0][1] * depth.shape[0] / color.shape[0])
            try:
                u, v, depth_mm = select_valid_depth(depth, u, v)
            except ValueError as exc:
                print(f"{exc}; 다른 박스 픽셀을 선택하세요")
                selected[0] = None
                continue
            xyz_mm = openni2.convert_depth_to_world(camera.depth_stream, u, v, depth_mm)
            xyz_m = tuple(float(value) / 1000.0 for value in xyz_mm)
            args.output_dir.mkdir(parents=True, exist_ok=True)
            np.save(args.output_dir / "camera_point.npy", np.asarray(xyz_m))
            print(f"camera point (m): {xyz_m}")
            print(f"saved {args.output_dir / 'camera_point.npy'}")
            return 0
    finally:
        camera.release()
        if window_created:
            cv2.destroyWindow("object RGB")


if __name__ == "__main__":
    raise SystemExit(main())
