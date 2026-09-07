"""Depth | RGB | wrist 3-way live preview for the Astra Pro Plus setup.

Astra Pro Plus splits into two separate USB devices (unlike the Astra S, which
was one device doing everything): a plain UVC "USB 2.0 Camera" for RGB, and a
vendor-specific "ORBBEC Depth Sensor" that only OpenNI2 can talk to. So unlike
orbbec_color_camera.py's ThreadedOrbbecRGBDCamera (built for the Astra S, where
one openni2.Device.open_any() handle gives you both streams), depth and RGB
here have to come from two independent sources: OpenNI2 for depth,
cv2.VideoCapture for RGB.
"""

import argparse
import time

import cv2
import numpy as np
from primesense import openni2

from orbbec_color_camera import DEFAULT_OPENNI2_REDIST_DIR

DEPTH_MIN_MM = 350
DEPTH_MAX_MM = 800


def depth_to_vis(depth_mm: np.ndarray) -> np.ndarray:
    clipped = np.clip(depth_mm, DEPTH_MIN_MM, DEPTH_MAX_MM).astype(np.float32)
    norm = ((clipped - DEPTH_MIN_MM) / (DEPTH_MAX_MM - DEPTH_MIN_MM) * 255.0).astype(np.uint8)
    vis = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    vis[depth_mm == 0] = (0, 0, 0)
    return vis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb", type=int, default=6, help="/dev/videoN for the Astra Pro Plus RGB module")
    parser.add_argument("--wrist", type=int, default=4, help="/dev/videoN for the wrist camera")
    args = parser.parse_args()

    openni2.initialize(str(DEFAULT_OPENNI2_REDIST_DIR))
    device = openni2.Device.open_any()
    depth_stream = device.create_depth_stream()
    if depth_stream is None:
        print("[preview_all] 뎁스 스트림을 열 수 없습니다.")
        raise SystemExit(1)
    try:
        depth_stream.configure_mode(640, 480, 30, openni2.PIXEL_FORMAT_DEPTH_1_MM)
    except Exception as e:
        print(f"[preview_all] 뎁스 모드 설정 실패, 기본 모드로 진행: {e}")
    depth_stream.start()

    cap_rgb = cv2.VideoCapture(args.rgb)
    cap_wrist = cv2.VideoCapture(args.wrist)
    if not cap_rgb.isOpened():
        print(f"[preview_all] RGB(/dev/video{args.rgb})를 열 수 없습니다.")
        raise SystemExit(1)
    if not cap_wrist.isOpened():
        print(f"[preview_all] 손목캠(/dev/video{args.wrist})을 열 수 없습니다.")
        raise SystemExit(1)

    print("Depth | RGB | Wrist 미리보기. 'q'로 종료.")
    t_report = time.time()
    try:
        while True:
            oni_frame = depth_stream.read_frame()
            dh, dw = oni_frame.height, oni_frame.width
            depth_raw = bytes(oni_frame.get_buffer_as_uint16())
            depth_mm = np.frombuffer(depth_raw, dtype=np.uint16).reshape((dh, dw))
            depth_vis = depth_to_vis(depth_mm)

            ret_rgb, rgb_frame = cap_rgb.read()
            ret_wrist, wrist_frame = cap_wrist.read()
            if not ret_rgb or not ret_wrist:
                continue

            if time.time() - t_report >= 1.0:
                valid = depth_mm[depth_mm > 0]
                if valid.size > 0:
                    print(f"[depth] min={int(valid.min())}mm max={int(valid.max())}mm median={int(np.median(valid))}mm")
                t_report = time.time()

            h = min(depth_vis.shape[0], rgb_frame.shape[0], wrist_frame.shape[0])
            d = cv2.resize(depth_vis, (int(depth_vis.shape[1] * h / depth_vis.shape[0]), h))
            r = cv2.resize(rgb_frame, (int(rgb_frame.shape[1] * h / rgb_frame.shape[0]), h))
            w = cv2.resize(wrist_frame, (int(wrist_frame.shape[1] * h / wrist_frame.shape[0]), h))
            combined = cv2.hconcat([d, r, w])
            cv2.putText(combined, "DEPTH", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(combined, "RGB", (d.shape[1] + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(combined, "WRIST", (d.shape[1] + r.shape[1] + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Depth | RGB | Wrist", combined)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        depth_stream.stop()
        device.close()
        cap_rgb.release()
        cap_wrist.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
