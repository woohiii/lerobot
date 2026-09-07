"""
Red cube detection on the Orbbec Astra S color stream.

Modes
-----
snapshot (default): grab a few frames, run detection once, save
    annotated.png / mask.png next to this script, print the result, exit.
    Useful for headless / remote iteration (no GUI needed to inspect).

live: open OpenCV windows with HSV trackbars for interactive tuning.
    Press 's' to save the current thresholds to hsv_config.json,
    'q' or ESC to quit.

Usage:
    python red_cube_detect.py --mode snapshot
    python red_cube_detect.py --mode live
"""
import argparse
import json
import os

import cv2
import numpy as np

from orbbec_camera import OrbbecColorCamera

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "hsv_config.json")

# Reasonable starting point for a saturated red cube under normal indoor
# lighting. Red wraps around hue 0/180, so we keep two bands and OR them.
DEFAULT_HSV = {
    "h1_low": 0, "h1_high": 10,
    "h2_low": 170, "h2_high": 180,
    "s_low": 90, "s_high": 255,
    "v_low": 60, "v_high": 255,
}

MIN_AREA = 400  # px^2, filters out tiny noise blobs


def load_hsv():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return dict(DEFAULT_HSV)


def save_hsv(hsv):
    with open(CONFIG_PATH, "w") as f:
        json.dump(hsv, f, indent=2)
    print(f"[saved] {CONFIG_PATH}")


def red_mask(bgr, hsv_cfg):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower1 = np.array([hsv_cfg["h1_low"], hsv_cfg["s_low"], hsv_cfg["v_low"]])
    upper1 = np.array([hsv_cfg["h1_high"], hsv_cfg["s_high"], hsv_cfg["v_high"]])
    lower2 = np.array([hsv_cfg["h2_low"], hsv_cfg["s_low"], hsv_cfg["v_low"]])
    upper2 = np.array([hsv_cfg["h2_high"], hsv_cfg["s_high"], hsv_cfg["v_high"]])
    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return mask


def find_cube(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < MIN_AREA:
        return None
    x, y, w, h = cv2.boundingRect(c)
    cx, cy = x + w // 2, y + h // 2
    return {"bbox": (x, y, w, h), "center": (cx, cy), "area": area}


def annotate(bgr, det):
    out = bgr.copy()
    if det:
        x, y, w, h = det["bbox"]
        cx, cy = det["center"]
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.drawMarker(out, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(out, f"({cx},{cy}) area={int(det['area'])}", (x, max(0, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(out, "no red cube found", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return out


def run_snapshot(n_warmup=10):
    cam = OrbbecColorCamera()
    hsv_cfg = load_hsv()
    bgr = None
    for _ in range(n_warmup):  # let auto-exposure settle
        bgr = cam.read_bgr()
    cam.close()

    mask = red_mask(bgr, hsv_cfg)
    det = find_cube(mask)
    annotated = annotate(bgr, det)

    raw_path = os.path.join(HERE, "raw.png")
    mask_path = os.path.join(HERE, "mask.png")
    annotated_path = os.path.join(HERE, "annotated.png")
    cv2.imwrite(raw_path, bgr)
    cv2.imwrite(mask_path, mask)
    cv2.imwrite(annotated_path, annotated)

    print(f"frame shape: {bgr.shape}")
    print(f"hsv config: {hsv_cfg}")
    print(f"detection: {det}")
    print(f"wrote: {raw_path}\n       {mask_path}\n       {annotated_path}")


def run_live():
    cam = OrbbecColorCamera()
    hsv_cfg = load_hsv()

    win = "red_cube_detect (s=save thresholds, q=quit)"
    cv2.namedWindow(win)

    def nop(_):
        pass

    for key, val in hsv_cfg.items():
        maxval = 180 if key.startswith("h") else 255
        cv2.createTrackbar(key, win, val, maxval, nop)

    try:
        while True:
            bgr = cam.read_bgr()
            for key in hsv_cfg:
                hsv_cfg[key] = cv2.getTrackbarPos(key, win)

            mask = red_mask(bgr, hsv_cfg)
            det = find_cube(mask)
            annotated = annotate(bgr, det)

            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            combined = np.hstack([annotated, mask_bgr])
            cv2.imshow(win, combined)

            k = cv2.waitKey(1) & 0xFF
            if k in (ord('q'), 27):
                break
            elif k == ord('s'):
                save_hsv(hsv_cfg)
    finally:
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["snapshot", "live"], default="snapshot")
    args = parser.parse_args()
    if args.mode == "snapshot":
        run_snapshot()
    else:
        run_live()
