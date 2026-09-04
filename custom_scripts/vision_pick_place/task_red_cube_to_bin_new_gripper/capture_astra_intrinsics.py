"""ABANDONED 2026-09-01 - kept only as a record of what was tried, per this
project's own convention of documenting dead ends instead of deleting them.
The cross-check below (own formula vs. the SDK's own convert_depth_to_world,
run on real depth data) came back with up to 555mm error, and swapping the
depth stream's FOV for the color stream's (registration puts the depth map in
the color camera's pixel space) made no difference - both streams report the
identical FOV, ruling that hypothesis out too. Conclusion: this sensor's real
world-coordinate conversion folds in more than its advertised FOV (almost
certainly its own factory depth-calibration table, which isn't exposed via
get_horizontal_fov()/get_vertical_fov()) - guessing further at the formula
was a dead end, not a bug to keep chasing. Per explicit user decision, the
project is NOT pursuing a real per-pixel 3D backprojection right now; the LLM-
grounded pick pipeline (llm_pick_place.py) instead reuses the existing,
already-validated homography (xy) + Astra depth-DELTA (z) coarse-estimate
approach - the same one perception.py's HSV detectors have always used - just
with perception_qwen.detect_qwen swapped in as the detect_fn. Revisit this
file only if that approach's accuracy proves insufficient for objects that
aren't roughly table-flat.

Original goal (not achieved): reads the Astra S depth stream's own reported
field-of-view and saves the pinhole intrinsics derived from it (fx, fy, cx,
cy) to astra_intrinsics.json - meant to be the missing piece for turning a
(pixel, depth_mm) pair into a real camera-frame 3D point.

Why FOV instead of a checkerboard calibration: OpenNI2/PrimeSense depth
sensors (this Astra S included) report their own horizontal/vertical FOV via
the SDK, and this library's own `openni2.convert_depth_to_world()` derives its
pinhole intrinsics from exactly that FOV internally (fx = (width/2) /
tan(hFOV/2), principal point at image center) - not a separate lens
calibration. Reading the FOV once and deriving fx/fy/cx/cy ourselves lets the
CONTROL venv (this one - no openni2/primesense package, no live device
access) redo the same pixel+depth -> XYZ math read-only, the same "reuse the
already-validated formula without a second live camera connection" pattern
already used by perception_zeroshot.py/zeroshot_pick.py for the homography
math. See verify_against_convert_depth_to_world() below for the actual proof
this reimplementation matches the SDK's own conversion, not just an assumption.

Must run in ~/lerobot_song_venv (has `primesense`/openni2, GUI opencv) - same
venv as camera_hub.py/astra_s_live.py. Only one process may hold the Astra S
device at a time, so make sure camera_hub.py/astra_s_live.py are NOT running
before this.

Run:
  source ~/lerobot_song_venv/bin/activate
  python custom_scripts/vision_pick_place/task_red_cube_to_bin_new_gripper/capture_astra_intrinsics.py
"""

import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # orbbec_color_camera.py's dir

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "astra_intrinsics.json")


def pixel_depth_to_camera_xyz_mm(u: float, v: float, depth_mm: float, width: int, height: int, fx: float, fy: float):
    """Same pinhole formula PrimeSense's convert_depth_to_world uses
    internally (principal point at image center, fx/fy from FOV) - the
    read-only reimplementation this capture script exists to validate."""
    cx, cy = width / 2.0, height / 2.0
    x = (u - cx) * depth_mm / fx
    y = (v - cy) * depth_mm / fy
    return x, y, depth_mm


def main() -> None:
    from orbbec_color_camera import ThreadedOrbbecRGBDCamera
    from primesense import openni2

    cam = ThreadedOrbbecRGBDCamera(width=640, height=480, fps=30)
    if not cam.isOpened():
        print("[오류] Astra S를 열 수 없습니다 - 다른 프로세스(camera_hub.py/astra_s_live.py)가 켜져있는지 확인.")
        sys.exit(1)

    try:
        hfov = cam.depth_stream.get_horizontal_fov()  # radians
        vfov = cam.depth_stream.get_vertical_fov()
        w, h = cam.width, cam.height
        fx = (w / 2.0) / math.tan(hfov / 2.0)
        fy = (h / 2.0) / math.tan(vfov / 2.0)
        cx, cy = w / 2.0, h / 2.0
        print(f"[intrinsics] hfov={math.degrees(hfov):.2f}deg vfov={math.degrees(vfov):.2f}deg")
        print(f"[intrinsics] fx={fx:.2f} fy={fy:.2f} cx={cx:.1f} cy={cy:.1f} (at {w}x{h})")

        # Cross-check: grab one real depth frame, pick a handful of pixels with
        # valid depth, compare our formula against the SDK's own
        # convert_depth_to_world on the SAME (pixel, depth) - confirms the
        # reimplementation is right before anything downstream trusts it.
        print("[검증] 실제 프레임으로 공식 vs SDK 내장 변환 비교 중...")
        # read()/read_raw_depth_mm()은 백그라운드 스레드가 채우는 값이라 스트림
        # 시작 직후엔 아직 첫 프레임이 안 들어왔을 수 있음 (isOpened()는 스트림
        # 시작만 확인하지 프레임 도착까지 기다리지 않음) - 짧게 재시도.
        ret, color, depth_vis = False, None, None
        depth_mm = None
        for _ in range(20):
            ret, color, depth_vis = cam.read()
            depth_mm = cam.read_raw_depth_mm()
            if ret and depth_mm is not None:
                break
            time.sleep(0.2)
        if not ret or depth_mm is None:
            print("[경고] 검증용 프레임을 못 읽었습니다 - intrinsics는 저장하지만 교차검증은 건너뜁니다.")
        else:
            import numpy as np

            valid_ys, valid_xs = np.where(depth_mm > 0)
            sample_idx = np.linspace(0, len(valid_xs) - 1, min(8, len(valid_xs)), dtype=int)
            max_err_mm = 0.0
            for i in sample_idx:
                u, v = int(valid_xs[i]), int(valid_ys[i])
                d = float(depth_mm[v, u])
                mine = pixel_depth_to_camera_xyz_mm(u, v, d, w, h, fx, fy)
                sdk = openni2.convert_depth_to_world(cam.depth_stream, float(u), float(v), d)
                err = math.dist(mine, sdk)
                max_err_mm = max(max_err_mm, err)
                print(f"  px=({u},{v}) d={d:.0f}mm  mine={tuple(round(c,1) for c in mine)}  sdk={tuple(round(c,1) for c in sdk)}  err={err:.2f}mm")
            print(f"[검증] 최대 오차 {max_err_mm:.2f}mm" + (" - 일치함, 신뢰 가능" if max_err_mm < 1.0 else " - 불일치, 공식 재검토 필요"))

        with open(OUT_PATH, "w") as f:
            json.dump({"width": w, "height": h, "fx": fx, "fy": fy, "cx": cx, "cy": cy,
                       "hfov_deg": math.degrees(hfov), "vfov_deg": math.degrees(vfov)}, f, indent=2)
        print(f"[저장 완료] {OUT_PATH}")
    finally:
        cam.release()


if __name__ == "__main__":
    main()
