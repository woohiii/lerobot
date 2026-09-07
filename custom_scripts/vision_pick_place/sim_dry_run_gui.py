"""GUI version of sim_dry_run.py - same simulation (no robot, no real camera),
but renders a synthetic "wrist-cam view" window live as each case runs, so it
can actually be watched instead of only read as console text.

Must run in ~/lerobot_song_venv (GUI opencv) rather than via `uv run` in
~/lerobot's own venv (headless opencv, imshow raises there) - unlike the real
camera scripts, this doesn't need feetech-servo-sdk for anything actually
called (only robot_control.CollisionDetected, a plain exception class), and
lerobot_song_venv turns out to already have the `lerobot` package importable
too, so everything sim_dry_run.py needs is available here as well:
    source ~/lerobot_song_venv/bin/activate
    python3 sim_dry_run_gui.py
"""

from __future__ import annotations

import time

import cv2
import numpy as np

import visual_servo_pick_place as vsp
from sim_dry_run import (
    _ORIGINAL_estimate_cube_height_m,
    DummyCap,
    FakeRobotController,
    make_synthetic_detect_fn,
)

WINDOW = "가상 시뮬레이션 (실제 카메라/로봇 아님)"
FRAME_DELAY_S = 0.06  # slow enough to actually watch, fast enough not to be tedious


def render(canvas_title: str, det, iteration: int, err_norm: float | None, extra: str = "") -> None:
    frame = np.full((vsp.FRAME_H, vsp.FRAME_W, 3), 235, dtype=np.uint8)  # light gray "table"

    # GRASP_TARGET_PX crosshair - the point servo_to_target actually drives
    # error to zero against (see that constant's comment in
    # visual_servo_pick_place.py), drawn in blue.
    tx, ty = int(vsp.GRASP_TARGET_PX[0]), int(vsp.GRASP_TARGET_PX[1])
    cv2.drawMarker(frame, (tx, ty), (200, 100, 0), cv2.MARKER_CROSS, 24, 2)
    cv2.putText(frame, "GRASP_TARGET_PX", (tx + 14, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 100, 0), 1)

    # image center, for reference, in gray dashed-ish marker
    cx0, cy0 = int(vsp.IMG_CENTER[0]), int(vsp.IMG_CENTER[1])
    cv2.drawMarker(frame, (cx0, cy0), (140, 140, 140), cv2.MARKER_DIAMOND, 14, 1)

    if det is not None:
        cv2.circle(frame, (int(det.cx), int(det.cy)), 18, (40, 40, 220), -1)  # the "cube" (red, BGR)
        cv2.circle(frame, (int(det.cx), int(det.cy)), 18, (0, 0, 0), 1)
    else:
        cv2.putText(frame, "타겟 놓침", (250, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2)

    cv2.putText(frame, canvas_title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(frame, f"iter {iteration}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    if err_norm is not None:
        cv2.putText(frame, f"err={err_norm:.1f}px", (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    if extra:
        cv2.putText(frame, extra, (10, vsp.FRAME_H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 100, 0), 1)

    cv2.imshow(WINDOW, frame)
    cv2.waitKey(1)


def make_gui_detect_fn(base_detect_fn, canvas_title: str, counter: list[int]):
    def wrapped(frame):
        det = base_detect_fn(frame)
        counter[0] += 1
        err = None
        if det is not None:
            err = float(np.linalg.norm(vsp.GRASP_TARGET_PX - np.array([det.cx, det.cy])))
        render(canvas_title, det, counter[0], err)
        time.sleep(FRAME_DELAY_S)
        return det

    return wrapped


def run_case_gui(label, target_xy, object_present, edge_bias=(0.0, 0.0)):
    print(f"\n[GUI 시뮬레이션] {label}")
    home = vsp.HOME_XYZ
    rc = FakeRobotController(home, object_present=object_present)
    cap = DummyCap()
    base_detect_fn = make_synthetic_detect_fn(rc, target_xy, edge_bias=edge_bias)
    counter = [0]
    detect_fn = make_gui_detect_fn(base_detect_fn, label, counter)

    ok = vsp.servo_to_target(rc, cap, detect_fn, label)
    print(f"  -> servo_to_target: {ok}")
    if ok:
        grasped = vsp.descend_and_grasp(rc)
        print(f"  -> descend_and_grasp: grasped={grasped} (실제 정답={object_present})")
        result_text = f"결과: {'성공' if grasped == object_present else '검증 실패!'} (grasped={grasped})"
    else:
        result_text = "결과: 서보잉 실패"

    # Hold the final frame on screen for a moment so the outcome is readable
    # before moving to the next case, instead of instantly jumping ahead.
    for _ in range(20):
        render(label, None, counter[0], None, extra=result_text)
        time.sleep(0.1)


def run_height_test_gui():
    """GUI version of sim_dry_run.py's test_estimate_cube_height() - builds
    the same synthetic Astra RGB + depth pair, calls the real (non-
    monkeypatched) estimate_cube_height_m(), and shows both images side by
    side with the result overlaid, instead of only asserting in a console.
    Uses throwaway temp paths (via estimate_cube_height_m's overridable
    rgb_path/depth_path), not the shared production paths - a real
    astra_s_live.py running at the same time would otherwise race these
    writes, as it did once for real when this test used the shared paths."""
    import os
    import tempfile

    print("\n[GUI 시뮬레이션] 4/4: Astra 높이 추정 (합성 데이터)")
    color = np.full((480, 640, 3), 200, dtype=np.uint8)
    cv2.rectangle(color, (300, 200), (340, 240), (30, 30, 200), -1)

    depth_mm = np.full((240, 320), 500, dtype=np.uint16)
    KNOWN_HEIGHT_MM = 25
    depth_mm[100:120, 150:170] = 500 - KNOWN_HEIGHT_MM

    with tempfile.TemporaryDirectory() as tmpdir:
        rgb_path = os.path.join(tmpdir, "rgb.png")
        depth_path = os.path.join(tmpdir, "depth_mm.npy")
        cv2.imwrite(rgb_path, color)
        np.save(depth_path.replace(".npy", ""), depth_mm)

        height = _ORIGINAL_estimate_cube_height_m(rgb_path=rgb_path, depth_path=depth_path)
        depth_vis = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_mm, alpha=255.0 / depth_mm.max()), cv2.COLORMAP_JET
        )
        depth_vis = cv2.resize(depth_vis, (640, 480))
        combined = cv2.hconcat([color, depth_vis])
        ok = height is not None and abs(height - KNOWN_HEIGHT_MM / 1000.0) < 0.003
        text = f"추정: {height*1000:.1f}mm / 실제: {KNOWN_HEIGHT_MM}mm -> {'검증 통과' if ok else '검증 실패'}"
        cv2.putText(combined, "합성 Astra RGB", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(combined, "합성 Astra Depth", (650, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(
            combined, text, (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (0, 130, 0) if ok else (0, 0, 200), 2,
        )
        print(f"  -> {text}")
        cv2.imshow(WINDOW, combined)
        cv2.waitKey(1)
        time.sleep(2.0)


def main():
    home = vsp.HOME_XYZ
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, vsp.FRAME_W, vsp.FRAME_H)
    print("가상 시뮬레이션 GUI 실행 중 - 실제 로봇/카메라와 무관합니다. 창을 닫으려면 아무 키나 누르세요.")

    run_case_gui(
        "1/4: 정상 접근 + 실제로 집힘",
        target_xy=(home[0] - 0.06, home[1] + 0.02),
        object_present=True,
    )
    run_case_gui(
        "2/4: 정상 접근 + 실제로는 못 집음 (오늘 실제 버그 재현)",
        target_xy=(home[0] - 0.06, home[1] + 0.02),
        object_present=False,
    )
    run_case_gui(
        "3/4: 화면 가장자리에서 시작 (coarse-center)",
        target_xy=(home[0] - 0.04, home[1] - 0.02),
        object_present=True,
        edge_bias=(-300.0, 0.0),
    )

    run_height_test_gui()

    print("\n모든 케이스 완료. 아무 키나 누르면 창을 닫습니다.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
