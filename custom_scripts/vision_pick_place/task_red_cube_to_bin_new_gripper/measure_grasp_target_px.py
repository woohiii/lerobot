"""One-off tool: re-measure config.GRASP_TARGET_PX for the new gripper.

Shows the current wrist-cam frame (same PublishedFrameSource main.py uses -
does not open the camera device itself, see perception.py's module
docstring). Click where the new gripper's jaw tips actually are in-frame;
each click prints the pixel and redraws a crosshair there so you can see
what you picked before committing. 'q'/ESC to quit and print the final
value to paste into config.py.

Needs camera_hub.py already running and publishing to config.WRIST_FRAME_PATH
(main.py's docstring - separate process, GUI opencv venv).

Run: uv run python3 custom_scripts/vision_pick_place/task_red_cube_to_bin_new_gripper/measure_grasp_target_px.py
"""

from __future__ import annotations

import cv2

import config
from perception import PublishedFrameSource

WINDOW = "click the new gripper's jaw tips - 'q' to finish"


def main() -> None:
    cap = PublishedFrameSource(config.WRIST_FRAME_PATH)
    if not cap.isOpened():
        print(
            "[measure] 손목캠 프레임이 없습니다. 먼저 camera_hub.py를 실행해주세요:\n"
            "  source ~/lerobot_song_venv/bin/activate && "
            "python custom_scripts/vision_pick_place/camera_hub.py"
        )
        return

    ok, frame = cap.read()
    if not ok:
        print("[measure] 프레임을 읽지 못했습니다.")
        return

    picked: list[tuple[int, int]] = []

    def on_click(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            picked.append((x, y))
            print(f"[measure] 클릭: ({x}, {y})")

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_click)

    print("[measure] 그리퍼 집게 끝(그립 지점)을 클릭하세요. 여러 번 클릭 가능 - 마지막 클릭이 채택됩니다.")
    while True:
        display = frame.copy()
        if picked:
            cv2.drawMarker(display, picked[-1], (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
        cv2.imshow(WINDOW, display)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
    cv2.destroyAllWindows()
    cap.release()

    if not picked:
        print("[measure] 클릭 없이 종료했습니다 - config.py를 바꾸지 않았습니다.")
        return

    x, y = picked[-1]
    print(f"\nconfig.py에 반영하세요:\nGRASP_TARGET_PX = ({float(x)}, {float(y)})")


if __name__ == "__main__":
    main()
