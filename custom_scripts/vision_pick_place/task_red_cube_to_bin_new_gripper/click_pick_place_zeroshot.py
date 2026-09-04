"""Click-driven pick-and-place on the live Gemini-labeled view - per the
user's 2026-09-01 request to click WHATEVER object Gemini is currently
boxing/naming (not just the red cube), then click a table spot to place it.

Combines three already-real-hardware-validated pieces rather than building
anything new underneath:
  - click_pick_place.py's click-to-select UI
  - llm_pick_place.py's OPEN-LOOP downstream (move -> descend_and_grasp,
    no wrist-cam HSV fine_servo) - reused because fine_servo's closed-loop
    refinement only knows detect_red_cube/detect_black_bin; an arbitrary
    clicked object can't use it (same documented tradeoff as
    perception_zeroshot.py's module docstring: Gemini's ~8-9s/call is too
    slow for a per-frame servo loop)
  - zeroshot_viewer.py's LabelWorker (background thread running Gemini's
    label_all_objects on its own cadence) as the click surface, so what you
    click is what Gemini is currently boxing, not a blind Astra frame

Click near a box -> that box's own bbox center is used (not the raw click
pixel) if the click landed inside a labeled box, for a less noisy homography
lookup than a hand click; falls back to the raw click pixel if it missed
every box (e.g. clicking a table spot to place, or Gemini hasn't boxed that
object yet). Height estimation during descend (descend_and_grasp ->
perception.estimate_cube_height_m) only works for the red cube (see that
function) - falls back safely to TABLE_Z-based contact detection for
anything else, same as llm_pick_place.py.

2026-09-01: the window used to be opened/destroyed by a per-click
wait_for_click() call, so it visibly disappeared during the arm move between
the pick and place clicks - per the user's "화면이 꺼지는데 안꺼지도록" ask,
ClickWindow below now owns ONE window for the whole run, and the arm/grasp
sequence (run_task) moves to its OWN thread so it never blocks the display
loop. GUI calls (imshow/waitKey/namedWindow) stay on the MAIN thread only -
every other GUI loop in this project does the same (cv2's Qt highgui isn't
reliably thread-safe for those calls otherwise); run_task only touches the
robot (pyserial I/O, no GUI), so there's no cross-thread GUI conflict.

Needs camera_hub.py AND astra_s_live.py already running and publishing (see
main.py's docstring - camera devices aren't opened here).

Run: uv run python3 custom_scripts/vision_pick_place/task_red_cube_to_bin_new_gripper/click_pick_place_zeroshot.py
"""

from __future__ import annotations

import queue
import sys
import threading
import time

import cv2

import config
import gripper
import perception
import task_state_machine as tsm
from kinematics import CollisionDetected, SOArm101
from perception_zeroshot import draw_labeled_boxes
from zeroshot_viewer import LabelWorker, combine_with_depth

WINDOW = "click an object (or a table spot) - 'q' to cancel"


def _box_at(labeled, x: int, y: int):
    """First labeled entry whose bbox contains (x, y), or None - used to
    snap a click to that box's own center instead of the raw click pixel."""
    for det, mask, yaw, label, score in labeled:
        bx, by, bw, bh = det.bbox
        if bx <= x <= bx + bw and by <= y <= by + bh:
            return det, label
    return None


class ClickWindow:
    """Owns the one persistent window for the whole run (see module
    docstring). run_display_loop() is MAIN-thread-only and blocks until
    close(); next_click() is called from run_task()'s own thread and blocks
    only that thread."""

    def __init__(self, worker: LabelWorker):
        self.worker = worker
        self._clicks: queue.Queue[tuple[int, int]] = queue.Queue()
        self._done = threading.Event()
        self.cancelled = False
        # Set each tick in run_display_loop to the Astra frame's own width,
        # BEFORE the depth/wrist panels are appended alongside it - lets
        # _on_click reject clicks that land on those panels instead of
        # passing their pixel straight into pixel_to_xy's homography
        # (same out-of-workspace risk llm_pick_place.py's
        # is_xy_within_safe_workspace guards against for bad LLM detections;
        # here the "bad detection" would just be a click on the wrong panel).
        self._astra_width = None

    def _on_click(self, event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self._astra_width is not None and x >= self._astra_width:
                print("[click_pick_place_zeroshot] 카메라 패널(뎁스/손목) 클릭 무시 - Astra 화면 안쪽을 클릭하세요.")
                return
            self._clicks.put((x, y))

    def next_click(self, prompt: str) -> tuple[int, int] | None:
        """Blocks the CALLING thread until a new click arrives. Drains any
        clicks already queued before this call (leftover from the previous
        stage) so they can't be silently reused as this stage's click.
        Returns the clicked box's own center (see _box_at) if the click
        landed inside one, else the raw pixel. None if cancelled/closed."""
        while not self._clicks.empty():
            self._clicks.get_nowait()
        print(f"[click_pick_place_zeroshot] {prompt}")
        while not self._done.is_set():
            try:
                x, y = self._clicks.get(timeout=0.1)
            except queue.Empty:
                continue
            labeled, _ts = self.worker.latest()
            hit = _box_at(labeled or [], x, y)
            if hit is not None:
                det, label = hit
                print(f"[click_pick_place_zeroshot] '{label}' 선택됨 (박스 중심으로 스냅: {det.cx:.0f},{det.cy:.0f})")
                return (int(det.cx), int(det.cy))
            print(f"[click_pick_place_zeroshot] 박스 밖 클릭 - 클릭 좌표 그대로 사용: ({x},{y})")
            return (x, y)
        return None

    def close(self) -> None:
        self._done.set()

    def run_display_loop(self) -> None:
        """MAIN thread only - see module docstring. Keeps redrawing the
        live Astra view + Gemini's current boxes every tick for the whole
        run, never destroying/recreating the window between clicks."""
        cv2.namedWindow(WINDOW)
        cv2.setMouseCallback(WINDOW, self._on_click)
        cap = perception.PublishedFrameSource(config.ASTRA_RGB_FRAME_PATH)
        try:
            while not self._done.is_set():
                ok, frame = cap.read()
                labeled, _ts = self.worker.latest()
                if ok:
                    self._astra_width = frame.shape[1]
                    display = draw_labeled_boxes(frame, labeled) if labeled else frame.copy()
                    display = combine_with_depth(display)
                    cv2.imshow(WINDOW, display)
                key = cv2.waitKey(30) & 0xFF
                if key in (ord("q"), 27):
                    self.cancelled = True
                    self._done.set()
                    break
        finally:
            cv2.destroyWindow(WINDOW)


def move_and_grasp(arm: SOArm101, pt: tuple[int, int]) -> bool:
    xy = perception.pixel_to_xy(*pt)
    if xy is None:
        print("[click_pick_place_zeroshot] 호모그래피 없음 (homography.json 확인 필요)")
        return False
    target = (xy[0], xy[1], config.SEARCH_HOVER_XYZ[2])
    print(f"[click_pick_place_zeroshot] 대상 위치로 이동: ({xy[0]:.3f}, {xy[1]:.3f})")
    try:
        arm.move_to_xyz_converge(target, tolerance_m=0.015, max_iters=20)
    except CollisionDetected as e:
        print(f"[click_pick_place_zeroshot] 이동 중 충돌 감지 ({e}) - 취소")
        return False
    # 손목캠 폐루프 보정 없음 (open-loop) - 모듈 docstring 참고.
    return tsm.descend_and_grasp(arm)


def move_and_place(arm: SOArm101, pt: tuple[int, int]) -> bool:
    xy = perception.pixel_to_xy(*pt)
    if xy is None:
        print("[click_pick_place_zeroshot] 호모그래피 없음")
        return False
    hover = (xy[0], xy[1], config.SEARCH_HOVER_XYZ[2])
    print(f"[click_pick_place_zeroshot] 놓을 위치로 이동: ({xy[0]:.3f}, {xy[1]:.3f})")
    try:
        arm.move_to_xyz_converge(hover, tolerance_m=0.015, max_iters=20)
    except CollisionDetected as e:
        print(f"[click_pick_place_zeroshot] 이동 중 충돌 감지 ({e}) - 현재 위치에서 계속")

    try:
        arm.move_z(-config.BIN_DESCEND_M, steps=15, step_delay_s=0.05)
    except CollisionDetected:
        print("[click_pick_place_zeroshot] 하강 중 접촉 감지 - 현재 위치에서 놓음")
    gripper.open_gripper(arm)
    time.sleep(0.3)
    arm.move_z(config.BIN_DESCEND_M, steps=15, step_delay_s=0.05)
    return True


def run_task(arm: SOArm101, click_window: ClickWindow) -> bool:
    """Runs on its OWN thread (not main) - see module docstring. Only
    touches the robot, never cv2."""
    gripper.open_gripper(arm)
    pt = click_window.next_click("집을 물체를 클릭하세요 (q=취소)")
    if pt is None or not move_and_grasp(arm, pt):
        print("[click_pick_place_zeroshot] 픽업 실패/취소 - 홈으로 복귀합니다.")
        return False
    print("[click_pick_place_zeroshot] 파지 성공 - 상승")
    arm.move_z(config.LIFT_M, steps=20, step_delay_s=0.05)

    pt2 = click_window.next_click("놓을 위치를 클릭하세요 (q=취소, 든 채로 홈 복귀)")
    if pt2 is None or not move_and_place(arm, pt2):
        print("[click_pick_place_zeroshot] 놓기 실패/취소 - 든 채로 홈 복귀합니다.")
        return False
    print("[click_pick_place_zeroshot] 완료")
    return True


def main() -> bool:
    astra_cap = perception.PublishedFrameSource(config.ASTRA_RGB_FRAME_PATH)
    if not astra_cap.isOpened():
        print(
            "[click_pick_place_zeroshot] Astra 프레임이 없습니다. 먼저 astra_s_live.py를 실행해주세요:\n"
            "  source ~/lerobot_song_venv/bin/activate && "
            "python custom_scripts/vision_pick_place/astra_s_live.py"
        )
        return False

    arm = SOArm101()
    arm.connect()
    print("[click_pick_place_zeroshot] 연결 성공. 현재 관절각:", arm.get_joint_deg())
    home_pose = arm.get_joint_deg()
    home_xyz = tuple(arm.kin.forward_kinematics(home_pose[: len(config.ARM_JOINTS)])[:3, 3])

    label_worker = LabelWorker()
    label_worker.start()
    click_window = ClickWindow(label_worker)

    result = {"ok": False}

    def task():
        try:
            result["ok"] = run_task(arm, click_window)
        except KeyboardInterrupt:
            print("\n[click_pick_place_zeroshot] 사용자가 중단했습니다.")
        except Exception as e:  # noqa: BLE001 - must not skip close()/home-return below
            print(f"[click_pick_place_zeroshot] 예외 발생: {type(e).__name__}: {e}")
        finally:
            click_window.close()  # always releases the main thread's display loop

    task_thread = threading.Thread(target=task, daemon=True)
    task_thread.start()
    click_window.run_display_loop()  # MAIN thread, blocks until task() calls close()
    task_thread.join(timeout=10)

    label_worker.stop()
    label_worker.join(timeout=2)
    print(f"[click_pick_place_zeroshot] 홈 포즈로 복귀: {home_xyz}")
    try:
        arm.move_to_xyz_converge(home_xyz, tolerance_m=0.015, max_iters=20)
    except CollisionDetected as e:
        print(f"[click_pick_place_zeroshot] 홈 복귀 중 충돌 감지, 안전 위치에서 정지: {e}")
    except Exception as e:
        print(f"[click_pick_place_zeroshot] 홈 복귀 실패: {e}")
    try:
        arm.disconnect()
    except Exception as e:
        # 2026-09-01: seen for real - a mid-task comm dropout (USB/serial
        # flakiness) can leave the bus unable to even service disconnect()'s
        # own disable_torque write, raising uncaught and burying the actual
        # home-return outcome above under a traceback. Nothing more to do
        # here either way - just don't let this be the last, loudest thing.
        print(f"[click_pick_place_zeroshot] disconnect() 실패 (통신 문제로 보임): {e}")
    return result["ok"]


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
