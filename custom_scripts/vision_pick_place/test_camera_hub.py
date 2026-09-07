"""camera_hub의 양쪽 손목캠 선택/부재 처리를 실제 하드웨어 없이 검증한다.

Run: uv run python3 test_camera_hub.py
"""

import contextlib
import importlib
import io
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import camera_utils


class FakeCameraWorker:
    created = []

    def __init__(self, index, out_path, label, annotate_fn, v4l2_ctrls=None, filter_corruption=True, fourcc=None):
        self.index = index
        self.out_path = out_path
        self.label = label
        self.v4l2_ctrls = v4l2_ctrls
        self.filter_corruption = filter_corruption
        self.fourcc = fourcc
        self.cap = self
        self.started = False
        FakeCameraWorker.created.append(self)

    def isOpened(self):
        return True

    def start(self):
        self.started = True

    def latest(self):
        return None

    def corruption_pct(self):
        return 0.0

    def stop(self):
        pass

    def join(self, timeout=None):
        pass

    def release(self):
        pass


def load_camera_hub(indices):
    calls = []
    original = camera_utils.find_camera_index

    def fake_find_camera_index(name_substring, occurrence=0):
        calls.append((name_substring, occurrence))
        return indices.get((name_substring, occurrence))

    camera_utils.find_camera_index = fake_find_camera_index
    try:
        sys.modules.pop("camera_hub", None)
        return importlib.import_module("camera_hub"), calls
    finally:
        camera_utils.find_camera_index = original


def run_main_once(camera_hub):
    FakeCameraWorker.created = []
    camera_hub.CameraWorker = FakeCameraWorker
    camera_hub.cv2.waitKey = lambda delay: ord("q")
    camera_hub.cv2.destroyAllWindows = lambda: None
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        camera_hub.main()
    return output.getvalue()


def test_distinct_wrist_cameras_and_paths():
    camera_hub, calls = load_camera_hub(
        {
            ("USB Camera", 0): 1,
            ("USB 2.0 PC Cam", 0): 2,
            ("Innomaker-U20CAM-720P", 0): 3,
        }
    )

    assert ("USB 2.0 PC Cam", 0) in calls
    assert ("Innomaker-U20CAM-720P", 0) in calls

    run_main_once(camera_hub)
    workers = {(worker.index, worker.out_path) for worker in FakeCameraWorker.created}
    assert (2, "/tmp/vsp_wrist.png") in workers
    assert (3, "/tmp/vsp_wrist_left.png") in workers
    left = next(worker for worker in FakeCameraWorker.created if worker.index == 3)
    assert not left.filter_corruption
    assert left.v4l2_ctrls is None
    assert left.fourcc == "MJPG"
    print("[PASS] 왼쪽 Innomaker는 별도 경로로 publish하고 USB 2.0 PC Cam 전용 손상 필터/설정을 적용하지 않습니다.")


def test_missing_left_wrist_camera_keeps_other_workers_running():
    camera_hub, _ = load_camera_hub(
        {
            ("USB Camera", 0): 1,
            ("USB 2.0 PC Cam", 0): 2,
            ("Innomaker-U20CAM-720P", 0): None,
        }
    )

    output = run_main_once(camera_hub)
    assert "왼쪽 손목캠을 찾을 수 없습니다" in output
    assert {(worker.index, worker.out_path) for worker in FakeCameraWorker.created} == {
        (1, "/tmp/vsp_rgb.png"),
        (2, "/tmp/vsp_wrist.png"),
    }
    assert all(worker.started for worker in FakeCameraWorker.created)
    print("[PASS] 왼쪽 손목캠 미연결 시 경고만 출력하고 기존 워커를 계속 실행합니다.")


if __name__ == "__main__":
    test_distinct_wrist_cameras_and_paths()
    test_missing_left_wrist_camera_keeps_other_workers_running()
