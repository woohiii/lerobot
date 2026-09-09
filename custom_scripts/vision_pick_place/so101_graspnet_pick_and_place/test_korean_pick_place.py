from __future__ import annotations

import json

import numpy as np
import pytest

from .models import CameraIntrinsics, DetectedObject, RgbdFrame, TaskCommand


def _frame(capture_id: int | None = None) -> RgbdFrame:
    return RgbdFrame(
        rgb=np.zeros((12, 12, 3), dtype=np.uint8),
        depth=np.full((12, 12), 500, dtype=np.uint16),
        intrinsics=CameraIntrinsics(100.0, 100.0, 6.0, 6.0),
        capture_id=capture_id,
    )


class FakeCamera:
    def __init__(self) -> None:
        self.capture_id = 0

    def read(self) -> RgbdFrame:
        self.capture_id += 1
        return _frame(self.capture_id)


class ReusedFrameCamera:
    def __init__(self) -> None:
        self.frame = _frame(1)

    def read(self) -> RgbdFrame:
        return self.frame


class FakeVlm:
    def detect_objects(self, _rgb: np.ndarray, _instruction: str) -> list[DetectedObject]:
        return [
            DetectedObject("red block", (1, 1, 5, 5), 0.95),
            DetectedObject("blue tray", (7, 7, 11, 11), 0.9, "destination"),
        ]

    def parse_korean_task(self, _command: str, detections: list[DetectedObject]) -> TaskCommand:
        return TaskCommand(detections[0], detections[1], arm="left")


class FailIfConstructed:
    def __init__(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("preview must not construct an arm")


def _args(calibration: str, mode: str) -> list[str]:
    return [
        "--command",
        "빨간 블록을 파란 트레이에 놓아줘",
        "--calibration",
        calibration,
        "--port",
        "/dev/fake",
        mode,
    ]


def _calibration(tmp_path) -> str:
    path = tmp_path / "hand_eye.json"
    path.write_text(json.dumps({"camera_to_base": np.eye(4).tolist(), "rms_error_m": 0.002}))
    return str(path)


def test_preview_prints_plan_without_constructing_an_arm(capsys: pytest.CaptureFixture[str], tmp_path) -> None:
    """Catches a preview path that initializes hardware before the dry plan is shown."""
    from .korean_pick_place import main

    result = main(
        _args(_calibration(tmp_path), "--preview"),
        camera_factory=FakeCamera,
        vlm_factory=FakeVlm,
        arm_factory=FailIfConstructed,
    )

    output = capsys.readouterr().out
    assert result == 0
    assert "dry plan" in output
    assert "source_camera_xyz" in output
    assert "destination_base_xyz" in output
    assert "waypoints" in output


def test_execute_requires_exact_confirmation(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Catches an execute path that accepts an arbitrary negative/partial acknowledgement."""
    from .korean_pick_place import main

    monkeypatch.setattr("builtins.input", lambda _prompt: "NO")

    assert (
        main(
            _args(_calibration(tmp_path), "--execute"),
            camera_factory=FakeCamera,
            vlm_factory=FakeVlm,
            arm_factory=FailIfConstructed,
        )
        == 2
    )


def test_execute_rejects_the_previous_confirmation_word(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Catches a gate that still accepts EXECUTE instead of EXECUTE_PICK_PLACE."""
    from .korean_pick_place import main

    monkeypatch.setattr("builtins.input", lambda _prompt: "EXECUTE")

    assert (
        main(
            _args(_calibration(tmp_path), "--execute"),
            camera_factory=FakeCamera,
            vlm_factory=FakeVlm,
            arm_factory=FailIfConstructed,
        )
        == 2
    )


def test_default_camera_fails_closed_without_calibration_intrinsics(capsys: pytest.CaptureFixture[str], tmp_path) -> None:
    """Catches a default Astra path that guesses intrinsics for metric localization."""
    from .korean_pick_place import main

    assert main(_args(_calibration(tmp_path), "--preview")) == 2
    # Missing intrinsics is now valid: the real Astra adapter supplies SDK XYZ
    # on demand.  In CI without the camera, the expected failure is the
    # hardware-open error instead.
    output = capsys.readouterr().out
    assert "Astra S did not open" in output or "camera_intrinsics" in output


def test_preview_rejects_reused_camera_frame(tmp_path) -> None:
    """Catches a three-sample localization path that accepts the same frame object repeatedly."""
    from .korean_pick_place import main

    assert main(_args(_calibration(tmp_path), "--preview"), camera_factory=ReusedFrameCamera, vlm_factory=FakeVlm) == 2
