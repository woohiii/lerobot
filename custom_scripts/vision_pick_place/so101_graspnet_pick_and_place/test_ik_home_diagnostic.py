from . import ik_home_diagnostic


def test_diagnostic_uses_explicit_port(monkeypatch) -> None:
    ports: list[str] = []

    class FakeArm:
        def __init__(self, *, port: str) -> None:
            ports.append(port)

        def connect(self) -> None:
            pass

        def get_joint_deg(self):
            return [0, 0, 0, 0, 0, 0]

        def gripper_xyz(self):
            return [0.2, 0.0, 0.1]

        def preview_move(self, target):
            return {"max_abs_delta_deg": 1.0}

        def disconnect(self) -> None:
            pass

    assert ik_home_diagnostic.main(["--port", "/dev/ttyACM3"], arm_class=FakeArm) == 0
    assert ports == ["/dev/ttyACM3"]
