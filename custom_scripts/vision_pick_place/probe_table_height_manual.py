"""Hand-guided TABLE_Z re-measurement - safer alternative to
probe_table_height.py's auto-descent.

2026-09-01: the auto-descent probe failed to detect real contact on the new
gripper - it kept reporting "접촉 없음" past the point the user could see/feel
it touching the table, ran to the HARD_FLOOR_Z backstop, then errored
returning to safe height. Same failure mode probe_table_height.py's original
docstring already flagged for the OLD gripper ("either TABLE_Z doesn't reach
the table, or contact detection isn't triggering for a light touch") -
apparently still true on this jaw. Motor-driven descent can't be trusted to
stop itself here.

This cuts torque instead, so YOU drive the gripper down onto the table by
hand until it just touches (no motor pushing anything), re-engages torque at
that exact pose, and reads it back via forward kinematics - the same method
that originally fixed the OLD gripper's TABLE_Z (see config.py's TABLE_Z
comment: "the user physically drove the gripper down onto the table and read
back the live xyz at actual contact").

Run: uv run python3 custom_scripts/vision_pick_place/probe_table_height_manual.py
(needs to live beside robot_control.py, same as manual_grasp_calibration.py -
that's a bare `from robot_control import ...`, not a package import.)
"""

from __future__ import annotations

from robot_control import RobotController

PORT = "/dev/so101_follower"


def main() -> None:
    rc = RobotController(port=PORT)
    rc.connect()
    try:
        rc.emergency_stop()
        print("토크를 껐습니다 - 팔이 힘없이 움직일 수 있는 상태입니다.")
        print("그리퍼를 손으로 테이블에 '살짝' 닿을 때까지만 내려주세요 (누르지 마세요).")
        input("위치를 잡으셨으면 Enter... ")
        rc.enable_torque()
        print("현재 위치에서 토크 재활성화 완료 (팔이 그 자리에서 버팁니다).")

        xyz = rc.current_gripper_xyz()
        print(f"[결과] 현재 그리퍼 xyz: {xyz}")
        print(f"\nconfig.py에 반영하세요:\nTABLE_Z = {xyz[2] + 0.003:.4f}  # 3mm 여유 포함")
    finally:
        rc.disconnect()


if __name__ == "__main__":
    main()
