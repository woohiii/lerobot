"""Re-records just the gripper's range of motion (not the whole arm's calibration -
homing offsets and the other 5 joints' ranges are left untouched). Disables torque
on the gripper only, so it can be moved by hand, for a fixed window instead of the
usual interactive Enter-to-stop (that needs a live TTY this script doesn't have).
"""

import time

from robot_control import RobotController

RECORD_SECONDS = 20


def main():
    rc = RobotController()
    rc.connect()
    bus = rc.robot.bus
    try:
        current_cal = bus.calibration["gripper"]
        print(f"기존 gripper 캘리브레이션: {current_cal}")

        bus.disable_torque(["gripper"])
        print(f"\n토크 해제됨 - {RECORD_SECONDS}초 동안 손으로 그리퍼를 완전히 닫았다 열었다 여러 번 반복해주세요.")
        print("(진짜 끝까지 닫힌 지점, 진짜 끝까지 열린 지점을 확실히 지나가야 합니다)\n")

        start_pos = bus.sync_read("Present_Position", ["gripper"], normalize=False)["gripper"]
        vmin = vmax = start_pos
        t0 = time.time()
        while time.time() - t0 < RECORD_SECONDS:
            pos = bus.sync_read("Present_Position", ["gripper"], normalize=False)["gripper"]
            vmin = min(vmin, pos)
            vmax = max(vmax, pos)
            remaining = RECORD_SECONDS - (time.time() - t0)
            print(f"\r남은 시간 {remaining:4.1f}s | 현재={pos:5d} min={vmin:5d} max={vmax:5d}", end="", flush=True)
            time.sleep(0.05)
        print()

        bus.enable_torque(["gripper"])

        if vmax - vmin < 100:
            print(f"\n[중단] 관측된 범위가 너무 좁습니다 (min={vmin}, max={vmax}, span={vmax-vmin}). "
                  "그리퍼를 손으로 충분히 움직이지 못한 것 같아요 - 다시 시도해주세요.")
            return

        from lerobot.motors.motors_bus import MotorCalibration

        new_cal = MotorCalibration(
            id=current_cal.id,
            drive_mode=current_cal.drive_mode,
            homing_offset=current_cal.homing_offset,
            range_min=vmin,
            range_max=vmax,
        )
        print(f"\n새 gripper 캘리브레이션: {new_cal}")

        rc.robot.calibration["gripper"] = new_cal
        bus.write_calibration(rc.robot.calibration)
        rc.robot._save_calibration()
        print(f"저장 완료: {rc.robot.calibration_fpath}")

        # sanity check: read back normalized position, should now read close to
        # one end depending on where the jaw happens to be sitting
        pos_now = bus.sync_read("Present_Position", ["gripper"])["gripper"]
        print(f"현재 정규화된 위치: {pos_now:.1f} (0-100)")
    finally:
        rc.disconnect()


if __name__ == "__main__":
    main()
