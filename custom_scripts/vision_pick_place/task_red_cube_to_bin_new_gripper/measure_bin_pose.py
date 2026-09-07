"""One-off tool: hand-guided measurement of config.BIN_POSE_XYZ (the fixed
hover pose above the trash bin that click_grasp_bimanual.py moves to and
opens the gripper at, after a successful grasp+lift).

Same method already used for TABLE_Z (see probe_table_height_manual.py /
config.py's TABLE_Z comment): cuts torque so you can hand-guide the arm by
physically pushing it, re-engages torque at that exact pose, and reads it
back via forward kinematics - motor-driven positioning isn't trusted for
hardware bring-up measurements in this task.

Run once per arm (the bin position differs per arm - different mount point
on the trolley, different reach). Use the PHYSICALLY correct port (confirmed
via `uv run lerobot-find-port`, don't guess - see config.py's 2026-09-07 note):

    uv run python3 custom_scripts/vision_pick_place/task_red_cube_to_bin_new_gripper/measure_bin_pose.py --port /dev/so101_follower
"""

from __future__ import annotations

import argparse

import config
from kinematics import SOArm101


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="측정할 팔의 follower 포트")
    args = parser.parse_args()

    arm = SOArm101(port=args.port)
    arm.connect()
    try:
        arm.release_torque()
        print("토크를 껐습니다 - 팔이 힘없이 움직일 수 있는 상태입니다.")
        print("그리퍼를 손으로 쓰레기통 위, 놓기 좋은 위치까지 옮겨주세요.")
        input("위치를 잡으셨으면 Enter... ")
        arm.enable_torque()
        print("현재 위치에서 토크 재활성화 완료 (팔이 그 자리에서 버팁니다).")

        joints = arm.get_joint_deg()
        xyz = tuple(arm.kin.forward_kinematics(joints[: len(config.ARM_JOINTS)])[:3, 3])
        print(f"[결과] 현재 그리퍼 xyz: {xyz}")
        print("\nconfig.py에 반영하세요 (오른팔=BIN_POSE_XYZ, 왼팔=LEFT_OVERRIDES['BIN_POSE_XYZ']):")
        print(f"BIN_POSE_XYZ = ({xyz[0]:.4f}, {xyz[1]:.4f}, {xyz[2]:.4f})")
    finally:
        arm.disconnect()


if __name__ == "__main__":
    main()
