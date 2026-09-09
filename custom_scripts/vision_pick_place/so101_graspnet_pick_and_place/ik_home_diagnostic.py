"""Read-only FK/IK sensitivity check for the currently held SO-101 pose."""

from __future__ import annotations

import argparse

import numpy as np


def main(argv: list[str] | None = None, arm_class=None) -> int:
    parser = argparse.ArgumentParser(description="Read-only SO-101 home-pose FK/IK diagnostic")
    parser.add_argument("--port", required=True, help="explicit SO-101 follower serial port")
    args = parser.parse_args(argv)

    if arm_class is None:
        from .left_arm import LeftSOArm101

        arm_class = LeftSOArm101
    arm = arm_class(port=args.port)
    connected = False
    try:
        arm.connect()
        connected = True
        joints = arm.get_joint_deg()
        xyz = arm.gripper_xyz()
        print("current joints:", np.round(joints, 2))
        print("current FK TCP:", np.round(xyz, 4))
        for name, target in (
            ("+X 1cm", (xyz[0] + 0.01, xyz[1], xyz[2])),
            ("+Y 1cm", (xyz[0], xyz[1] + 0.01, xyz[2])),
        ):
            plan = arm.preview_move(target)
            print(f"{name} max joint delta: {plan['max_abs_delta_deg']:.2f} deg")
    finally:
        if connected:
            arm.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
