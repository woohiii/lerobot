"""One-off: move the arm's 5 arm joints directly (joint-space, not IK) back to the
pose recorded before the joint-limit bug in the first test - not by IK-to-xyz
(which failed to get back exactly, since the true start pose needed
shoulder_lift/elbow_flex outside the old too-tight clamp)."""

import time

from robot_control import ALL_JOINTS, RobotController
import numpy as np

# Exactly what get_joint_deg() printed before the first test_move.py run.
TARGET_DEG = np.array([-5.40659341, -109.45054945, 99.34065934, 91.03296703, -98.68131868, 0.0])

rc = RobotController()
rc.connect()
try:
    current = rc.get_joint_deg()
    print("current:", current)
    print("target: ", TARGET_DEG)
    delta = TARGET_DEG - current
    max_abs = np.max(np.abs(delta))
    print("delta:  ", delta, " max abs:", max_abs)
    if max_abs > 20.0:
        print(f"[abort] max delta {max_abs:.1f}deg > 20deg safety threshold - not moving.")
        raise SystemExit(1)

    steps = 40
    for i in range(1, steps + 1):
        interp = current + delta * (i / steps)
        rc.send_joint_deg(interp)
        time.sleep(0.05)
    time.sleep(0.5)

    final = rc.get_joint_deg()
    print("final:  ", final)
    print("error:  ", final - TARGET_DEG)
finally:
    rc.disconnect()
