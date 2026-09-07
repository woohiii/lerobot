"""Thin direct-control wrapper around the SO-101 follower for the pick-and-place
task - connects straight to the follower (no leader/teleop involved) and sends
joint-degree targets computed by kinematics_helper's IK.

Safety layers (defense in depth - any one of these alone would probably be
enough, but a bad IK solution during early testing is exactly the kind of bug
that should be caught by more than one guard):
  1. JOINT_LIMITS_DEG hard-clamps every commanded degree to the arm's known
     physical range before it's ever sent, regardless of what IK computed.
  2. move_to_xyz refuses (raises, sends nothing) if the resulting per-joint
     delta from the *current* pose exceeds MAX_MOVE_DELTA_DEG - catches a wildly
     wrong target (e.g. from a bad calibration point) before any motion at all.
  3. Every actual send is broken into small interpolated steps (not one jump).
  4. lerobot's own max_relative_target backstops every individual send_action
     call underneath all of the above.
  5. preview_move() computes and returns a plan without moving anything, for a
     dry-run/confirm-before-moving workflow.
"""

from __future__ import annotations

import time

import numpy as np

from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SOFollower

from kinematics_helper import ARM_JOINTS, build_kinematics, gripper_position, solve_ik

FOLLOWER_PORT = "/dev/so101_follower"
ALL_JOINTS = ARM_JOINTS + ["gripper"]


class CollisionDetected(RuntimeError):
    """Raised when a move is aborted because the arm appears to be blocked -
    actual joint position stopped tracking the commanded trajectory for several
    consecutive checks (the same signature this session already saw for real:
    the gripper hard-stalling at 47.8% against a physical obstruction, and the
    arm plateauing ~0.017m from a joint-limited target). The arm is sent back
    to its last known-good pose before this is raised, so callers land near
    where the block was first noticed rather than pushed into it."""


# Per-joint gap between commanded and actual position that counts as "not
# really moving". Checked every STALL_CHECK_EVERY steps (not every step) so
# ordinary servo response lag doesn't false-trigger, and only counts once it
# persists for STALL_CONSECUTIVE checks in a row (a single slow step is normal;
# several in a row with no progress is a real block).
STALL_THRESHOLD_DEG = 10.0
STALL_CHECK_EVERY = 3
STALL_CONSECUTIVE = 3

# This robot's actual calibrated range, verified two ways: (1) hand-computed from
# this unit's calibration file
# (~/.cache/huggingface/lerobot/calibration/robots/so_follower/follower.json) via
# lerobot's tick->degree formula, and (2) cross-checked by calling the live bus's
# own bus._normalize() directly on its loaded range_min/range_max - both agree
# exactly. A first hand-picked-from-a-different-project guess (leisaac's sim
# joint limits) was wrong for this specific real unit: it clamped shoulder_lift/
# elbow_flex tighter than a pose the arm was already safely resting at, and a
# "return to start" move couldn't get back to start as a result. These numbers
# are grounded in this robot's own calibration, not copied from elsewhere - if a
# joint ever needs to be recalibrated, regenerate this dict from bus.calibration
# via bus._normalize() rather than hand-editing it again.
#
# gripper is NOT in degrees - so_follower.py configures it with
# MotorNormMode.RANGE_0_100 specifically (0-100, unlike the other 5 joints),
# so its limit lives in this same dict but means "percent open", not degrees.
JOINT_LIMITS_DEG = {
    "shoulder_pan": (-118.0, 118.0),
    "shoulder_lift": (-105.0, 105.0),
    "elbow_flex": (-98.0, 98.0),
    "wrist_flex": (-102.0, 102.0),
    "wrist_roll": (-179.0, 179.0),
    "gripper": (-8.0, 99.0),
}

# Per send_action() call - lerobot's own clip on how far any single command may
# move a motor from its *current* position.
MAX_RELATIVE_TARGET_DEG = 15.0

# Per move_to_xyz()/set_gripper_deg() call - refuse the whole move outright
# (before sending anything) if any joint would need to travel further than this
# from where it is right now. Small and deliberate while this code is still
# freshly tested; raise once real moves are confirmed safe and this stops being
# an obstacle for legitimate reaches across the workspace.
MAX_MOVE_DELTA_DEG = 40.0


def clamp_joint_deg(joint_deg: np.ndarray) -> np.ndarray:
    clamped = joint_deg.copy()
    for i, name in enumerate(ALL_JOINTS):
        lo, hi = JOINT_LIMITS_DEG[name]
        clamped[i] = np.clip(clamped[i], lo, hi)
    return clamped


class RobotController:
    def __init__(self, port: str = FOLLOWER_PORT):
        self.robot = SOFollower(
            SOFollowerRobotConfig(
                port=port,
                id="follower",
                use_degrees=True,
                max_relative_target=MAX_RELATIVE_TARGET_DEG,
            )
        )
        self.kin = build_kinematics()

    def connect(self) -> None:
        self.robot.connect(calibrate=False)  # reuse the existing stored calibration, don't re-run it

    def disconnect(self) -> None:
        self.robot.disconnect()

    def emergency_stop(self) -> None:
        """Cuts torque immediately - the arm goes limp and stops resisting
        gravity/motion. Not a controlled stop, just an immediate one."""
        for motor in ALL_JOINTS:
            self.robot.bus.write("Torque_Enable", motor, 0)

    def enable_torque(self) -> None:
        """Re-engages holding torque at wherever the arm currently is (e.g.
        after emergency_stop() + hand-positioning it) - deliberately does NOT
        just flip Torque_Enable back on, since each motor's Goal_Position
        register still holds whatever it was last commanded to *before*
        torque was cut, not the hand-moved position - re-enabling torque
        directly would yank the arm back toward that stale goal. Reads the
        actual current position first and re-sends it as the goal (a no-op
        move, safe with torque still off) so holding torque engages exactly
        where the arm already is."""
        current = self.get_joint_deg()  # Present_Position reads fine with torque off
        self.send_joint_deg(current)  # rewrite Goal_Position to match - no motion, torque still off
        for motor in ALL_JOINTS:
            self.robot.bus.write("Torque_Enable", motor, 1)

    def get_joint_deg(self) -> np.ndarray:
        obs = self.robot.get_observation()
        return np.array([obs[f"{j}.pos"] for j in ALL_JOINTS])

    def send_joint_deg(self, joint_deg: np.ndarray) -> None:
        action = {f"{j}.pos": float(v) for j, v in zip(ALL_JOINTS, clamp_joint_deg(joint_deg))}
        self.robot.send_action(action)

    def current_gripper_xyz(self) -> np.ndarray:
        return gripper_position(self.kin, self.get_joint_deg())

    def preview_move(self, xyz: tuple[float, float, float]) -> dict:
        """Computes (without moving) what move_to_xyz(xyz) would do. Returns a
        dict with current/target joints and the per-joint delta, for printing
        and confirming before actually committing to a move."""
        current = self.get_joint_deg()
        target_arm = solve_ik(self.kin, current, xyz)
        target = clamp_joint_deg(np.concatenate([target_arm, current[len(ARM_JOINTS):]]))
        delta = target - current
        return {
            "current_deg": current,
            "target_deg": target,
            "delta_deg": delta,
            "max_abs_delta_deg": float(np.max(np.abs(delta))),
            "reachable_xyz": gripper_position(self.kin, target),
        }

    def move_to_xyz(
        self,
        xyz: tuple[float, float, float],
        steps: int = 20,
        step_delay_s: float = 0.05,
        enforce_cap: bool = True,
        stall_check: bool = True,
    ) -> None:
        """Linearly interpolates joint space from the current pose to the IK
        solution for xyz, in `steps` increments.

        enforce_cap=True (the default, for direct one-shot calls) refuses outright
        if the *total* delta exceeds MAX_MOVE_DELTA_DEG. move_to_xyz_converge calls
        with enforce_cap=False: its whole point is covering a large total distance
        across several retries, each of which already only executes a small
        physical step (lerobot's own max_relative_target clamps every actual
        send_action to 15deg regardless of what's requested here) - the outright
        refusal was fighting that design rather than adding real protection on
        top of it, once total-distance triggered it mid-sequence.

        stall_check=True (default) watches actual-vs-commanded position as it
        moves and raises CollisionDetected (after retreating to the last
        known-good pose) if the arm stops making progress for several checks in
        a row - i.e. it hit something. Leave it on unless a caller has its own
        reason to move blind (there isn't one in this codebase yet)."""
        plan = self.preview_move(xyz)
        if enforce_cap and plan["max_abs_delta_deg"] > MAX_MOVE_DELTA_DEG:
            raise RuntimeError(
                f"move_to_xyz refused: largest single-joint delta would be "
                f"{plan['max_abs_delta_deg']:.1f}deg, over the {MAX_MOVE_DELTA_DEG}deg safety cap. "
                f"Move in smaller stages (or via move_to_xyz_converge), or raise MAX_MOVE_DELTA_DEG "
                f"once this path is trusted."
            )
        current, target = plan["current_deg"], plan["target_deg"]
        last_good = current.copy()
        stall_count = 0
        for i in range(1, steps + 1):
            interp = current + (target - current) * (i / steps)
            self.send_joint_deg(interp)
            time.sleep(step_delay_s)

            if not stall_check or i % STALL_CHECK_EVERY != 0:
                continue
            actual = self.get_joint_deg()
            lag = float(np.max(np.abs(actual[: len(ARM_JOINTS)] - interp[: len(ARM_JOINTS)])))
            if lag > STALL_THRESHOLD_DEG:
                stall_count += 1
            else:
                stall_count = 0
                last_good = actual
            if stall_count >= STALL_CONSECUTIVE:
                print(f"   [안전] 충돌 의심 (관절 오차 {lag:.1f}deg) - 진행 중단, 이전 위치로 복귀합니다.")
                self.send_joint_deg(last_good)
                time.sleep(0.3)
                raise CollisionDetected(
                    f"move_to_xyz aborted: joint lag {lag:.1f}deg exceeded {STALL_THRESHOLD_DEG}deg "
                    f"for {STALL_CONSECUTIVE} consecutive checks - retreated to last known-good pose."
                )

    def move_to_xyz_converge(
        self, xyz: tuple[float, float, float], tolerance_m: float = 0.005, max_iters: int = 15
    ) -> np.ndarray:
        """Repeatedly calls move_to_xyz toward the same target and re-checks the
        actual resulting position, instead of trusting one interpolated move to
        land exactly on target. lerobot's own max_relative_target (15deg/call)
        throttles how far a single send_action can move each joint regardless of
        what move_to_xyz's interpolation intends, so one call to a far-away target
        routinely stalls partway there - this converges by retrying with a fresh
        delta each time, the same fix already used for the gripper's open/close.
        Returns the final actual gripper xyz."""
        for _ in range(max_iters):
            current_xyz = self.current_gripper_xyz()
            if np.linalg.norm(np.array(xyz) - current_xyz) <= tolerance_m:
                return current_xyz
            # steps/step_delay_s slowed down from an earlier 10/0.04: a real
            # run hit CollisionDetected on a large (58deg) coordinated move at
            # that pace, and the lag looked like ordinary STS3215 catch-up lag
            # (this servo is genuinely slow to track fast coordinated multi-
            # joint commands, already noted elsewhere this session) rather
            # than a real block - a slower commanded pace should stop asking
            # for more than the servo can deliver without weakening real
            # collision detection, since a truly stuck arm stays stuck
            # regardless of how gently it's asked to move.
            self.move_to_xyz(xyz, steps=18, step_delay_s=0.06, enforce_cap=False)
            time.sleep(0.15)
        return self.current_gripper_xyz()

    def nudge_xy(
        self, dx: float, dy: float, steps: int = 6, step_delay_s: float = 0.03, stall_check: bool = True
    ) -> np.ndarray:
        """Small relative move in the table plane (x, y unchanged z) - the core
        primitive for visual servoing: each iteration asks for a small pixel-error
        -driven nudge, not an absolute target. Uses enforce_cap=False since these
        are deliberately tiny (a handful of mm/iteration) - the outright-refusal
        cap is for catching wildly wrong one-shot targets, not for this. Collision
        stall_check stays on by default - see move_to_xyz."""
        current_xyz = self.current_gripper_xyz()
        target = (current_xyz[0] + dx, current_xyz[1] + dy, current_xyz[2])
        self.move_to_xyz(target, steps=steps, step_delay_s=step_delay_s, enforce_cap=False, stall_check=stall_check)
        return self.current_gripper_xyz()

    def move_z(
        self, dz: float, steps: int = 10, step_delay_s: float = 0.04, stall_check: bool = True
    ) -> np.ndarray:
        """Small relative move in z only (straight up/down), same rationale as
        nudge_xy. This is the one used for the blind final descent onto the
        table in pick_place.py, so stall_check here is what actually catches
        "gripper came down on the cube/table wrong" instead of grinding into it."""
        current_xyz = self.current_gripper_xyz()
        target = (current_xyz[0], current_xyz[1], current_xyz[2] + dz)
        self.move_to_xyz(target, steps=steps, step_delay_s=step_delay_s, enforce_cap=False, stall_check=stall_check)
        return self.current_gripper_xyz()

    def set_gripper_pct(self, pct: float, steps: int = 10, step_delay_s: float = 0.03) -> None:
        """pct: 0-100 (this joint's actual unit - see JOINT_LIMITS_DEG's note on why
        gripper isn't in degrees like the other 5 joints). Confirmed against real
        hardware: 100 = open, 0 = closed."""
        current = self.get_joint_deg()
        delta = abs(pct - current[-1])
        if delta > MAX_MOVE_DELTA_DEG:
            raise RuntimeError(f"set_gripper_pct refused: {delta:.1f} delta exceeds {MAX_MOVE_DELTA_DEG} cap.")
        target = current.copy()
        target[-1] = pct
        for i in range(1, steps + 1):
            interp = current + (target - current) * (i / steps)
            self.send_joint_deg(interp)
            time.sleep(step_delay_s)

    def set_gripper_pct_converge(
        self, pct: float, tolerance: float = 3.0, max_iters: int = 15, steps: int = 8, step_delay_s: float = 0.03
    ) -> float:
        """Same retry-then-recheck idea as move_to_xyz_converge, for the gripper:
        set_gripper_pct's own 40deg one-shot cap refuses outright whenever the
        current position is more than 40 away from the target (e.g. fully closed
        -> fully open is a 100 jump), so a single call can't reach far targets.
        Re-reads the actual position each iteration and re-issues a fresh
        (<=35) delta instead. Returns the final actual position."""
        for _ in range(max_iters):
            current = self.get_joint_deg()[-1]
            if abs(current - pct) <= tolerance:
                return current
            step_target = current + max(-35.0, min(35.0, pct - current))
            self.set_gripper_pct(step_target, steps=steps, step_delay_s=step_delay_s)
            time.sleep(0.1)
        return self.get_joint_deg()[-1]
