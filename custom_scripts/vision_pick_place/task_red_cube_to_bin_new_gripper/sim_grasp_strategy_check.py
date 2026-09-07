"""Virtual (no-hardware) Monte Carlo check: does FINE_SERVO's closed-loop
wrist-cam visual servo actually buy a higher grasp-success probability than
a blind IK-only open-loop descend (click_grasp_bimanual.py's current
approach for arbitrary click targets)?

This is a structure/parameter sanity check, NOT a physics simulation and NOT
a substitute for real-hardware results (see the design guide PDF's own
warning, section 13.5: simulation success != real success). It reuses this
task's actual measured config.py constants instead of guessed numbers, so
the comparison is grounded in this hardware's real tolerances.

Model:
  - Both scenarios start from the same IK-hover positioning error (one IK
    move's accumulated calibration + servo error, modeled as isotropic
    Gaussian noise on the horizontal plane).
  - Open-loop (scenario A) grasps directly from that error.
  - FINE_SERVO (scenario B) then runs closed-loop pixel-space correction:
    each iteration halves the remaining error (SERVO_GAIN=0.5 applied to
    the Jacobian-inverse-mapped error, config.py) but is limited by the
    real anisotropic image Jacobian (measured singular-value ratio ~6.6x,
    task_state_machine.fine_servo docstring) which makes one axis converge
    slower than the other, and stops at config.PHYSICAL_TOLERANCE_M or
    config.MAX_SERVO_ITERS, whichever first.
  - A grasp succeeds if final horizontal error stays within the gripper's
    real closing margin (approximated here via config.PHYSICAL_TOLERANCE_M
    as the position tolerance the system was actually tuned to need).
  - config.MAX_GRASP_ATTEMPTS independent retries are modeled to show the
    system-level success rate actually deployed (task_state_machine.run's
    retry loop), not just a single attempt.

Run: uv run python3 custom_scripts/vision_pick_place/task_red_cube_to_bin_new_gripper/sim_grasp_strategy_check.py
"""

from __future__ import annotations

import numpy as np

import config

N_TRIALS = 10_000
SEED = 7

# IK-hover positioning error before any correction: has no directly-measured
# constant of its own, so it's set relative to config.COARSE_TARGET_PX /
# config's own Jacobian-inverse relationship, expressed directly in meters
# using PHYSICAL_TOLERANCE_M as the unit this system was tuned around: an
# open-loop IK move typically lands within a few multiples of the tolerance
# FINE_SERVO was built to close, per the real grasp failures config.py's
# TABLE_Z/GRASP_TARGET_PX comments describe (tens of mm calibration errors
# before those were re-measured).
INITIAL_ERROR_SIGMA_M = 6.0 * config.PHYSICAL_TOLERANCE_M

# Anisotropic Jacobian: measured singular values 8152 vs 1233 px/m (~6.6x),
# see task_state_machine.fine_servo docstring. The low-sensitivity axis
# converges proportionally slower per servo iteration.
JACOBIAN_ANISOTROPY = 8152.0 / 1233.0


def simulate_open_loop(rng: np.random.Generator) -> np.ndarray:
    """Final horizontal error (m) with no correction - straight IK hover -> descend."""
    return rng.normal(0.0, INITIAL_ERROR_SIGMA_M, size=(N_TRIALS, 2))


def simulate_fine_servo(rng: np.random.Generator) -> np.ndarray:
    """Final horizontal error (m) after FINE_SERVO's closed-loop correction,
    starting from the same initial error as the open-loop scenario."""
    err = rng.normal(0.0, INITIAL_ERROR_SIGMA_M, size=(N_TRIALS, 2))
    per_axis_gain = np.array([config.SERVO_GAIN, config.SERVO_GAIN / JACOBIAN_ANISOTROPY])
    tol = config.PHYSICAL_TOLERANCE_M
    for _ in range(config.MAX_SERVO_ITERS):
        dist = np.linalg.norm(err, axis=1)
        converged = dist <= tol
        if np.all(converged):
            break
        # detection noise added each iteration, same order of magnitude as
        # BROYDEN_MIN_STEP_M (the step size below which the code itself
        # treats a correction as noise-dominated, config.py).
        noise = rng.normal(0.0, config.BROYDEN_MIN_STEP_M * 0.3, size=err.shape)
        step = err * per_axis_gain
        err = np.where(converged[:, None], err, err - step + noise)
    return err


def grasp_success(final_err_m: np.ndarray) -> np.ndarray:
    """A grasp succeeds if the final horizontal error is within the
    tolerance the system's own closed-loop servo was tuned to require."""
    return np.linalg.norm(final_err_m, axis=1) <= config.PHYSICAL_TOLERANCE_M


def success_rate_with_retries(single_attempt_success: np.ndarray, attempts: int) -> float:
    """Independent-retry model of task_state_machine.run's MAX_GRASP_ATTEMPTS
    loop: P(at least one success in `attempts` independent tries)."""
    p = float(np.mean(single_attempt_success))
    return 1.0 - (1.0 - p) ** attempts


def main() -> None:
    rng = np.random.default_rng(SEED)

    open_loop_err = simulate_open_loop(rng)
    fine_servo_err = simulate_fine_servo(rng)

    open_loop_ok = grasp_success(open_loop_err)
    fine_servo_ok = grasp_success(fine_servo_err)

    p_open_1 = float(np.mean(open_loop_ok))
    p_open_n = success_rate_with_retries(open_loop_ok, config.MAX_GRASP_ATTEMPTS)
    p_servo_1 = float(np.mean(fine_servo_ok))
    p_servo_n = success_rate_with_retries(fine_servo_ok, config.MAX_GRASP_ATTEMPTS)

    print("=== 가상 검증: 오픈루프(click_grasp_bimanual) vs FINE_SERVO(legacy) ===")
    print(f"N_TRIALS={N_TRIALS}, PHYSICAL_TOLERANCE_M={config.PHYSICAL_TOLERANCE_M*1000:.1f}mm, "
          f"MAX_GRASP_ATTEMPTS={config.MAX_GRASP_ATTEMPTS}, 초기오차 sigma={INITIAL_ERROR_SIGMA_M*1000:.1f}mm")
    print()
    print(f"{'방식':<20}{'1회 성공률':>12}{f'{config.MAX_GRASP_ATTEMPTS}회 재시도 성공률':>18}")
    print(f"{'오픈루프 (IK만)':<20}{p_open_1:>11.1%}{p_open_n:>17.1%}")
    print(f"{'FINE_SERVO (legacy)':<20}{p_servo_1:>11.1%}{p_servo_n:>17.1%}")
    print()
    print("주의: 이는 config.py 실측 상수를 이용한 구조/파라미터 검증용 몬테카를로")
    print("시뮬레이션이며, 실제 하드웨어 성공률을 대체하지 않는다 (설계 가이드 13.5절).")

    assert p_servo_1 >= p_open_1, "FINE_SERVO closed-loop correction should not underperform open-loop"


if __name__ == "__main__":
    main()
