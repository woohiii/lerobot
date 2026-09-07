"""Offline full-pose IK refinement for a plan made by plan_click_pick_place.

It reads a plan JSON, searches only the URDF kinematic model, and writes a
second JSON.  No serial device, camera device, motor driver, or actuator is
imported.  This is the fallback after fixed-wrist-roll IK cannot keep the
gripper sufficiently downward at a clicked target.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
TASK_DIR = THIS_DIR.parent / "task_red_cube_to_bin_new_gripper"
sys.path.insert(0, str(TASK_DIR))

import config  # noqa: E402, I001
from kinematics import build_kinematics, gripper_position, make_pose  # noqa: E402


@dataclass
class Candidate:
    """A pose-IK solution assessed solely against the URDF model."""

    joint_deg: list[float]
    target_rpy_deg: list[float]
    orientation_weight: float
    position_error_mm: float
    model_downward_cosine: float


def _within_limits(joints: np.ndarray) -> bool:
    return all(
        config.JOINT_LIMITS_DEG[name][0] <= q <= config.JOINT_LIMITS_DEG[name][1]
        for name, q in zip(config.ARM_JOINTS, joints, strict=True)
    )


def _solve_stage(kin, target_xyz: tuple[float, float, float], seed: np.ndarray) -> Candidate | None:
    """Find a close pose solution with a down-facing tool axis in the model."""
    feasible: list[Candidate] = []
    for weight in (0.0005, 0.001, 0.002, 0.005):
        for rx in range(120, 241, 30):
            for ry in range(-60, 61, 30):
                for rz in range(-180, 181, 30):
                    joints = seed.copy()
                    pose = make_pose(target_xyz, (float(rx), float(ry), float(rz)))
                    for _ in range(12):
                        joints = kin.inverse_kinematics(joints, pose, orientation_weight=weight)
                    if not _within_limits(joints):
                        continue
                    err_mm = float(np.linalg.norm(gripper_position(kin, joints) - target_xyz) * 1000.0)
                    down = float(-kin.forward_kinematics(joints)[2, 2])
                    if err_mm <= 5.0 and down >= 0.85:
                        feasible.append(Candidate(
                            [float(value) for value in joints], [float(rx), float(ry), float(rz)], weight, err_mm, down
                        ))
    if not feasible:
        return None
    return min(feasible, key=lambda c: (c.position_error_mm, -c.model_downward_cosine))


def main() -> int:
    """Refine a saved plan using simulation-only full-pose IK."""
    source = THIS_DIR / "latest_plan.json"
    output = THIS_DIR / "latest_orientation_ik_plan.json"
    if not source.exists():
        print(f"[blocked] plan missing: {source}")
        return 2
    plan = json.loads(source.read_text())
    home = np.array(plan["home"]["joint_deg"], dtype=float)
    home_source = plan["home"]["source"]
    readonly_home = THIS_DIR / "home_pose_readonly.json"
    if home_source == "synthetic_zero_seed_not_a_real_home" and readonly_home.exists():
        home_record = json.loads(readonly_home.read_text())
        home = np.array(home_record["arm_joint_deg"], dtype=float)
        home_source = f"read_only_observation:{readonly_home}"
    kin = build_kinematics()
    refined: dict[str, dict] = {}
    seed = home
    for stage in plan["stages"]:
        candidate = _solve_stage(kin, tuple(stage["xyz_m"]), seed)
        if candidate is None:
            refined[stage["name"]] = {"model_feasible": False}
            continue
        refined[stage["name"]] = {"model_feasible": True, **asdict(candidate)}
        seed = np.array(candidate.joint_deg)

    critical = ("grasp_descend", "place_descend")
    critical_ok = all(refined.get(name, {}).get("model_feasible") for name in critical)
    result = {
        "mode": "OFFLINE_IK_REFINEMENT_NO_SERIAL_NO_ACTUATION",
        "source_plan": str(source),
        "home_source": home_source,
        "home_joint_deg": [float(value) for value in home],
        "critical_descend_ik_pass": critical_ok,
        "execution_allowed": False,
        "execution_blockers": [
            "No real-home joint observation was supplied.",
            "Camera-to-base extrinsics/workspace require physical calibration validation.",
            "URDF collision warnings and real table clearance remain unvalidated.",
            "This program has no actuator implementation.",
        ],
        "stages": refined,
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[saved] {output}")
    print(f"[critical descend IK] {'PASS' if critical_ok else 'FAIL'}; execution remains blocked")
    return 0 if critical_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
