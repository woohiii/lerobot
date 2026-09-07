"""No-actuation click-to-plan tool for the SO-101 pick-and-place task.

This program intentionally never imports the SOFollower driver, opens a
serial port, or sends a motor command.  It only reads an RGB image and its
matching Astra depth .npy file, asks for two grasp-edge midpoint clicks and a
drop click, then writes an auditable plan JSON.  It is the preflight stage
before any real-arm execution is considered.

Run from the repository root (the Astra publisher may be running separately):
    uv run python custom_scripts/vision_pick_place/click_pick_place_safe/plan_click_pick_place.py

The default inputs are the frames published by astra_s_live.py.  The program
does not start that publisher and thus never opens the Astra device itself.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
TASK_DIR = THIS_DIR.parent / "task_red_cube_to_bin_new_gripper"
sys.path.insert(0, str(TASK_DIR))

import config  # noqa: E402, I001
from grasp_ik import solve_fixed_roll_ik, solved_position_error_m  # noqa: E402
from kinematics import build_kinematics  # noqa: E402
from perception import is_xy_within_safe_workspace, pixel_to_xy  # noqa: E402


DEFAULT_RGB = Path("/tmp/vsp_astra_rgb.png")
DEFAULT_DEPTH = Path("/tmp/vsp_astra_depth_mm.npy")
DEFAULT_OUTPUT = THIS_DIR / "latest_plan.json"
WINDOW = "SAFE PLAN: click two cube-edge midpoints, then drop point"


@dataclass
class Stage:
    """One non-actuating planned TCP pose and requested gripper state."""

    name: str
    xyz_m: list[float]
    gripper: str
    joint_deg: list[float]
    ik_position_error_mm: float
    tool_z_axis_base: list[float]
    model_downward_cosine: float


def _parse_home(value: str | None) -> np.ndarray:
    """Use a supplied/read-only home observation; never query hardware here."""
    if value is None:
        # A synthetic seed permits offline solver verification only.  It is
        # deliberately not called a home pose and makes the plan non-executable.
        return np.zeros(5, dtype=float)
    values = np.fromstring(value, sep=",", dtype=float)
    if values.size != 5:
        raise ValueError("--home-deg must contain exactly 5 arm joint degrees, comma separated")
    return values


def _parse_points(value: str) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Parse `x1,y1;x2,y2;x3,y3` for repeatable headless validation."""
    try:
        points = [tuple(int(v) for v in pair.split(",")) for pair in value.split(";")]
    except ValueError as exc:
        raise ValueError("--points format is x1,y1;x2,y2;x3,y3") from exc
    if len(points) != 3 or any(len(point) != 2 for point in points):
        raise ValueError("--points format is x1,y1;x2,y2;x3,y3")
    return points[0], points[1], points[2]


def _depth_at(depth_mm: np.ndarray, rgb_shape: tuple[int, int], px: tuple[int, int]) -> float | None:
    """Median of a small valid-depth neighbourhood, mapped RGB -> depth size."""
    h, w = rgb_shape
    dh, dw = depth_mm.shape[:2]
    x = int(px[0] * dw / w)
    y = int(px[1] * dh / h)
    patch = depth_mm[max(0, y - 3) : min(dh, y + 4), max(0, x - 3) : min(dw, x + 4)]
    valid = patch[patch > 0]
    return None if valid.size < 5 else float(np.median(valid))


def _height_delta_m(depth_mm: np.ndarray, rgb_shape: tuple[int, int], center: tuple[int, int]) -> float | None:
    """Conservative relative-height check, not an absolute 3-D transform.

    Astra depth is registered RGB/QVGA depth and its pinhole backprojection
    was previously shown inaccurate on this setup.  We use only its robust
    table-vs-object depth difference, exactly as a validation signal.
    """
    obj = _depth_at(depth_mm, rgb_shape, center)
    valid = depth_mm[depth_mm > 0]
    if obj is None or valid.size < 100:
        return None
    delta = (float(np.median(valid)) - obj) / 1000.0
    return delta if 0.003 <= delta <= 0.08 else None


def _clicks(frame: np.ndarray) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None:
    points: list[tuple[int, int]] = []
    prompts = [
        "1/3: cube side midpoint A",
        "2/3: opposite cube side midpoint B",
        "3/3: desired drop position",
    ]

    def callback(event, x, y, _flags, _userdata):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 3:
            points.append((x, y))

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 960, 720)
    cv2.setMouseCallback(WINDOW, callback)
    try:
        while len(points) < 3:
            canvas = frame.copy()
            for index, point in enumerate(points):
                cv2.circle(canvas, point, 7, (0, 255, 255), -1)
                cv2.putText(canvas, str(index + 1), (point[0] + 10, point[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(canvas, prompts[len(points)], (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 0), 2)
            cv2.putText(canvas, "q / ESC: cancel. This tool never moves the robot.", (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 180, 255), 1)
            cv2.imshow(WINDOW, canvas)
            key = cv2.waitKey(25) & 0xFF
            if key in (27, ord("q")):
                return None
    finally:
        cv2.destroyWindow(WINDOW)
    return points[0], points[1], points[2]


def _stage(kin, seed: np.ndarray, name: str, xyz: tuple[float, float, float], gripper: str, roll: float) -> tuple[Stage, np.ndarray]:
    joints = solve_fixed_roll_ik(kin, seed, xyz, roll)
    err_mm = solved_position_error_m(kin, joints, xyz) * 1000.0
    tool_z = kin.forward_kinematics(joints)[:3, 2]
    # Base +Z is the table normal in this existing URDF/calibration convention;
    # a tool +Z aligned with base -Z is a downward approach in the model.
    downward_cosine = float(-tool_z[2])
    return Stage(
        name,
        list(xyz),
        gripper,
        [float(x) for x in joints],
        float(err_mm),
        [float(x) for x in tool_z],
        downward_cosine,
    ), joints


def _save_overlay(
    frame: np.ndarray,
    edge_a: tuple[int, int],
    edge_b: tuple[int, int],
    drop_px: tuple[int, int],
    output_path: Path,
) -> None:
    """Save the exact click evidence used to make a plan for later review."""
    canvas = frame.copy()
    center = (round((edge_a[0] + edge_b[0]) / 2), round((edge_a[1] + edge_b[1]) / 2))
    cv2.line(canvas, edge_a, edge_b, (0, 255, 255), 2)
    for label, point, color in [("edge A", edge_a, (0, 255, 255)), ("edge B", edge_b, (0, 255, 255)), ("grasp center", center, (0, 255, 0)), ("drop", drop_px, (255, 0, 0))]:
        cv2.circle(canvas, point, 7, color, -1)
        cv2.putText(canvas, label, (point[0] + 9, point[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    cv2.putText(canvas, "PLAN ONLY - NO ROBOT ACTUATION", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 255), 2)
    cv2.imwrite(str(output_path), canvas)


def _write_attempt(output: Path, payload: dict) -> None:
    """Persist click evidence even when a safety gate rejects a plan."""
    attempt = output.with_name("latest_attempt.json")
    attempt.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> int:
    """Collect clicks and save a blocked, inspectable motion plan."""
    parser = argparse.ArgumentParser(description="SO-101 RGB-D click planner (no hardware actuation)")
    parser.add_argument("--rgb", type=Path, default=DEFAULT_RGB)
    parser.add_argument("--depth", type=Path, default=DEFAULT_DEPTH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--home-deg", help="five observed arm degrees; optional only for offline IK seed")
    parser.add_argument("--down-roll-deg", type=float, default=-90.0, help="candidate wrist-roll value; must be physically verified later")
    parser.add_argument("--points", help="headless test: x1,y1;x2,y2;x3,y3 (skips GUI clicks)")
    args = parser.parse_args()

    frame = cv2.imread(str(args.rgb))
    if frame is None:
        print(f"[blocked] RGB frame not readable: {args.rgb}")
        return 2
    try:
        depth_mm = np.load(args.depth)
    except (OSError, ValueError) as exc:
        print(f"[blocked] Depth frame not readable: {args.depth} ({exc})")
        return 2
    if depth_mm.ndim != 2:
        print("[blocked] depth must be a HxW uint16/mm array")
        return 2

    try:
        picked = _parse_points(args.points) if args.points else _clicks(frame)
    except ValueError as exc:
        print(f"[blocked] {exc}")
        return 2
    if picked is None:
        print("[cancelled] no plan written")
        return 0
    edge_a, edge_b, drop_px = picked
    h, w = frame.shape[:2]
    all_points = (edge_a, edge_b, drop_px)
    attempt = {
        "created_utc": datetime.now(UTC).isoformat(),
        "mode": "PLAN_ONLY_NO_SERIAL_NO_ACTUATION",
        "edge_midpoints_px": [edge_a, edge_b],
        "drop_px": drop_px,
        "result": "pending_validation",
    }
    if any(not (0 <= x < w and 0 <= y < h) for x, y in all_points):
        attempt.update(result="blocked", reason="clicked point is outside the RGB image")
        _write_attempt(args.output, attempt)
        print("[blocked] clicked point is outside the RGB image")
        return 2
    center = (round((edge_a[0] + edge_b[0]) / 2), round((edge_a[1] + edge_b[1]) / 2))
    edge_distance_px = float(np.hypot(edge_b[0] - edge_a[0], edge_b[1] - edge_a[1]))
    attempt["grasp_center_px"] = center
    attempt["grasp_edge_spacing_px"] = edge_distance_px
    if not 10.0 <= edge_distance_px <= 250.0:
        attempt.update(result="blocked", reason="grasp edge spacing is implausible")
        _write_attempt(args.output, attempt)
        print(f"[blocked] grasp-edge spacing {edge_distance_px:.1f}px is implausible; click opposite cube edges")
        return 2
    grasp_xy, drop_xy = pixel_to_xy(*center), pixel_to_xy(*drop_px)
    if grasp_xy is None or drop_xy is None:
        attempt.update(result="blocked", reason="homography unavailable")
        _write_attempt(args.output, attempt)
        print("[blocked] homography.json is unavailable; do not infer a robot-base target")
        return 2
    if not is_xy_within_safe_workspace(*grasp_xy) or not is_xy_within_safe_workspace(*drop_xy):
        attempt.update(result="blocked", reason="clicked target outside calibrated workspace", grasp_xy=grasp_xy, drop_xy=drop_xy)
        _write_attempt(args.output, attempt)
        print("[blocked] a clicked target lies outside the calibrated workspace margin")
        return 2

    height_m = _height_delta_m(depth_mm, frame.shape[:2], center)
    # Existing calibrated table height is used; depth only confirms that an
    # object is present.  No unsafe absolute camera-frame Z conversion.
    grasp_z = config.TABLE_Z + (height_m if height_m is not None else 0.02) - config.DESCEND_MARGIN_M
    hover_z = max(grasp_z + 0.06, config.SEARCH_HOVER_XYZ[2])
    lift_z = grasp_z + 0.04  # user-specified 4 cm lift
    place_z = config.TABLE_Z + 0.01

    home = _parse_home(args.home_deg)
    kin = build_kinematics()
    stages: list[Stage] = []
    seed = home
    for name, xyz, gripper in [
        ("pre_grasp", (grasp_xy[0], grasp_xy[1], hover_z), "open"),
        ("grasp_descend", (grasp_xy[0], grasp_xy[1], grasp_z), "open"),
        ("grasp_close", (grasp_xy[0], grasp_xy[1], grasp_z), "close"),
        ("lift_hold_4cm", (grasp_xy[0], grasp_xy[1], lift_z), "closed"),
        ("pre_place", (drop_xy[0], drop_xy[1], hover_z), "closed"),
        ("place_descend", (drop_xy[0], drop_xy[1], place_z), "closed"),
        ("release", (drop_xy[0], drop_xy[1], place_z), "open"),
        ("retreat", (drop_xy[0], drop_xy[1], hover_z), "open"),
    ]:
        stage, seed = _stage(kin, seed, name, xyz, gripper, args.down_roll_deg)
        stages.append(stage)

    max_error = max(stage.ik_position_error_mm for stage in stages)
    min_downward = min(stage.model_downward_cosine for stage in stages)
    grasp_downward = next(stage.model_downward_cosine for stage in stages if stage.name == "grasp_descend")
    place_downward = next(stage.model_downward_cosine for stage in stages if stage.name == "place_descend")
    overlay_path = args.output.with_suffix(".png")
    _save_overlay(frame, edge_a, edge_b, drop_px, overlay_path)
    plan = {
        "created_utc": datetime.now(UTC).isoformat(),
        "mode": "PLAN_ONLY_NO_SERIAL_NO_ACTUATION",
        "inputs": {
            "rgb": str(args.rgb), "depth": str(args.depth), "edge_midpoints_px": [edge_a, edge_b],
            "grasp_center_px": center, "drop_px": drop_px, "grasp_edge_spacing_px": edge_distance_px,
            "grasp_axis_image_deg": float(np.degrees(np.arctan2(edge_b[1] - edge_a[1], edge_b[0] - edge_a[0])),),
            "click_overlay": str(overlay_path),
        },
        "depth_validation": {"relative_object_height_m": height_m, "absolute_3d_used": False},
        "home": {"joint_deg": [float(x) for x in home], "source": "user_supplied" if args.home_deg else "synthetic_zero_seed_not_a_real_home"},
        "wrist_down": {"candidate_wrist_roll_deg": args.down_roll_deg, "physically_verified": False},
        "safety_gate": {
            "execution_allowed": False,
            "reasons": [
                "This program is planning-only and has no actuator implementation.",
                "Camera-to-base absolute 3-D extrinsics are not validated for Astra S.",
                "The wrist-down roll value has not been physically verified.",
                "Real collision/table-clearance validation remains required.",
            ],
            "max_ik_position_error_mm": max_error,
            "minimum_model_downward_cosine": min_downward,
            "grasp_descend_model_downward_cosine": grasp_downward,
            "place_descend_model_downward_cosine": place_downward,
            "grasp_orientation_model_check": grasp_downward >= 0.85,
        },
        "stages": [asdict(stage) for stage in stages],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2) + "\n")
    print(f"[plan saved] {args.output}")
    print(f"[click evidence] {overlay_path}")
    print(f"  grasp={tuple(round(v, 4) for v in (grasp_xy[0], grasp_xy[1], grasp_z))}, lift=4 cm, max IK error={max_error:.2f} mm")
    print("  EXECUTION BLOCKED: this is an offline plan only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
