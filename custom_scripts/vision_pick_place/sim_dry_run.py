"""Pure-simulation dry run of visual_servo_pick_place.py's control logic - NO
robot connection, NO camera device, nothing physical touched. Exercises the
real search_for_target / coarse_center / estimate_jacobian / servo_to_target /
descend_and_grasp functions unmodified, against a synthetic robot + synthetic
camera model, to sanity-check today's two changes before ever running them on
hardware:

  1. Broyden's-rank-one online Jacobian update in servo_to_target (see that
     function's docstring/comments for the citation and rationale).
  2. Gripper-position-based grasp verification in descend_and_grasp.
  3. GRASP_TARGET_PX (servo target is the gripper's jaw position in-frame,
     not raw image center - see that constant's comment).
  4. Shape filtering (solidity + aspect ratio + max-area cap) in
     cube_detector.py so a hand/arm/cable in frame can't out-vote the actual
     cube/bin just by being the largest same-colored blob.

descend_and_grasp now also calls estimate_cube_height_m(), which reads real
files (ASTRA_RGB_FRAME_PATH / ASTRA_DEPTH_MM_PATH) that astra_s_live.py
publishes on real hardware - this simulation monkeypatches it to always
return None (Astra unavailable), both because there's no real Astra data to
give it here and to keep this deterministic regardless of whatever stale
files a previous real run happened to leave on disk.

The synthetic camera model deliberately makes the true image Jacobian vary
with position (scale grows on approach, like real perspective does) so a
FIXED Jacobian would be expected to drift, the same failure mode documented
from real runs (12.9px -> drifted back to 95px+). This is what actually tests
whether Broyden's per-step update helps, not just whether the code runs.

Run: python3 sim_dry_run.py   (works in either venv - no cv2 GUI, no hardware
SDKs needed at all; only numpy + the pure-Python control-flow functions).
"""

from __future__ import annotations

import numpy as np

import visual_servo_pick_place as vsp
from cube_detector import Detection, detect_black_bin, detect_red_cube
from robot_control import CollisionDetected

# See module docstring - keeps the sim isolated from real Astra files on disk.
# Saved before patching so test_estimate_cube_height() below can still call
# the real implementation directly.
_ORIGINAL_estimate_cube_height_m = vsp.estimate_cube_height_m
vsp.estimate_cube_height_m = lambda: None

# Same reasoning, for search_for_target's Astra-homography coarse-position
# guess (estimate_xy_from_astra) - also reads real published files, also
# patched out here so the servo-loop sim cases exercise the blind-grid
# fallback path deterministically instead of racing/depending on whatever
# real Astra state happens to be on disk.
_ORIGINAL_estimate_xy_from_astra = vsp.estimate_xy_from_astra
vsp.estimate_xy_from_astra = lambda detect_fn: None

FRAME_W, FRAME_H = vsp.FRAME_W, vsp.FRAME_H
RNG = np.random.default_rng(7)


class FakeRobotController:
    """Implements exactly the RobotController surface visual_servo_pick_place.py
    calls, backed by a purely virtual xyz - no serial port, no motors. Can be
    told to simulate a collision at a given trigger point, and to simulate
    whether an object is actually between the gripper jaws when it closes."""

    def __init__(self, start_xyz, collide_after_n_moves=None, object_present=True):
        self._xyz = np.array(start_xyz, dtype=float)
        self._n_moves = 0
        self._collide_after_n_moves = collide_after_n_moves
        self._object_present = object_present
        self._gripper_pct = 100.0

    def current_gripper_xyz(self):
        return self._xyz.copy()

    def get_joint_deg(self):
        return np.zeros(6)  # not used by any of the functions under test

    def _maybe_collide(self):
        self._n_moves += 1
        if self._collide_after_n_moves is not None and self._n_moves >= self._collide_after_n_moves:
            raise CollisionDetected("[sim] 시뮬레이션된 충돌")

    def move_to_xyz(self, xyz, steps=20, step_delay_s=0.05, enforce_cap=True, stall_check=True):
        self._maybe_collide()
        self._xyz = np.array(xyz, dtype=float)

    def move_to_xyz_converge(self, xyz, tolerance_m=0.005, max_iters=15):
        self.move_to_xyz(xyz)
        return self.current_gripper_xyz()

    def nudge_xy(self, dx, dy, steps=6, step_delay_s=0.03, stall_check=True):
        self._maybe_collide()
        self._xyz[0] += dx
        self._xyz[1] += dy
        return self.current_gripper_xyz()

    def move_z(self, dz, steps=10, step_delay_s=0.04, stall_check=True):
        self._maybe_collide()
        self._xyz[2] += dz
        return self.current_gripper_xyz()

    def set_gripper_pct_converge(self, pct, tolerance=3.0, max_iters=15, steps=8, step_delay_s=0.03):
        # Simulates the real physical fact this update relies on: closing on
        # an actual object stops short of the commanded fully-closed target.
        if pct <= 5.0 and self._object_present:
            self._gripper_pct = 24.0  # "wedged against the cube" - plausible stop point
        else:
            self._gripper_pct = pct
        return self._gripper_pct


def make_synthetic_detect_fn(rc: FakeRobotController, target_xy, edge_bias=(0.0, 0.0)):
    """Returns a detect_fn(frame) that computes where the target WOULD appear
    in the wrist camera given rc's current virtual xyz, using a J_true that
    scales up as the arm gets closer to target_xy (mimicking real perspective
    change - the thing that made a fixed-J estimate go stale for real).
    edge_bias offsets the very first (pre-move) reading toward a frame edge,
    to exercise coarse_center's edge-start path without needing an
    unrealistically large starting offset from target_xy."""
    call_count = 0

    def detect_fn(frame):
        nonlocal call_count
        call_count += 1
        dx, dy = rc.current_gripper_xyz()[0] - target_xy[0], rc.current_gripper_xyz()[1] - target_xy[1]
        dist = float(np.hypot(dx, dy))
        # Scale (px per meter of arm displacement) grows ~10x as the arm
        # approaches - real coarse-center steps at a distance produced modest
        # shifts (e.g. a real 0.025m step moved a target ~7px, local gain
        # ~280px/m) while a real close-in estimate_jacobian readout was in
        # the thousands - this range is picked to match both regimes seen on
        # actual hardware, not just "big number goes up".
        # 800 (far) -> 8000 (close) px/m - matches the real estimate_jacobian
        # readout from 2026-08-26 (singular values 8152 and 1233 px/m) at the
        # close end once combined with the anisotropic base matrix below (its
        # smaller singular value is ~0.16x the larger one). An earlier, much
        # smaller scale (300->3000) made the sim's "sloppy" direction far
        # less sensitive than real hardware ever showed, which made
        # PHYSICAL_TOLERANCE_M look unreachable in sim even though the real
        # anisotropy ratio was matched - magnitude matters too, not just ratio.
        scale = 800.0 * (1.0 + 9.0 * np.exp(-dist / 0.02))
        # Anisotropic on purpose (condition number ~6, matching the same real
        # readout's ratio 8152/1233 ~= 6.6) - a near-isotropic J_true here
        # would never exercise PHYSICAL_TOLERANCE_M's whole reason for
        # existing (pixel-space "centered" can still be several mm off in the
        # sloppy direction; see that constant's comment in
        # visual_servo_pick_place.py).
        J_true = scale * np.array([[1.0, 0.05], [-0.08, 0.16]])
        pixel_offset = J_true @ np.array([dx, dy])
        # Bias fades out over the first few reads so it only affects the
        # initial "where was it found" moment, not the whole closed loop.
        bias_weight = max(0.0, 1.0 - 0.15 * call_count)
        bias = np.array(edge_bias) * bias_weight
        noise = RNG.normal(0, 1.5, size=2)  # a few px of detection jitter, same as real cameras
        # Ground truth: when the arm's jaws are exactly at target_xy (dx=dy=0),
        # the target must appear at GRASP_TARGET_PX by construction - that's
        # the point servo_to_target now drives error to zero against (see
        # that constant's comment in visual_servo_pick_place.py for why it's
        # not IMG_CENTER). Using IMG_CENTER here instead would silently test
        # against the wrong ground truth after today's fix.
        cx, cy = vsp.GRASP_TARGET_PX + pixel_offset + bias + noise
        if not (0 <= cx <= FRAME_W and 0 <= cy <= FRAME_H):
            return None
        return Detection(cx=float(cx), cy=float(cy), bbox=(int(cx) - 10, int(cy) - 10, 20, 20), area=400.0)

    return detect_fn


class DummyCap:
    def read(self):
        return True, np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)


def run_case(label, start_xyz, target_xy, object_present, collide_after=None, edge_bias=(0.0, 0.0)):
    print(f"\n{'=' * 70}\n[시뮬레이션] {label}\n{'=' * 70}")
    rc = FakeRobotController(start_xyz, collide_after_n_moves=collide_after, object_present=object_present)
    cap = DummyCap()
    detect_fn = make_synthetic_detect_fn(rc, target_xy, edge_bias=edge_bias)

    ok = vsp.servo_to_target(rc, cap, detect_fn, f"가상 타겟({label})")
    print(f"-> servo_to_target 결과: {ok}, 최종 xyz={rc.current_gripper_xyz()}")
    if not ok:
        return

    grasped = vsp.descend_and_grasp(rc)
    print(f"-> descend_and_grasp 결과: grasped={grasped} (object_present={object_present}였음)")
    assert grasped == object_present, "grasp-verification 로직이 시뮬레이션 정답과 어긋남!"
    print("   [검증 통과] grasp 판정이 시뮬레이션 정답과 일치합니다.")


def main():
    # HOME_XYZ = (0.23, 0.0, 0.13) - search_for_target always starts its
    # sweep there regardless of start_xyz, so target_xy is placed a realistic
    # few cm away from it (not from start_xyz) to get a gentle low-gain start.
    home = vsp.HOME_XYZ

    # Case 1: target starts well within view, near center - the common case.
    # object_present=True checks the grasp-success branch.
    run_case(
        "정상 접근 + 실제로 집힘",
        start_xyz=home,
        target_xy=(home[0] - 0.06, home[1] + 0.02),
        object_present=True,
    )

    # Case 2: same approach, but nothing is actually between the jaws when it
    # closes (today's real bug) - checks the grasp-FAILURE branch is caught
    # instead of silently treated as success.
    run_case(
        "정상 접근 + 실제로는 못 집음 (오늘 실제로 겪은 버그 재현)",
        start_xyz=home,
        target_xy=(home[0] - 0.06, home[1] + 0.02),
        object_present=False,
    )

    # Case 3: target found near a frame edge on the very first read -
    # exercises coarse_center's edge-start path before the Jacobian probe,
    # same as the real x=5px case documented earlier.
    run_case(
        "화면 가장자리에서 시작 (coarse-center 경로)",
        start_xyz=home,
        target_xy=(home[0] - 0.04, home[1] - 0.02),
        object_present=True,
        edge_bias=(-300.0, 0.0),
    )

    test_estimate_cube_height()
    test_estimate_xy_from_astra()
    test_hand_rejection()


def test_hand_rejection():
    """A hand/arm (or any intruding object) in the same color range as the
    cube/bin used to be able to win detection just by being the largest
    blob - see cube_detector.py's MIN_SOLIDITY/ASPECT_RATIO_RANGE comment.
    Builds a synthetic frame with the real target (small, solid, square) and
    a much bigger, star-shaped ("finger"-like, low-solidity) same-colored
    intruder, and checks detection still finds the real target, not the
    intruder - this is what would have silently broken before today's fix
    (max(contours, key=area) would have picked the star)."""
    import math

    import cv2

    print(f"\n{'=' * 70}\n[시뮬레이션] 손/팔 오탐 거부 테스트\n{'=' * 70}")

    def make_frame(color_bgr):
        frame = np.full((480, 640, 3), 220, dtype=np.uint8)
        cv2.rectangle(frame, (300, 200), (340, 240), color_bgr, -1)  # the real target
        pts = []
        for i in range(10):
            r = 90 if i % 2 == 0 else 30
            ang = math.pi * i / 5
            pts.append((int(150 + r * math.cos(ang)), int(150 + r * math.sin(ang))))
        cv2.fillPoly(frame, [np.array(pts)], color_bgr)  # a much bigger, low-solidity "hand"
        return frame

    cube_frame = make_frame((30, 30, 200))  # saturated red, BGR
    det = detect_red_cube(cube_frame)
    print(f"  큐브: {det}")
    assert det is not None, "빨간 사각형을 아예 못 찾음"
    assert 290 <= det.cx <= 350 and 190 <= det.cy <= 250, f"별 모양(손) 오탐: {det}"
    print("  [검증 통과] 손 모양 무시하고 실제 큐브를 찾음.")

    bin_frame = make_frame((20, 20, 20))  # near-black, BGR
    det = detect_black_bin(bin_frame)
    print(f"  쓰레기통: {det}")
    assert det is not None, "검은 사각형을 아예 못 찾음"
    assert 290 <= det.cx <= 350 and 190 <= det.cy <= 250, f"별 모양(손) 오탐: {det}"
    print("  [검증 통과] 손 모양 무시하고 실제 쓰레기통을 찾음.")


def test_estimate_cube_height():
    """Unlike the servo cases above, this exercises estimate_cube_height_m()'s
    real (non-monkeypatched) logic directly: writes a synthetic Astra RGB
    frame + depth array and calls the real function against them, checking
    it recovers the known height.

    2026-08-26: this used to write to the real published paths
    (ASTRA_RGB_FRAME_PATH/ASTRA_DEPTH_MM_PATH) and failed intermittently -
    astra_s_live.py was genuinely running at the time (a real, normal thing
    to have running) and racing this test's writes, sometimes clobbering the
    synthetic RGB or depth file with real camera data between the two
    writes/before the read. estimate_cube_height_m() takes overridable
    rgb_path/depth_path specifically so this test can use throwaway temp
    paths instead of the shared production ones - no coordination with
    whatever else happens to be running required."""
    import os
    import tempfile

    import cv2

    print(f"\n{'=' * 70}\n[시뮬레이션] estimate_cube_height_m() 단위 테스트\n{'=' * 70}")

    color = np.full((480, 640, 3), 200, dtype=np.uint8)  # light gray "table"
    cv2.rectangle(color, (300, 200), (340, 240), (30, 30, 200), -1)  # a red square (BGR)

    depth_mm = np.full((240, 320), 500, dtype=np.uint16)  # table at 500mm, matches color's scale (2x downsampled)
    KNOWN_HEIGHT_MM = 25
    depth_mm[100:120, 150:170] = 500 - KNOWN_HEIGHT_MM  # cube region closer to camera by the known height

    with tempfile.TemporaryDirectory() as tmpdir:
        rgb_path = os.path.join(tmpdir, "rgb.png")
        depth_path = os.path.join(tmpdir, "depth_mm.npy")
        cv2.imwrite(rgb_path, color)
        np.save(depth_path.replace(".npy", ""), depth_mm)  # np.save appends .npy itself

        # the real function, not the servo-test monkeypatch (which only
        # patches the zero-arg call visual_servo_pick_place.py's own code
        # makes internally - calling the underlying function directly with
        # explicit paths bypasses that entirely)
        height = _ORIGINAL_estimate_cube_height_m(rgb_path=rgb_path, depth_path=depth_path)
        print(f"   추정 높이: {height}")
        assert height is not None, "높이 추정이 None을 반환함 (합성 데이터인데도 실패)"
        expected = KNOWN_HEIGHT_MM / 1000.0
        assert abs(height - expected) < 0.003, f"추정치 {height} != 기대값 {expected} (허용오차 3mm)"
        print(f"   [검증 통과] 추정 높이 {height*1000:.1f}mm ≈ 실제 {KNOWN_HEIGHT_MM}mm")


def test_estimate_xy_from_astra():
    """Same idea as test_estimate_cube_height, for the new homography-based
    coarse xy estimate: a known (pure-affine, no perspective skew) homography
    applied to a red square at a known pixel should recover a known robot-
    frame xy, checked against the real (non-monkeypatched)
    estimate_xy_from_astra with an isolated temp frame - not the real
    homography.json (whatever it happens to contain) or the real published
    Astra frame."""
    import os
    import tempfile

    import cv2
    from cube_detector import detect_red_cube

    print(f"\n{'=' * 70}\n[시뮬레이션] estimate_xy_from_astra() 단위 테스트\n{'=' * 70}")

    color = np.full((480, 640, 3), 200, dtype=np.uint8)
    px, py = 300, 200
    cv2.rectangle(color, (px - 20, py - 20), (px + 20, py + 20), (30, 30, 200), -1)

    # pure affine (bottom row [0,0,1]) so the expected mapping is exact and
    # trivial to hand-check: x = a*px + b, y = c*py + d.
    a, b, c, d = 0.001, 0.05, -0.0008, 0.6
    H = np.array([[a, 0.0, b], [0.0, c, d], [0.0, 0.0, 1.0]])
    expected_x, expected_y = a * px + b, c * py + d

    with tempfile.TemporaryDirectory() as tmpdir:
        rgb_path = os.path.join(tmpdir, "rgb.png")
        cv2.imwrite(rgb_path, color)

        result = _ORIGINAL_estimate_xy_from_astra(detect_red_cube, rgb_path=rgb_path, homography=H)
        print(f"   추정 xy: {result} (기대값: ({expected_x:.4f}, {expected_y:.4f}))")
        assert result is not None, "xy 추정이 None을 반환함 (합성 데이터인데도 실패)"
        assert abs(result[0] - expected_x) < 1e-6 and abs(result[1] - expected_y) < 1e-6, (
            f"추정치 {result} != 기대값 ({expected_x}, {expected_y})"
        )
        print("   [검증 통과] 호모그래피 매핑이 정확히 복원됨.")


if __name__ == "__main__":
    main()
