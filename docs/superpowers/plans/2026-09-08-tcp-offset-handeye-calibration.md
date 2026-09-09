# TCP Offset Hand-Eye Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Save robot joint poses during manual hand-eye collection and jointly estimate the camera-to-base transform and fixed TCP offset.

**Architecture:** Extend the collector to save `joint_points.npy` alongside camera/base points. Add a calibration solver that uses the SO-101 FK pose for each joint vector and optimizes camera-to-base rotation/translation plus a flange-to-TCP offset. The CLI writes `hand_eye.json` only when the resulting RMS is below the requested threshold.

**Tech Stack:** Python 3.12, NumPy, SciPy least-squares, existing LeRobot `RobotKinematics`, pytest, uv.

**Spec:** User-requested four-step TCP refinement: save joint angles, compute FK TCP poses, jointly optimize camera transform and TCP offset, replace `hand_eye.json` only after RMS validation.

## Global Constraints

- Preserve existing markerless/manual collection behavior and add new files or backward-compatible options only.
- Keep strict depth behavior: zero-depth selections are retried, never silently substituted by default.
- Require at least six non-degenerate correspondences for joint TCP calibration.
- Do not overwrite a valid calibration when the new RMS exceeds the requested threshold.

### Task 1: Persist joint poses during collection

**Files:**
- Modify: `custom_scripts/vision_pick_place/so101_graspnet_pick_and_place/collect_handeye_points.py`
- Test: `custom_scripts/vision_pick_place/so101_graspnet_pick_and_place/test_collect_handeye_points.py`

**Interfaces:**
- Produces `/tmp/.../joint_points.npy` with shape `(N, len(ARM_JOINTS))`.

- [ ] Add a failing test for validating a joint-point array shape and finite values.
- [ ] Run the focused test and confirm it fails before implementation.
- [ ] Capture `arm.get_joint_deg()` at the same instant as each base point and save `joint_points.npy`.
- [ ] Run focused and package tests.

### Task 2: Implement joint TCP calibration solver

**Files:**
- Modify: `custom_scripts/vision_pick_place/so101_graspnet_pick_and_place/calibration.py`
- Test: `custom_scripts/vision_pick_place/so101_graspnet_pick_and_place/test_calibration.py`

**Interfaces:**
- Add `solve_joint_tcp_transform(camera_points, joint_points, kin) -> JointCalibrationResult`.
- Result contains `camera_to_base`, `tcp_offset_m`, and `rms_error_m`.

- [ ] Add a synthetic failing test with a known transform, known flange poses, and known TCP offset.
- [ ] Run the test and verify the missing solver causes failure.
- [ ] Use SciPy `least_squares` over rotation-vector, translation, and 3-D TCP offset parameters.
- [ ] Validate finite arrays, matching counts, at least six points, and non-degenerate spatial spread.
- [ ] Run the synthetic test and the full package tests.

### Task 3: Add calibration CLI and safe output replacement

**Files:**
- Create: `custom_scripts/vision_pick_place/so101_graspnet_pick_and_place/calibrate_joint_tcp.py`
- Test: `custom_scripts/vision_pick_place/so101_graspnet_pick_and_place/test_calibrate_joint_tcp.py`

**Interfaces:**
- CLI arguments: `camera_points`, `joint_points`, `--port` or a local FK configuration, `--output`, `--max-rms-m`.
- Output JSON includes `camera_to_base`, `tcp_offset_m`, `rms_error_m`, and `calibration_type: joint_tcp`.

- [ ] Add failing CLI validation tests for missing/shape-mismatched arrays and RMS rejection.
- [ ] Implement FK construction using the existing SO-101 URDF/config and run the solver.
- [ ] Write output atomically via a temporary sibling file and replace only after RMS passes.
- [ ] Run CLI tests and package tests.

### Task 4: Recollect and verify on hardware

**Files:**
- No source changes.
- Runtime artifacts: `/tmp/handeye_left/{camera_points,joint_points,base_points}.npy` and `hand_eye.json`.

- [ ] Run collector with fixed gripper orientation and 8–12 visible points.
- [ ] Confirm all saved arrays have matching row counts and finite values.
- [ ] Run joint TCP calibration with `--max-rms-m 0.005`.
- [ ] Verify output RMS and inspect the saved TCP offset before enabling motion.
