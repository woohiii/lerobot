# Korean Command Pick-and-Place Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fail-closed left SO-101 MVP that accepts a Korean command describing a source object and destination, localizes both in Astra S RGB-D, and previews or executes one pick-and-place.

**Architecture:** Retain `OllamaVlmClient` as the local Korean VLM boundary and force typed JSON/unique detections. A new stability localizer turns three registered RGB-D observations into validated base-frame poses. A left-arm-only execution layer receives a preflighted trajectory and performs hover, vertical approach, grasp, lift, destination release, and physical retreat; the CLI is preview-first and requires a distinct confirmation for motion.

**Tech Stack:** Python 3.12, NumPy, OpenCV, Astra S/OpenNI, local Ollama VLM, existing SO-101 kinematics and Pytest.

**Spec:** `docs/superpowers/specs/2026-09-09-korean-command-pick-place-design.md`

## Global Constraints

- Never convert free Korean text directly into motor commands; parse it to typed source/destination labels and resolve each to exactly one visible detection.
- Use `/dev/ttyACM3`, `LeftSOArm101`, and calibration ID `so101_left_follower` for the current left follower; do not use the right-arm adapter.
- Require the current hand-eye JSON RMS to be at most 0.03 m before any preview or execution.
- Preview is the default and must issue no motor commands. Execution requires `--execute` plus a unique confirmation string after displaying the exact plan.
- Reject zero/misaligned depth, non-unique/low-confidence detections, unstable 3-frame localization, workspace/table violations, and any IK preflight failure.
- First hardware execution remains vertical top-grasp only and has one source, one destination, one arm, low speed, a real retreat, and fail-closed aborts.

---

### Task 1: Korean command and unique VLM resolution

**Files:**
- Modify: `custom_scripts/vision_pick_place/so101_graspnet_pick_and_place/local_vlm.py`
- Modify: `custom_scripts/vision_pick_place/so101_graspnet_pick_and_place/models.py`
- Modify: `custom_scripts/vision_pick_place/so101_graspnet_pick_and_place/test_local_vlm.py`

**Interfaces:**
- Produces `KoreanTaskQuery(source_query: str, destination_query: str)` in `models.py`.
- Produces `OllamaVlmClient.parse_korean_task(command: str, detections: Sequence[DetectedObject]) -> TaskCommand`.
- Consumes existing `DetectedObject`, `TaskCommand`, `_match_detection`, and `_COMMAND_SCHEMA`.

- [ ] **Step 1: Write failing parsing tests**

```python
def test_parse_korean_task_resolves_source_and_destination() -> None:
    detections = [
        DetectedObject("파란색 펜", (1, 2, 30, 40), 0.9, "object"),
        DetectedObject("검은색 박스", (60, 20, 120, 100), 0.9, "destination"),
    ]
    command = parse_korean_task_json(
        '{"source_query":"파란색 펜","destination_query":"검은색 박스"}', detections
    )
    assert command.source.label == "파란색 펜"
    assert command.destination.label == "검은색 박스"

def test_parse_korean_task_rejects_ambiguous_source() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        parse_korean_task_json('{"source_query":"펜","destination_query":"검은색 박스"}', detections)
```

- [ ] **Step 2: Run tests and confirm they fail because `parse_korean_task_json` is missing**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q custom_scripts/vision_pick_place/so101_graspnet_pick_and_place/test_local_vlm.py -k korean`

Expected: failure naming the missing parser.

- [ ] **Step 3: Implement the typed Korean command parser and client prompt**

```python
def parse_korean_task_json(text: str, detections: Sequence[DetectedObject]) -> TaskCommand:
    payload = _decode_json(text)
    source = payload.get("source_query")
    destination = payload.get("destination_query")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("Korean command JSON needs source_query")
    if not isinstance(destination, str) or not destination.strip():
        raise ValueError("Korean command JSON needs destination_query")
    return TaskCommand(
        source=_match_detection(source, detections, "object"),
        destination=_match_detection(destination, detections, "destination"),
        arm="left",
    )
```

`parse_korean_task` must show the model only visible labels/kinds and demand JSON with those two exact keys. It must never select by confidence rank when a phrase matches more than one detection.

- [ ] **Step 4: Run focused tests**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q custom_scripts/vision_pick_place/so101_graspnet_pick_and_place/test_local_vlm.py -k korean`

Expected: PASS.

- [ ] **Step 5: Commit this independently testable parser change**

```bash
git add custom_scripts/vision_pick_place/so101_graspnet_pick_and_place/{local_vlm.py,models.py,test_local_vlm.py}
git commit -m "feat: parse Korean pick-place commands"
```

### Task 2: Three-frame registered RGB-D localization

**Files:**
- Create: `custom_scripts/vision_pick_place/so101_graspnet_pick_and_place/stable_localization.py`
- Create: `custom_scripts/vision_pick_place/so101_graspnet_pick_and_place/test_stable_localization.py`
- Modify: `custom_scripts/vision_pick_place/so101_graspnet_pick_and_place/local_planner.py`

**Interfaces:**
- Produces `StablePose(camera_xyz: np.ndarray, base_xyz: np.ndarray, max_spread_m: float)`.
- Produces `localize_stable_roi(frames: Sequence[RgbdFrame], roi: ObjectRoi, camera_to_base: np.ndarray, *, min_valid_depth_ratio: float, max_spread_m: float) -> StablePose`.
- `plan_local_task` consumes two `StablePose` values rather than assuming one ROI median is safe.

- [ ] **Step 1: Write failing stability tests**

```python
def test_localize_stable_roi_transforms_three_consistent_depth_frames() -> None:
    pose = localize_stable_roi(frames, roi, np.eye(4), min_valid_depth_ratio=0.5, max_spread_m=0.005)
    assert pose.camera_xyz == pytest.approx((0.02, -0.01, 0.60), abs=0.002)
    assert pose.max_spread_m < 0.005

def test_localize_stable_roi_rejects_depth_jitter() -> None:
    with pytest.raises(ValueError, match="unstable"):
        localize_stable_roi(jittered_frames, roi, np.eye(4), min_valid_depth_ratio=0.5, max_spread_m=0.005)
```

- [ ] **Step 2: Run tests and confirm they fail because the module is missing**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q custom_scripts/vision_pick_place/so101_graspnet_pick_and_place/test_stable_localization.py`

Expected: collection error for missing `stable_localization`.

- [ ] **Step 3: Implement median-per-frame localization and max Euclidean spread validation**

```python
points = np.stack([_roi_center_point(frame, roi, min_valid_ratio=min_valid_depth_ratio) for frame in frames])
spread = float(np.max(np.linalg.norm(points - np.median(points, axis=0), axis=1)))
if spread > max_spread_m:
    raise ValueError(f"ROI localization is unstable: {spread:.4f} m")
```

Require exactly three RGB-D frames with matching dimensions/intrinsics and refuse unverified registration before this function is called.

- [ ] **Step 4: Run focused tests**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q custom_scripts/vision_pick_place/so101_graspnet_pick_and_place/test_stable_localization.py custom_scripts/vision_pick_place/so101_graspnet_pick_and_place/test_local_planner.py`

Expected: PASS.

- [ ] **Step 5: Commit the localizer change**

```bash
git add custom_scripts/vision_pick_place/so101_graspnet_pick_and_place/{stable_localization.py,test_stable_localization.py,local_planner.py}
git commit -m "feat: require stable RGB-D localization"
```

### Task 3: Left-arm preflight and real safe retreat

**Files:**
- Create: `custom_scripts/vision_pick_place/so101_graspnet_pick_and_place/left_pick_executor.py`
- Create: `custom_scripts/vision_pick_place/so101_graspnet_pick_and_place/test_left_pick_executor.py`
- Modify: `custom_scripts/vision_pick_place/so101_graspnet_pick_and_place/adapters.py`

**Interfaces:**
- Produces `PickPlaceWaypoints(source_hover, source_grasp, source_lift, destination_hover, destination_release, retreat)`.
- Produces `build_and_validate_waypoints(arm: LeftSOArm101, source_xyz: np.ndarray, destination_xyz: np.ndarray, *, table_z: float, hover_height_m: float, lift_height_m: float, max_joint_delta_deg: float) -> PickPlaceWaypoints`.
- `SO101ArmAdapter.retreat()` must command the already validated safe retreat waypoint rather than return `None`.

- [ ] **Step 1: Write failing unit tests for waypoint geometry and preflight rejection**

```python
def test_waypoints_use_vertical_source_approach_and_destination_release() -> None:
    waypoints = build_waypoints(np.array([0.20, 0.0, 0.01]), np.array([0.14, -0.1, 0.02]), hover_height_m=0.03, lift_height_m=0.05)
    assert waypoints.source_hover[2] == pytest.approx(0.04)
    assert waypoints.destination_release[2] == pytest.approx(0.02)

def test_preflight_rejects_an_ik_step_above_joint_limit() -> None:
    with pytest.raises(ValueError, match="joint"):
        build_and_validate_waypoints(fake_arm, source, destination, table_z=-0.04, hover_height_m=0.03, lift_height_m=0.05, max_joint_delta_deg=10.0)
```

- [ ] **Step 2: Run tests and confirm they fail because the executor module is missing**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q custom_scripts/vision_pick_place/so101_graspnet_pick_and_place/test_left_pick_executor.py`

Expected: collection error for missing `left_pick_executor`.

- [ ] **Step 3: Implement dry IK validation for every waypoint before any actuator call**

Use `LeftSOArm101.kin.solve_ik`/existing kinematics at every leg, require finite joint vectors, reject table penetration and per-joint delta above `max_joint_delta_deg`. Store the validated retreat pose and make adapter retreat send it. Never reuse the generic right-arm SO101 configuration.

- [ ] **Step 4: Run focused tests**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q custom_scripts/vision_pick_place/so101_graspnet_pick_and_place/test_left_pick_executor.py custom_scripts/vision_pick_place/so101_graspnet_pick_and_place/test_orchestrator.py`

Expected: PASS.

- [ ] **Step 5: Commit the preflight/retreat change**

```bash
git add custom_scripts/vision_pick_place/so101_graspnet_pick_and_place/{left_pick_executor.py,test_left_pick_executor.py,adapters.py}
git commit -m "feat: preflight left arm pick trajectory"
```

### Task 4: Korean command preview CLI

**Files:**
- Create: `custom_scripts/vision_pick_place/so101_graspnet_pick_and_place/korean_pick_place.py`
- Create: `custom_scripts/vision_pick_place/so101_graspnet_pick_and_place/test_korean_pick_place.py`
- Modify: `custom_scripts/vision_pick_place/so101_graspnet_pick_and_place/local_vlm.py`

**Interfaces:**
- Produces `main(argv: list[str] | None = None) -> int`.
- Consumes `--command`, `--calibration`, `--port`, `--preview`, `--execute`, and injected camera/VLM factories in tests.
- Prints one JSON-safe dry plan containing source/destination labels, camera/base XYZ, confidence, stability spread, and all waypoints.

- [ ] **Step 1: Write failing CLI tests**

```python
def test_preview_prints_plan_without_constructing_an_arm(capsys: pytest.CaptureFixture[str]) -> None:
    result = main(args, camera_factory=fake_camera, vlm_factory=fake_vlm, arm_factory=FailIfConstructed)
    assert result == 0
    assert "dry plan" in capsys.readouterr().out

def test_execute_requires_exact_confirmation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("builtins.input", lambda _: "NO")
    assert main(execute_args, camera_factory=fake_camera, vlm_factory=fake_vlm, arm_factory=fake_arm) == 2
```

- [ ] **Step 2: Run tests and confirm they fail because the CLI module is missing**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q custom_scripts/vision_pick_place/so101_graspnet_pick_and_place/test_korean_pick_place.py`

Expected: collection error for missing `korean_pick_place`.

- [ ] **Step 3: Implement preview-first CLI**

Read `/tmp/handeye_left_v7/hand_eye.json` only through its path argument, reject RMS above 0.03 m, collect three frames, call the VLM once for detections then once for Korean command resolution, localize both target ROIs, build preflighted waypoints, and draw source/destination bboxes plus coordinates in the RGB preview. `--preview` exits after display/printed plan with no `LeftSOArm101` construction.

- [ ] **Step 4: Run focused tests**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q custom_scripts/vision_pick_place/so101_graspnet_pick_and_place/test_korean_pick_place.py`

Expected: PASS.

- [ ] **Step 5: Commit the CLI preview change**

```bash
git add custom_scripts/vision_pick_place/so101_graspnet_pick_and_place/{korean_pick_place.py,test_korean_pick_place.py,local_vlm.py}
git commit -m "feat: preview Korean pick-place command"
```

### Task 5: One-shot confirmed hardware execution and gate validation

**Files:**
- Modify: `custom_scripts/vision_pick_place/so101_graspnet_pick_and_place/korean_pick_place.py`
- Modify: `custom_scripts/vision_pick_place/so101_graspnet_pick_and_place/left_pick_executor.py`
- Modify: `custom_scripts/vision_pick_place/so101_graspnet_pick_and_place/test_korean_pick_place.py`
- Modify: `custom_scripts/vision_pick_place/so101_graspnet_pick_and_place/test_left_pick_executor.py`

**Interfaces:**
- Consumes `--execute` and exact input `EXECUTE_PICK_PLACE` after preview output.
- Produces `execute_validated_pick_place(arm: LeftSOArm101, waypoints: PickPlaceWaypoints) -> None`.

- [ ] **Step 1: Write failing state-order and abort tests**

```python
def test_execute_uses_prevalidated_waypoint_order() -> None:
    execute_validated_pick_place(fake_arm, waypoints)
    assert fake_arm.events == ["source_hover", "open", "source_grasp", "close", "source_lift", "destination_hover", "destination_release", "open", "retreat"]

def test_execution_failure_retreats_before_reraising() -> None:
    fake_arm.raise_on = "source_grasp"
    with pytest.raises(SafetyAbortError):
        execute_validated_pick_place(fake_arm, waypoints)
    assert fake_arm.events[-1] == "retreat"
```

- [ ] **Step 2: Run tests and confirm they fail because the execution entry point is missing**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q custom_scripts/vision_pick_place/so101_graspnet_pick_and_place/test_left_pick_executor.py -k execution`

Expected: failure naming missing `execute_validated_pick_place`.

- [ ] **Step 3: Implement one-shot low-speed execution**

Connect only after confirmation. Enable torque at current pose, then send exactly the prevalidated waypoints at low speed. After close/lift, require finite joint feedback and a conservative grasp check; on any failure, command the stored safe retreat, preserve torque, and raise `SafetyAbortError`. Do not implement automatic retries.

- [ ] **Step 4: Run package gate**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q custom_scripts/vision_pick_place/so101_graspnet_pick_and_place custom_scripts/vision_pick_place/task_red_cube_to_bin_new_gripper/test_perception_qwen.py`

Expected: PASS with only the existing `hppfcl` deprecation warning.

- [ ] **Step 5: Hardware validation sequence (manual, not automated)**

```bash
cd ~/lerobot/custom_scripts/vision_pick_place
uv run python -m so101_graspnet_pick_and_place.korean_pick_place \
  --command "파란색 펜을 주워서 검은색 박스에다 담아줘" \
  --calibration /tmp/handeye_left_v7/hand_eye.json \
  --port /dev/ttyACM3 --preview
```

Accept only if both overlays, camera/base XYZ, stable depth, and all waypoints are visibly correct. Then repeat with `--execute`, type `EXECUTE_PICK_PLACE`, and keep an operator at the emergency stop.

- [ ] **Step 6: Commit the final execution gate**

```bash
git add custom_scripts/vision_pick_place/so101_graspnet_pick_and_place/{korean_pick_place.py,left_pick_executor.py,test_korean_pick_place.py,test_left_pick_executor.py}
git commit -m "feat: execute validated Korean pick-place"
```
