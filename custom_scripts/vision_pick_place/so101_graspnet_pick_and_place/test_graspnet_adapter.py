"""Tests for the dependency-light GraspNet adapter boundary."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from custom_scripts.vision_pick_place.so101_graspnet_pick_and_place.graspnet_adapter import (
    GraspCandidate,
    GraspNetUnavailable,
    filter_candidates,
    validate_graspnet_runtime,
)


def _candidate(score: float, *, position=(0.1, 0.2, 0.3), rotation=None) -> GraspCandidate:
    if rotation is None:
        rotation = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    return GraspCandidate(position_xyz=position, rotation_matrix=rotation, score=score)


def test_validate_runtime_accepts_existing_root_checkpoint_and_cuda(tmp_path: Path) -> None:
    """Runtime validation succeeds when all injected prerequisites are present."""
    root = tmp_path / "graspnet"
    root.mkdir()
    checkpoint = root / "checkpoint.tar"
    checkpoint.touch()

    validate_graspnet_runtime(root, checkpoint, cuda_available=lambda: True)


@pytest.mark.parametrize(
    ("root_exists", "checkpoint_exists", "cuda", "message"),
    [
        (False, False, True, "root"),
        (True, False, True, "checkpoint"),
        (True, True, False, "CUDA"),
    ],
)
def test_validate_runtime_rejects_missing_prerequisites(
    tmp_path: Path, root_exists: bool, checkpoint_exists: bool, cuda: bool, message: str
) -> None:
    """Runtime validation identifies each unavailable prerequisite."""
    root = tmp_path / "graspnet"
    if root_exists:
        root.mkdir()
    checkpoint = root / "checkpoint.tar"
    if checkpoint_exists:
        checkpoint.touch()

    with pytest.raises(GraspNetUnavailable, match=message):
        validate_graspnet_runtime(root, checkpoint, cuda_available=lambda: cuda)


def test_validate_runtime_can_skip_cuda_requirement(tmp_path: Path) -> None:
    """CPU dry-run validation does not call CUDA availability as a requirement."""
    root = tmp_path / "graspnet"
    root.mkdir()
    checkpoint = root / "checkpoint.tar"
    checkpoint.touch()

    validate_graspnet_runtime(root, checkpoint, require_cuda=False, cuda_available=lambda: False)


def test_filter_candidates_discards_malformed_nonfinite_and_low_score_candidates() -> None:
    """Malformed, non-finite, and insufficient-score candidates are excluded."""
    valid = _candidate(0.8)
    malformed_position = _candidate(0.9, position=(0.1, 0.2))
    malformed_rotation = _candidate(0.9, rotation=((1.0, 0.0),) * 3)
    nonfinite = _candidate(float("nan"))
    low_score = _candidate(0.2)

    assert filter_candidates(
        [valid, malformed_position, malformed_rotation, nonfinite, low_score], min_score=0.5
    ) == [valid]


def test_filter_candidates_returns_valid_candidates_in_descending_score_order() -> None:
    """Valid candidates retain score priority for the execution planner."""
    low = _candidate(0.6)
    high = _candidate(0.95)
    middle = _candidate(0.75)

    assert filter_candidates([low, high, middle], min_score=0.5) == [high, middle, low]


def test_filter_candidates_accepts_numpy_pose_arrays() -> None:
    """GraspNet-native numpy pose fields are preserved."""
    candidate = _candidate(0.9, position=np.array([0.1, 0.2, 0.3]), rotation=np.eye(3))

    assert filter_candidates([candidate]) == [candidate]
