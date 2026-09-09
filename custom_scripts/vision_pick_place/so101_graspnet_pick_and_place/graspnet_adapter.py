"""Dependency-light boundary for invoking GraspNet from the pick-and-place pipeline.

Model imports and checkpoint loading deliberately belong to the caller's runtime
adapter.  Keeping this module free of GraspNet imports makes startup validation
and candidate filtering available in dry-run and test environments.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

import numpy as np


class GraspNetUnavailable(RuntimeError):  # noqa: N818 - public contract name
    """Raised when the external GraspNet runtime cannot be used safely."""


@dataclass(frozen=True)
class GraspCandidate:
    """A scored 6-DOF grasp represented in camera coordinates."""

    position_xyz: Sequence[float]
    rotation_matrix: Sequence[Sequence[float]]
    score: float


def validate_graspnet_runtime(
    root: str | Path,
    checkpoint: str | Path,
    *,
    require_cuda: bool = True,
    cuda_available: Callable[[], bool] | None = None,
) -> None:
    """Check local GraspNet assets and, by default, CUDA availability.

    This only validates prerequisites; it never imports GraspNet or loads a model.
    """
    root_path = Path(root)
    checkpoint_path = Path(checkpoint)
    if not root_path.is_dir():
        raise GraspNetUnavailable(f"GraspNet root is not a directory: {root_path}")
    if not checkpoint_path.is_file():
        raise GraspNetUnavailable(f"GraspNet checkpoint is not a file: {checkpoint_path}")
    if require_cuda:
        is_cuda_available = cuda_available or _torch_cuda_available
        if not is_cuda_available():
            raise GraspNetUnavailable("CUDA is required for GraspNet inference but is unavailable")


def filter_candidates(
    candidates: Iterable[GraspCandidate], *, min_score: float = 0.0
) -> list[GraspCandidate]:
    """Return valid candidates at or above ``min_score``, best score first."""
    if not _is_finite_number(min_score):
        raise ValueError("min_score must be finite")

    valid = [
        candidate
        for candidate in candidates
        if _is_valid_candidate(candidate) and candidate.score >= min_score
    ]
    return sorted(valid, key=lambda candidate: candidate.score, reverse=True)


def _torch_cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def _is_valid_candidate(candidate: object) -> bool:
    if not isinstance(candidate, GraspCandidate):
        return False
    return (
        _has_finite_shape(candidate.position_xyz, (3,))
        and _has_finite_shape(candidate.rotation_matrix, (3, 3))
        and _is_finite_number(candidate.score)
    )


def _has_finite_shape(values: object, shape: tuple[int, ...]) -> bool:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return False
    return array.shape == shape and bool(np.isfinite(array).all())


def _is_finite_number(value: object) -> bool:
    try:
        return isfinite(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
