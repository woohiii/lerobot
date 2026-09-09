"""Lazy GraspNet-baseline inference adapter for an already-installed checkout."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from .graspnet_adapter import GraspCandidate, GraspNetUnavailable, validate_graspnet_runtime


class GraspNetRuntime:
    """Load GraspNet on first use and infer poses from an XYZRGB ROI cloud."""

    def __init__(self, root: str | Path, checkpoint: str | Path, *, num_points: int = 20_000) -> None:
        self.root = Path(root)
        self.checkpoint = Path(checkpoint)
        self.num_points = num_points
        self._net: object | None = None

    def infer(self, point_cloud: np.ndarray) -> list[GraspCandidate]:
        """Return GraspNet poses; caller performs workspace and IK filtering."""
        validate_graspnet_runtime(self.root, self.checkpoint)
        if point_cloud.ndim != 2 or point_cloud.shape[1] < 3 or len(point_cloud) == 0:
            raise ValueError("point_cloud must be a non-empty Nx3 or Nx6 array")
        net, torch, pred_decode, grasp_group = self._load()
        xyz = np.asarray(point_cloud[:, :3], dtype=np.float32)
        indices = np.random.default_rng().choice(
            len(xyz), self.num_points, replace=len(xyz) < self.num_points
        )
        tensor = torch.from_numpy(xyz[indices][None]).cuda()
        with torch.no_grad():
            predictions = pred_decode(net({"point_clouds": tensor}))[0].detach().cpu().numpy()
        group = grasp_group(predictions)
        group.nms()
        group.sort_by_score()
        return [
            GraspCandidate(row[13:16], row[4:13].reshape(3, 3), float(row[0]))
            for row in group.grasp_group_array
        ]

    def _load(self):
        if self._net is not None:
            return self._net
        for directory in (
            self.root / "models",
            self.root / "dataset",
            self.root / "utils",
            self.root / "pointnet2",
            self.root / "knn",
        ):
            if str(directory) not in sys.path:
                sys.path.insert(0, str(directory))
        try:
            import torch
            from graspnet import GraspNet, pred_decode
            from graspnetAPI import GraspGroup
        except ImportError as exc:
            raise GraspNetUnavailable("GraspNet-baseline dependencies are not importable from root") from exc
        net = GraspNet(
            input_feature_dim=0, num_view=300, num_angle=12, num_depth=4,
            cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01, 0.02, 0.03, 0.04], is_training=False,
        ).cuda()
        net.load_state_dict(torch.load(self.checkpoint, map_location="cuda:0")["model_state_dict"])
        net.eval()
        self._net = (net, torch, pred_decode, GraspGroup)
        return self._net
