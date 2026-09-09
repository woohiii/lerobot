from __future__ import annotations

import pytest

from custom_scripts.vision_pick_place.so101_graspnet_pick_and_place.routing import (
    ArmRoute,
    WorkspaceBounds,
    route_point,
)


def _arm(name: str, x_min: float, x_max: float) -> ArmRoute:
    return ArmRoute(
        name=name,
        workspace=WorkspaceBounds(x_min, x_max, -1.0, 1.0, 0.0, 2.0),
        bin_pose=(0.0, 0.0, 0.0),
    )


def test_routes_point_to_the_only_containing_arm() -> None:
    arms = [_arm("left", -2.0, -0.1), _arm("right", 0.1, 2.0)]

    assert route_point((-1.0, 0.0, 1.0), arms, 0.1).name == "left"
    assert route_point((1.0, 0.0, 1.0), arms, 0.1).name == "right"


def test_rejects_point_in_central_exclusion_zone() -> None:
    arms = [_arm("left", -2.0, 0.0), _arm("right", 0.0, 2.0)]

    with pytest.raises(ValueError, match="central exclusion"):
        route_point((0.05, 0.0, 1.0), arms, 0.1)


def test_rejects_point_outside_all_workspaces() -> None:
    arms = [_arm("left", -2.0, -0.1), _arm("right", 0.1, 2.0)]

    with pytest.raises(ValueError, match="workspace"):
        route_point((3.0, 0.0, 1.0), arms, 0.1)


def test_rejects_ambiguous_overlapping_workspaces() -> None:
    arms = [_arm("left", -2.0, 1.0), _arm("right", 0.0, 2.0)]

    with pytest.raises(ValueError, match="ambiguous"):
        route_point((0.5, 0.0, 1.0), arms, 0.1)
