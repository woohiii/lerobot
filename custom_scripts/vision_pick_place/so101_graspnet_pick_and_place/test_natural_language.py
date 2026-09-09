from __future__ import annotations

import pytest

from .local_vlm import parse_command
from .models import DetectedObject


def test_parse_command_selects_source_and_destination_labels() -> None:
    detections = [
        DetectedObject("red block", (10, 10, 30, 30), 0.9, "object"),
        DetectedObject("blue tray", (40, 40, 90, 90), 0.95, "destination"),
    ]

    command = parse_command(
        '{"source_label":"red block","destination_label":"blue tray","arm":null}',
        detections,
    )

    assert command.source.label == "red block"
    assert command.destination.label == "blue tray"
    assert command.arm is None


def test_parse_command_rejects_unknown_label() -> None:
    detections = [DetectedObject("cup", (1, 1, 5, 5), 0.8, "object")]

    with pytest.raises(ValueError, match="destination"):
        parse_command(
            '{"source_label":"cup","destination_label":"missing","arm":null}',
            detections,
        )
