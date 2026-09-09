from __future__ import annotations

import numpy as np
import pytest

from .local_vlm import OllamaVlmClient, parse_detections, parse_korean_task_json
from .models import DetectedObject


KOREAN_DETECTIONS = [
    DetectedObject("파란색 펜", (1, 2, 30, 40), 0.9, "object"),
    DetectedObject("검은색 박스", (60, 20, 120, 100), 0.9, "destination"),
]


def test_parse_detections_accepts_pixel_boxes_and_clamps_to_image() -> None:
    result = parse_detections(
        '{"objects":[{"label":"red block","box_xyxy":[-2,10,40,99],'
        '"confidence":0.91,"kind":"object"}]}',
        image_size=(80, 60),
    )

    assert result == [DetectedObject("red block", (0, 10, 40, 60), 0.91, "object")]


def test_parse_detections_rejects_missing_or_invalid_objects() -> None:
    assert parse_detections('{"objects":[]}', image_size=(80, 60)) == []

    with pytest.raises(ValueError, match="objects"):
        parse_detections('{}', image_size=(80, 60))

    with pytest.raises(ValueError, match="box_xyxy"):
        parse_detections(
            '{"objects":[{"label":"block","box_xyxy":[1,2],"confidence":0.9,"kind":"object"}]}',
            image_size=(80, 60),
        )


def test_ollama_client_sends_image_and_returns_json() -> None:
    calls: list[dict] = []

    def fake_post(url: str, *, json: dict, timeout: float) -> object:
        calls.append({"url": url, "json": json, "timeout": timeout})

        class Response:
            def raise_for_status(self) -> None:
                pass

            def json(self) -> dict:
                content = (
                    '{"objects":[{"label":"cup","box_xyxy":[1,2,8,9],'
                    '"confidence":0.8,"kind":"object"}]}'
                )
                return {"message": {"content": content}}

        return Response()

    client = OllamaVlmClient(post=fake_post)
    rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    objects = client.detect_objects(rgb, "detect all objects")

    assert objects[0].label == "cup"
    assert calls[0]["url"].endswith("/api/chat")
    assert calls[0]["json"]["format"]["type"] == "object"
    assert calls[0]["json"]["messages"][0]["images"]


def test_ollama_client_retries_once_after_malformed_json() -> None:
    responses = [
        '{"objects":[{"label":"broken"',
        '{"objects":[{"label":"cup","box_xyxy":[1,2,8,9],"confidence":0.8,"kind":"object"}]}',
    ]

    def fake_post(url: str, *, json: dict, timeout: float) -> object:
        class Response:
            def raise_for_status(self) -> None:
                pass

            def json(self) -> dict:
                return {"message": {"content": responses.pop(0)}}

        return Response()

    client = OllamaVlmClient(post=fake_post)
    objects = client.detect_objects(np.zeros((10, 10, 3), dtype=np.uint8), "detect objects")

    assert objects[0].label == "cup"
    assert not responses


def test_parse_korean_task_resolves_source_and_destination() -> None:
    command = parse_korean_task_json(
        '{"source_query":"파란색 펜","destination_query":"검은색 박스"}', KOREAN_DETECTIONS
    )

    assert command.source.label == "파란색 펜"
    assert command.destination.label == "검은색 박스"
    assert command.arm == "left"


def test_parse_korean_task_rejects_ambiguous_source() -> None:
    detections = [
        DetectedObject("파란색 펜", (1, 2, 30, 40), 0.9, "object"),
        DetectedObject("빨간색 펜", (40, 2, 70, 40), 0.8, "object"),
        KOREAN_DETECTIONS[1],
    ]

    with pytest.raises(ValueError, match="exactly one"):
        parse_korean_task_json(
            '{"source_query":"펜","destination_query":"검은색 박스"}', detections
        )
