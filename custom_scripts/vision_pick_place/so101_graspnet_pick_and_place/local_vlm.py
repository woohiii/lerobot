"""Local Ollama vision-language adapter with fail-closed JSON contracts."""

from __future__ import annotations

import base64
import io
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from PIL import Image

from .models import DetectedObject, KoreanTaskQuery, TaskCommand


class KoreanTaskVlm(Protocol):
    """The two-call VLM contract consumed by the Korean pick/place CLI."""

    def detect_objects(self, rgb: np.ndarray, instruction: str) -> list[DetectedObject]:
        """Detect scene objects from an RGB frame."""
        ...

    def parse_korean_task(self, command: str, detections: Sequence[DetectedObject]) -> TaskCommand:
        """Resolve a Korean command against the visible detections."""
        ...


def _decode_json(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("Ollama returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("Ollama JSON response must be an object")
    return value


def _clamp_box(box: Sequence[Any], width: int, height: int) -> tuple[int, int, int, int]:
    if len(box) != 4:
        raise ValueError("box_xyxy must contain four coordinates")
    values = tuple(int(round(float(value))) for value in box)
    x0, y0, x1, y1 = values
    result = (max(0, x0), max(0, y0), min(width, x1), min(height, y1))
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError("box_xyxy must describe a visible non-empty box")
    return result


def parse_detections(text: str, *, image_size: tuple[int, int]) -> list[DetectedObject]:
    """Parse absolute pixel ``[x0,y0,x1,y1]`` detections from model JSON."""
    payload = _decode_json(text)
    objects = payload.get("objects")
    if not isinstance(objects, list):
        raise ValueError("Ollama response must contain an objects list")
    width, height = image_size
    result: list[DetectedObject] = []
    for item in objects:
        if not isinstance(item, dict):
            raise ValueError("each detected object must be an object")
        label = item.get("label")
        confidence = item.get("confidence")
        kind = item.get("kind", "object")
        if not isinstance(label, str) or not label.strip():
            raise ValueError("each detected object needs a label")
        if not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1:
            raise ValueError("confidence must be between 0 and 1")
        if kind not in {"object", "destination"}:
            raise ValueError("kind must be object or destination")
        if "box_xyxy" not in item:
            raise ValueError("each detected object needs box_xyxy")
        result.append(
            DetectedObject(
                label=label.strip(),
                box_xyxy=_clamp_box(item["box_xyxy"], width, height),
                confidence=float(confidence),
                kind=kind,
            )
        )
    return result


def _match_detection(label: str, detections: Sequence[DetectedObject], kind: str) -> DetectedObject:
    normalized = label.strip().casefold()
    candidates = [item for item in detections if item.kind == kind]
    exact = [item for item in candidates if item.label.casefold() == normalized]
    if len(exact) == 1:
        return exact[0]
    partial = [
        item
        for item in candidates
        if normalized in item.label.casefold() or item.label.casefold() in normalized
    ]
    if len(partial) == 1:
        return partial[0]
    raise ValueError(f"{kind} label {label!r} does not identify exactly one detection")


def parse_command(text: str, detections: Sequence[DetectedObject]) -> TaskCommand:
    """Resolve a model-produced command JSON against the current detections."""
    payload = _decode_json(text)
    source_label = payload.get("source_label")
    destination_label = payload.get("destination_label")
    arm = payload.get("arm")
    if not isinstance(source_label, str) or not isinstance(destination_label, str):
        raise ValueError("command JSON needs source_label and destination_label")
    if arm is not None and not isinstance(arm, str):
        raise ValueError("arm must be a string or null")
    return TaskCommand(
        source=_match_detection(source_label, detections, "object"),
        destination=_match_detection(destination_label, detections, "destination"),
        arm=arm,
    )


def parse_korean_task_json(text: str, detections: Sequence[DetectedObject]) -> TaskCommand:
    """Resolve Korean source and destination phrases against visible detections."""
    payload = _decode_json(text)
    source = payload.get("source_query")
    destination = payload.get("destination_query")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("Korean command JSON needs source_query")
    if not isinstance(destination, str) or not destination.strip():
        raise ValueError("Korean command JSON needs destination_query")
    query = KoreanTaskQuery(source_query=source, destination_query=destination)
    return TaskCommand(
        source=_match_detection(query.source_query, detections, "object"),
        destination=_match_detection(query.destination_query, detections, "destination"),
        arm="left",
    )


def _image_base64(rgb: np.ndarray) -> str:
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


@dataclass
class OllamaVlmClient:
    """Small synchronous client for Ollama's local ``/api/chat`` endpoint."""

    endpoint: str = "http://127.0.0.1:11434"
    model: str = "qwen2.5vl:7b"
    timeout_s: float = 120.0
    post: Callable[..., Any] | None = None

    def _request(
        self, messages: list[dict[str, Any]], *, response_format: dict[str, Any] | str = "json"
    ) -> str:
        post = self.post
        if post is None:
            import requests

            post = requests.post
        response = post(
            self.endpoint.rstrip("/") + "/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "format": response_format,
                "options": {"temperature": 0, "num_predict": 512},
                "messages": messages,
            },
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        payload = response.json()
        try:
            content = payload["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise ValueError("Ollama response has no message.content") from exc
        if not isinstance(content, str):
            raise ValueError("Ollama message.content must be text")
        return content

    def detect_objects(self, rgb: np.ndarray, instruction: str) -> list[DetectedObject]:
        height, width = rgb.shape[:2]
        content = (
            instruction
            + " Return only JSON: {\"objects\":[{\"label\":string,\"box_xyxy\":[x0,y0,x1,y1],"
            + "\"confidence\":number,\"kind\":\"object\"|\"destination\"}]} ."
        )
        messages = [{"role": "user", "content": content, "images": [_image_base64(rgb)]}]
        text = self._request(messages, response_format=_DETECTION_SCHEMA)
        try:
            return parse_detections(text, image_size=(width, height))
        except ValueError as first_error:
            messages[0]["content"] += (
                " Previous output was invalid. Return at most 20 detections and no markdown, "
                "commentary, or extra keys."
            )
            try:
                retry_text = self._request(messages, response_format=_DETECTION_SCHEMA)
                return parse_detections(retry_text, image_size=(width, height))
            except ValueError:
                raise first_error

    def parse_task(self, command: str, detections: Sequence[DetectedObject]) -> TaskCommand:
        labels = [{"label": item.label, "kind": item.kind} for item in detections]
        content = (
            "Resolve this command against these visible detections: "
            + json.dumps(labels, ensure_ascii=False)
            + f"\nCommand: {command}\nReturn only JSON with source_label, destination_label, arm."
        )
        return parse_command(
            self._request([{"role": "user", "content": content}], response_format=_LEGACY_COMMAND_SCHEMA),
            detections,
        )

    def parse_korean_task(self, command: str, detections: Sequence[DetectedObject]) -> TaskCommand:
        labels = [{"label": item.label, "kind": item.kind} for item in detections]
        content = (
            "Visible detections only: "
            + json.dumps(labels, ensure_ascii=False)
            + "\nKorean command: "
            + command
            + "\nReturn only JSON with exactly these keys: source_query, destination_query."
        )
        return parse_korean_task_json(
            self._request([{"role": "user", "content": content}], response_format=_COMMAND_SCHEMA),
            detections,
        )


_DETECTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["objects"],
    "properties": {
        "objects": {
            "type": "array",
            "maxItems": 20,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["label", "box_xyxy", "confidence", "kind"],
                "properties": {
                    "label": {"type": "string"},
                    "box_xyxy": {"type": "array", "minItems": 4, "maxItems": 4, "items": {"type": "number"}},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "kind": {"type": "string", "enum": ["object", "destination"]},
                },
            },
        }
    },
}

_LEGACY_COMMAND_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["source_label", "destination_label", "arm"],
    "properties": {
        "source_label": {"type": "string"},
        "destination_label": {"type": "string"},
        "arm": {"type": ["string", "null"]},
    },
}

_COMMAND_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["source_query", "destination_query"],
    "properties": {
        "source_query": {"type": "string"},
        "destination_query": {"type": "string"},
    },
}
