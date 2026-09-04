"""Local, open-vocabulary object detection via Qwen2.5-VL-3B-Instruct - replaces
task_red_cube_to_bin/perception_zeroshot.py's Gemini cloud calls with a model
that runs on this machine's own GPU (RTX 5050, 8GB VRAM - Qwen2.5-VL-3B fits in
fp16 with room to spare). Picked over Llama-3.2-Vision because Qwen2.5-VL was
actually trained to emit bounding-box coordinates for a described object
(grounding); Llama-3.2-Vision's training is captioning/VQA-focused and has no
equivalent structured-box output.

Same Detection-shaped interface as perception_zeroshot.detect_zeroshot - a
drop-in detect_fn for any caller already built against that shape.

2026-09-01: started with Qwen2-VL-2B, real live-frame testing (a clean,
unannotated Astra frame with a black cube/small red cube/orange clip on the
table - the FIRST smoke test frame used before this had a debug HUD baked
into it by an earlier detector's own overlay, and Qwen2-VL's answers on that
one were suspiciously anchored to that baked-in text rather than the actual
object - caught only by checking a saved image directly, not trusted blindly)
showed real problems: (a) inconsistent coordinate convention call to call
(sometimes pixel-ish numbers in its own internally-resized input space,
sometimes plain 0-1 normalized fractions - see the small-value branch below),
and (b) only 1/3 objects (the large, high-contrast black cube) actually
localized correctly - the small red cube and the orange clip both came back
badly wrong. Switched to Qwen2.5-VL-3B-Instruct (same VRAM budget, same
prompt/parsing shape) and re-ran the identical 3-object test: all 3 landed
correctly this time, AND Qwen2.5-VL's bbox_2d came back already in the
ORIGINAL frame's pixel space (confirmed by comparing to each object's real
on-screen position directly) - no image_grid_thw-based rescale needed at all,
unlike Qwen2-VL. Kept the <=1.5 branch below as a defensive fallback in case
a future prompt/model ever answers in normalized fractions instead, but the
grid_thw rescale path Qwen2-VL needed is gone - verify again if this model is
ever swapped for a different one.
"""

from __future__ import annotations

import json
import re

import numpy as np
import torch

import config
from perception import Detection

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
BOX_THRESHOLD = 0.3  # same permissive default as perception_zeroshot - prompt-level filtering does most of the work
DEDUP_CENTER_PX = 40  # detect_all_qwen: two detections this close in center are treated as the same real object

_DETECT_PROMPT_TMPL = (
    "Detect {target} in this image. Ignore the robot arm/gripper hardware itself - "
    "only report a real object sitting on the table.\n"
    'Output a JSON object with "bbox_2d": [xmin, ymin, xmax, ymax], "label", and '
    '"confidence" (0.0-1.0). Return ONLY the JSON object, no markdown fences, no other text. '
    "If nothing matching is visible, output {{}}."
)

_model = None
_processor = None


def _lazy_load() -> None:
    global _model, _processor
    if _model is None:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="cuda")
        _processor = AutoProcessor.from_pretrained(MODEL_ID)


def _call_qwen(bgr_frame: np.ndarray, target_desc: str) -> tuple[float, tuple[int, int, int, int]] | None:
    """One Qwen2.5-VL detection call. Returns (confidence, (x, y, w, h)) in the
    ORIGINAL frame's pixel space, or None if nothing was returned/parseable."""
    _lazy_load()
    h, w = bgr_frame.shape[:2]
    rgb = np.ascontiguousarray(bgr_frame[:, :, ::-1])  # torch.from_numpy refuses the negative-stride ::-1 view directly

    prompt = _DETECT_PROMPT_TMPL.format(target=target_desc.strip().lower())
    messages = [{"role": "user", "content": [{"type": "image", "image": rgb}, {"type": "text", "text": prompt}]}]
    text = _processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _processor(text=[text], images=[rgb], return_tensors="pt").to(_model.device)

    with torch.no_grad():
        out_ids = _model.generate(**inputs, max_new_tokens=128, do_sample=False)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
    response = _processor.batch_decode(trimmed, skip_special_tokens=True)[0]

    m = re.search(r"\{.*\}", response, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        x0, y0, x1, y1 = obj["bbox_2d"]
        confidence = float(obj.get("confidence", 1.0))
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return None

    # Defensive fallback only - Qwen2.5-VL's bbox_2d was confirmed (real-frame
    # test, see module docstring) to already be in the ORIGINAL frame's pixel
    # space, unlike Qwen2-VL which needed an image_grid_thw-based rescale. An
    # all-small-values response (never seen from 2.5-VL in testing, but cheap
    # to guard) is treated as a 0-1 normalized fraction instead of silently
    # producing a nonsense tiny box.
    if max(x0, y0, x1, y1) <= 1.5:
        x0, y0, x1, y1 = x0 * w, y0 * h, x1 * w, y1 * h
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(w, int(x1)), min(h, int(y1))
    if x1 <= x0 or y1 <= y0:
        return None
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    ex0, ey0, ex1, ey1 = config.ROBOT_EXCLUSION_BBOX_PX
    if ex0 <= cx <= ex1 and ey0 <= cy <= ey1:
        # real false positive seen live (the perforated gripper housing
        # labeled "white box with holes") - see config.ROBOT_EXCLUSION_BBOX_PX
        return None
    return confidence, (x0, y0, x1 - x0, y1 - y0)


def detect_qwen(bgr_frame: np.ndarray, text_prompt: str, box_threshold: float = BOX_THRESHOLD) -> Detection | None:
    """Drop-in detect_fn, same Detection shape as perception.py's HSV
    detectors / perception_zeroshot.detect_zeroshot."""
    result = _call_qwen(bgr_frame, text_prompt)
    if result is None:
        return None
    confidence, (x, y, w, h) = result
    if confidence < box_threshold:
        return None
    return Detection(cx=x + w / 2.0, cy=y + h / 2.0, area=float(w * h), bbox=(x, y, w, h))


_LIST_NAMES_PROMPT = (
    "What distinct physical objects are sitting on the table in this image? Ignore the robot "
    "arm/gripper and cables. List only their short names, comma-separated, nothing else."
)


def _list_object_names(bgr_frame: np.ndarray) -> list[str]:
    """Free-text 'what's here' query - NOT bbox grounding. Exists only to
    seed candidate names for detect_all_qwen below; see that function's
    docstring for why a plain multi-object bbox-array prompt (asking for
    bbox_2d directly for every object in one call) was tried first and
    dropped."""
    _lazy_load()
    rgb = np.ascontiguousarray(bgr_frame[:, :, ::-1])
    messages = [{"role": "user", "content": [{"type": "image", "image": rgb}, {"type": "text", "text": _LIST_NAMES_PROMPT}]}]
    text = _processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _processor(text=[text], images=[rgb], return_tensors="pt").to(_model.device)
    with torch.no_grad():
        out_ids = _model.generate(**inputs, max_new_tokens=64, do_sample=False)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
    response = _processor.batch_decode(trimmed, skip_special_tokens=True)[0]
    names = [n.strip() for n in response.split(",") if n.strip()]
    return names


def detect_all_qwen(bgr_frame: np.ndarray, box_threshold: float = BOX_THRESHOLD) -> list[tuple[Detection, str]]:
    """The "show what's on the table" step of qwen_click_pick_place.py's
    flow (step 2: detect, step 3: user clicks one). Returns (Detection,
    label) pairs, largest-area first.

    2026-09-01: a single call asking Qwen to output a bbox_2d array for
    EVERY object at once was tried first and real-frame-tested against 3
    known objects (black cube/small red cube/orange clip) - it returned []
    (nothing) on the exact prompt from this module's single-object
    _DETECT_PROMPT_TMPL adapted to a list, and even a much simpler "detect
    all objects, output a JSON array" prompt only ever found 1-2 of the 3
    and with visibly wrong boxes. Multi-object enumeration is a
    qualitatively harder task for a 3B model than "find THIS described
    object" (already confirmed 3/3 reliable - see this module's earlier
    docstring), not a fixable prompt-wording bug - re-verified by trying a
    tight crop around each object's own known location and asking Qwen to
    detect "the object nearest the center of that crop": still 2/3 failed
    (returned nothing), the crop apparently removes context the model
    needs. Given both attempts at direct multi-object grounding failed on
    real data, this instead does the two-phase thing that DOES work: (1)
    _list_object_names asks a plain "what's here" free-text question (a
    much easier task - real-frame-tested, correctly named the black cube,
    though it still missed the small red cube entirely, i.e. recall here is
    NOT complete either), then (2) each named candidate is grounded
    SEPARATELY with detect_qwen against the FULL frame - reusing the one
    path already proven reliable, rather than trusting either of the direct
    multi-object attempts. Still an incomplete solution (an object this
    step's naming pass misses never gets a box) - qwen_click_pick_place.py
    keeps a manual-description fallback for exactly that gap, don't remove
    it thinking this function got fixed.

    Also real-frame-tested with a dedup issue: _list_object_names sometimes
    names the SAME physical object more than once with different wording
    ("Black object", "White object", "Black cube" all grounded to the
    identical bbox on one real frame) - deduped below by center-distance
    (DEDUP_CENTER_PX) rather than trusting the names to be distinct."""
    names = _list_object_names(bgr_frame)
    out: list[tuple[Detection, str]] = []
    for name in names:
        det = detect_qwen(bgr_frame, name, box_threshold=box_threshold)
        if det is None:
            continue
        if any(abs(det.cx - existing.cx) < DEDUP_CENTER_PX and abs(det.cy - existing.cy) < DEDUP_CENTER_PX
               for existing, _ in out):
            continue
        out.append((det, name))
    out.sort(key=lambda pair: -pair[0].area)
    return out
