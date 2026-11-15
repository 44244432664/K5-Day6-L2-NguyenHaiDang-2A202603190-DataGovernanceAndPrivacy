"""Configuration and tabular contracts for T05.

This module deliberately contains no duplicate-detection algorithm. It defines
the stable identifiers, schemas, and parameter constraints that later T05
stages share. In particular, a duplicate relation is distinct from a broader
scene-only relation used to keep related images in the same train/test split.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any


# Internal schemas. Array-valued feature columns are stored as DataFrame object
# columns until a later task supplies an explicit serialization format.
IMAGE_FEATURE_COLUMNS = (
    "image_id",
    "path",
    "camera_id",
    "captured_at",
    "sha256",
    "width",
    "height",
    "phash",
    "dhash",
    "color_histogram",
    "thumbnail",
)

PAIR_EVIDENCE_COLUMNS = (
    "pair_id",
    "img_a",
    "img_b",
    "exact_match",
    "phash_distance",
    "dhash_distance",
    "histogram_distance",
    "thumbnail_distance",
    "same_camera",
    "time_delta_seconds",
    "duplicate_score",
    "review_priority",
    "reason_codes",
)
CLASSIFIED_PAIR_COLUMNS = PAIR_EVIDENCE_COLUMNS + ("classification", "pred_dup")
PAIR_CLASSIFICATIONS = frozenset({"positive", "negative", "ambiguous"})
REVIEWED_PAIR_COLUMNS = CLASSIFIED_PAIR_COLUMNS + (
    "review_decision",
    "review_relation",
    "reviewer",
    "review_note",
    "reviewed_at",
    "review_unresolved",
)

PAIR_CONSTRAINT_COLUMNS = ("img_a", "img_b", "relation")
DUPLICATE_GROUP_COLUMNS = ("image_id", "duplicate_group_id")
GROUP_PAIR_COLUMNS = ("img_a", "img_b", "duplicate_group_id")
GROUP_CONFLICT_COLUMNS = (
    "conflict_type",
    "img_a",
    "img_b",
    "relation",
    "duplicate_group_id",
    "detail",
)
GROUP_ASSIGNMENT_COLUMNS = ("image_id", "duplicate_group_id", "split_group_id")
SPLIT_GROUP_COLUMNS = GROUP_ASSIGNMENT_COLUMNS + (
    "split_group_size",
    "exceeds_max_component_size",
)
PREDICTION_COLUMNS = ("img_a", "img_b", "score", "pred_dup")
SPLIT_MANIFEST_COLUMNS = ("image_id", "split")
PAIR_REVIEW_COLUMNS = (
    "rank",
    "item_id",
    "score",
    "reason",
    "suggested_action",
    "slice",
    "evidence_path",
    "reviewer",
    "reviewer_decision",
    "override_value",
    "review_note",
    "reviewed_at",
)

DUPLICATE_RELATION = "duplicate"
SCENE_ONLY_RELATION = "same_scene_only"
DIFFERENT_SCENE_RELATION = "different_scene"
RELATION_OVERRIDE_VALUES = frozenset(
    {DUPLICATE_RELATION, SCENE_ONLY_RELATION, DIFFERENT_SCENE_RELATION}
)


# Initial development defaults. These values must be frozen before official
# evaluation; changing them after seeing evaluation labels would be leakage.
DEFAULT_PARAMS = MappingProxyType(
    {
        "test_fraction": 0.3,
        "budget": 20,
        "phash_size": 8,
        "phash_highfreq_factor": 4,
        "dhash_size": 8,
        "histogram_bins": 16,
        "thumbnail_width": 32,
        "thumbnail_height": 18,
        "phash_distance_max": 8,
        "dhash_distance_max": 8,
        "histogram_distance_max": 0.08,
        "thumbnail_distance_max": 0.10,
        "min_visual_signals": 2,
        "duplicate_score_threshold": 0.75,
        "review_score_low": 0.55,
        "review_score_high": 0.85,
        "burst_window_seconds": 60.0,
        "burst_max_span_seconds": 60.0,
        "max_component_size": 20,
        "decision_max_unresolved": 0,
        # Development protocol v2: bounded RGB registration, no external model.
        "verification_enabled": 1,
        "verification_width": 48,
        "verification_height": 32,
        "verification_crop_max": 0.12,
        "verification_crop_steps": 6,
        "verification_error_max": 0.05,
        "verification_min_texture": 0.02,
    }
)


def _image_id(value: Any, argument: str) -> str:
    """Normalize an image identifier to a nonblank string."""
    if value is None:
        raise ValueError(f"{argument} must be a nonblank image identifier, got None")
    text = str(value)
    if not text.strip():
        raise ValueError(f"{argument} must be a nonblank image identifier")
    return text


def canonical_pair(img_a: Any, img_b: Any) -> tuple[str, str]:
    """Return a stable unordered image pair as two lexicographically sorted strings.

    Image identifiers are converted to strings but are otherwise preserved. A
    self-pair is invalid because the T05 metric evaluates pairs of distinct
    images. Blank identifiers and ``None`` raise :class:`ValueError`.
    """
    a = _image_id(img_a, "img_a")
    b = _image_id(img_b, "img_b")
    if a == b:
        raise ValueError(f"self-pair is not allowed for image_id {a!r}")
    return (a, b) if a < b else (b, a)


def make_pair_id(img_a: Any, img_b: Any) -> str:
    """Return a collision-safe identifier for an unordered pair.

    Each component is length-framed, so identifiers containing punctuation or
    apparent separators cannot create ambiguous IDs. Character lengths are
    deterministic for Python strings and sufficient because IDs are serialized
    and parsed as text rather than byte offsets.
    """
    a, b = canonical_pair(img_a, img_b)
    return f"pair:{len(a)}:{a}{len(b)}:{b}"


def _integer(name: str, value: Any, minimum: int, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"params.{name} must be an integer, got {value!r}")
    result = int(value)
    if result < minimum or (maximum is not None and result > maximum):
        limit = f"between {minimum} and {maximum}" if maximum is not None else f">= {minimum}"
        raise ValueError(f"params.{name} must be {limit}, got {result}")
    return result


def _number(
    name: str,
    value: Any,
    minimum: float,
    maximum: float | None = None,
    *,
    minimum_inclusive: bool = True,
    maximum_inclusive: bool = True,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"params.{name} must be a finite number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"params.{name} must be finite, got {value!r}")
    below = result < minimum if minimum_inclusive else result <= minimum
    above = False
    if maximum is not None:
        above = result > maximum if maximum_inclusive else result >= maximum
    if below or above:
        left = "[" if minimum_inclusive else "("
        right = "]" if maximum_inclusive else ")"
        upper = "inf" if maximum is None else str(maximum)
        raise ValueError(f"params.{name} must be in {left}{minimum}, {upper}{right}, got {result}")
    return result


def validate_params(params: Mapping[str, Any] | None = None) -> dict[str, int | float]:
    """Validate and normalize the flat T05 parameter mapping.

    Missing values receive :data:`DEFAULT_PARAMS`; callers' mappings are never
    mutated. Integer controls remain integers and continuous thresholds become
    floats. Unknown keys are rejected to catch misspellings and prevent a run
    from silently ignoring a supposedly frozen parameter.

    Returns:
        A new normalized dictionary containing every supported T05 parameter.

    Raises:
        ValueError: if the mapping, a value, or a cross-parameter constraint is
            invalid.
    """
    if params is None:
        supplied: dict[str, Any] = {}
    elif isinstance(params, Mapping):
        supplied = dict(params)
    else:
        raise ValueError(f"T05 params must be a mapping, got {type(params).__name__}")

    unknown = sorted(set(supplied) - set(DEFAULT_PARAMS))
    if unknown:
        raise ValueError(f"unknown T05 params: {unknown}; supported={sorted(DEFAULT_PARAMS)}")

    raw = dict(DEFAULT_PARAMS)
    raw.update(supplied)
    out: dict[str, int | float] = {}

    out["test_fraction"] = _number(
        "test_fraction", raw["test_fraction"], 0.0, 1.0,
        minimum_inclusive=False, maximum_inclusive=False,
    )
    out["budget"] = _integer("budget", raw["budget"], 1)
    out["phash_size"] = _integer("phash_size", raw["phash_size"], 4, 32)
    out["phash_highfreq_factor"] = _integer(
        "phash_highfreq_factor", raw["phash_highfreq_factor"], 2, 8
    )
    out["dhash_size"] = _integer("dhash_size", raw["dhash_size"], 4, 32)
    out["histogram_bins"] = _integer("histogram_bins", raw["histogram_bins"], 2, 256)
    out["thumbnail_width"] = _integer("thumbnail_width", raw["thumbnail_width"], 1, 1024)
    out["thumbnail_height"] = _integer("thumbnail_height", raw["thumbnail_height"], 1, 1024)

    out["phash_distance_max"] = _integer(
        "phash_distance_max", raw["phash_distance_max"], 0, out["phash_size"] ** 2
    )
    out["dhash_distance_max"] = _integer(
        "dhash_distance_max", raw["dhash_distance_max"], 0, out["dhash_size"] ** 2
    )
    out["histogram_distance_max"] = _number(
        "histogram_distance_max", raw["histogram_distance_max"], 0.0, 2.0
    )
    out["thumbnail_distance_max"] = _number(
        "thumbnail_distance_max", raw["thumbnail_distance_max"], 0.0, 1.0
    )
    out["min_visual_signals"] = _integer(
        "min_visual_signals", raw["min_visual_signals"], 1, 3
    )
    out["duplicate_score_threshold"] = _number(
        "duplicate_score_threshold", raw["duplicate_score_threshold"], 0.0, 1.0,
        minimum_inclusive=False,
    )
    out["review_score_low"] = _number(
        "review_score_low", raw["review_score_low"], 0.0, 1.0
    )
    out["review_score_high"] = _number(
        "review_score_high", raw["review_score_high"], 0.0, 1.0
    )
    out["burst_window_seconds"] = _number(
        "burst_window_seconds", raw["burst_window_seconds"], 0.0,
        minimum_inclusive=False,
    )
    out["burst_max_span_seconds"] = _number(
        "burst_max_span_seconds", raw["burst_max_span_seconds"], 0.0,
        minimum_inclusive=False,
    )
    out["max_component_size"] = _integer(
        "max_component_size", raw["max_component_size"], 2
    )
    out["decision_max_unresolved"] = _integer(
        "decision_max_unresolved", raw["decision_max_unresolved"], 0
    )
    out["verification_enabled"] = _integer("verification_enabled", raw["verification_enabled"], 0, 1)
    for name in ("verification_width", "verification_height"):
        out[name] = _integer(name, raw[name], 8, 256)
    out["verification_crop_steps"] = _integer(
        "verification_crop_steps", raw["verification_crop_steps"], 1, 20
    )
    out["verification_crop_max"] = _number(
        "verification_crop_max", raw["verification_crop_max"], 0.0, 0.24
    )
    for name in ("verification_error_max", "verification_min_texture"):
        out[name] = _number(name, raw[name], 0.0, 1.0, minimum_inclusive=False)

    threshold = out["duplicate_score_threshold"]
    low, high = out["review_score_low"], out["review_score_high"]
    if low > threshold or threshold > high:
        raise ValueError(
            "params review band must contain duplicate_score_threshold: "
            f"review_score_low={low}, duplicate_score_threshold={threshold}, review_score_high={high}"
        )
    if out["burst_max_span_seconds"] < out["burst_window_seconds"]:
        raise ValueError(
            "params.burst_max_span_seconds must be >= params.burst_window_seconds "
            f"({out['burst_max_span_seconds']} < {out['burst_window_seconds']})"
        )

    return out
