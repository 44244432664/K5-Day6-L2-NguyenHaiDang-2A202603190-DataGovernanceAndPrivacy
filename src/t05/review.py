"""Deterministic pair-review ranking and offline evidence rendering for T05.

Task 9 only prepares review artifacts. It never applies decisions or mutates
automatic classifications, duplicate groups, or provisional split assignments.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..common.contact_sheet import render_contact_sheet
from ..common.io import sha256_text, write_json
from ..common.review import (
    DEFAULT_ALLOWED_DECISIONS,
    REVIEW_COLUMNS,
    AuditLog,
    make_review_queue,
)
from .contracts import (
    CLASSIFIED_PAIR_COLUMNS,
    DIFFERENT_SCENE_RELATION,
    DUPLICATE_RELATION,
    PAIR_CLASSIFICATIONS,
    PAIR_CONSTRAINT_COLUMNS,
    PAIR_REVIEW_COLUMNS,
    PREDICTION_COLUMNS,
    RELATION_OVERRIDE_VALUES,
    REVIEWED_PAIR_COLUMNS,
    SCENE_ONLY_RELATION,
    canonical_pair,
    make_pair_id,
    validate_params,
)
from .matching import build_duplicate_groups, expand_group_pairs
from .splitting import assign_group_splits, build_split_groups, summarize_split


_QUEUE_IDENTITY_COLUMNS = PAIR_REVIEW_COLUMNS[:7]
_REVIEW_CONTEXT_VERSION = 1


def _parse_pair_id(value: Any) -> tuple[str, str]:
    """Parse the length-framed ID produced by :func:`make_pair_id`."""
    text = str(value)
    prefix = "pair:"
    if not text.startswith(prefix):
        raise ValueError(f"invalid T05 pair ID {text!r}: missing {prefix!r} prefix")
    cursor = len(prefix)

    def read_length(position: int) -> tuple[int, int]:
        colon = text.find(":", position)
        if colon < 0 or colon == position or not text[position:colon].isdigit():
            raise ValueError(f"invalid T05 pair ID {text!r}: malformed length")
        return int(text[position:colon]), colon + 1

    length_a, cursor = read_length(cursor)
    img_a = text[cursor : cursor + length_a]
    if len(img_a) != length_a:
        raise ValueError(f"invalid T05 pair ID {text!r}: truncated first image ID")
    cursor += length_a
    length_b, cursor = read_length(cursor)
    img_b = text[cursor : cursor + length_b]
    if len(img_b) != length_b or cursor + length_b != len(text):
        raise ValueError(f"invalid T05 pair ID {text!r}: truncated or trailing content")
    pair = canonical_pair(img_a, img_b)
    if make_pair_id(*pair) != text:
        raise ValueError(f"invalid or noncanonical T05 pair ID {text!r}")
    return pair


def _validated_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(pairs, pd.DataFrame):
        raise ValueError(f"pairs must be a pandas DataFrame, got {type(pairs).__name__}")
    missing = [column for column in CLASSIFIED_PAIR_COLUMNS if column not in pairs.columns]
    if missing:
        raise ValueError(f"pairs is missing required columns {missing}; has {list(pairs.columns)}")
    out = pairs.copy()
    if out.pair_id.isna().any() or (out.pair_id.astype(str).str.strip() == "").any():
        raise ValueError("pairs.pair_id values must be nonblank")
    out["pair_id"] = out.pair_id.astype(str)
    if out.pair_id.duplicated().any():
        raise ValueError("pairs.pair_id values must be unique")

    seen: set[tuple[str, str]] = set()
    for row in out.itertuples(index=False):
        pair = canonical_pair(row.img_a, row.img_b)
        if pair != (str(row.img_a), str(row.img_b)):
            raise ValueError(f"pairs contains noncanonical pair {(row.img_a, row.img_b)!r}")
        if pair in seen:
            raise ValueError(f"pairs contains duplicate unordered pair {pair!r}")
        if make_pair_id(*pair) != row.pair_id:
            raise ValueError(f"pairs.pair_id does not match pair {pair!r}")
        seen.add(pair)
    out["img_a"] = out.img_a.astype(str)
    out["img_b"] = out.img_b.astype(str)
    unknown_states = sorted(set(out.classification.astype(str)) - PAIR_CLASSIFICATIONS)
    if unknown_states:
        raise ValueError(f"pairs has unknown classification values {unknown_states}")
    out["classification"] = out.classification.astype(str)
    for column in ("duplicate_score", "review_priority", "histogram_distance"):
        try:
            values = pd.to_numeric(out[column], errors="raise").astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"pairs.{column} must be numeric") from exc
        if not np.isfinite(values).all():
            raise ValueError(f"pairs.{column} must contain finite values")
        out[column] = values
    if ((out.duplicate_score < 0) | (out.duplicate_score > 1)).any():
        raise ValueError("pairs.duplicate_score must be in [0,1]")
    if ((out.review_priority < 0) | (out.review_priority > 1)).any():
        raise ValueError("pairs.review_priority must be in [0,1]")
    if out.exact_match.map(lambda value: isinstance(value, (bool, np.bool_))).eq(False).any():
        raise ValueError("pairs.exact_match must be boolean")
    try:
        pred_dup = pd.to_numeric(out.pred_dup, errors="raise").astype(int)
    except (TypeError, ValueError) as exc:
        raise ValueError("pairs.pred_dup must contain 0/1 integers") from exc
    expected = (out.classification == "positive").astype(int)
    if not pred_dup.isin([0, 1]).all() or not pred_dup.equals(expected):
        raise ValueError("pairs.pred_dup must be 1 exactly for positive classifications")
    if (out.loc[out.exact_match.astype(bool), "classification"] != "positive").any():
        raise ValueError("exact pairs must be classified positive")
    out["pred_dup"] = pred_dup
    return out


def _validated_groups(groups: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(groups, pd.DataFrame):
        raise ValueError(f"groups must be a pandas DataFrame, got {type(groups).__name__}")
    required = ("image_id", "duplicate_group_id", "split_group_id")
    missing = [column for column in required if column not in groups.columns]
    if missing:
        raise ValueError(f"groups is missing required columns {missing}; has {list(groups.columns)}")
    out = groups.copy()
    for column in required:
        if out[column].isna().any():
            raise ValueError(f"groups.{column} values must be non-null")
        out[column] = out[column].astype(str).str.strip()
        if (out[column] == "").any():
            raise ValueError(f"groups.{column} values must be nonblank")
    if out.image_id.duplicated().any():
        raise ValueError("groups must contain exactly one row per image_id")
    return out.sort_values("image_id", kind="mergesort", ignore_index=True)


def _validated_split(provisional_split: pd.DataFrame, image_ids: set[str]) -> pd.DataFrame:
    if not isinstance(provisional_split, pd.DataFrame):
        raise ValueError(
            f"provisional_split must be a pandas DataFrame, got {type(provisional_split).__name__}"
        )
    required = ("image_id", "split")
    missing = [column for column in required if column not in provisional_split.columns]
    if missing:
        raise ValueError(
            f"provisional_split is missing required columns {missing}; has {list(provisional_split.columns)}"
        )
    out = provisional_split.copy()
    for column in required:
        if out[column].isna().any():
            raise ValueError(f"provisional_split.{column} values must be non-null")
        out[column] = out[column].astype(str).str.strip()
        if (out[column] == "").any():
            raise ValueError(f"provisional_split.{column} values must be nonblank")
    if out.image_id.duplicated().any():
        raise ValueError("provisional_split must contain exactly one row per image_id")
    actual = set(out.image_id)
    if actual != image_ids:
        raise ValueError(
            "provisional_split must cover groups exactly; "
            f"missing={sorted(image_ids - actual)[:5]}, extra={sorted(actual - image_ids)[:5]}"
        )
    unknown = sorted(set(out.split) - {"train", "test", "drop"})
    if unknown:
        raise ValueError(f"provisional_split has unsupported split values {unknown}")
    return out.sort_values("image_id", kind="mergesort", ignore_index=True)


def _visual_signal_count(reason_codes: Any) -> int | None:
    for code in str(reason_codes).split(";"):
        if code.startswith("visual_signals:"):
            value = code.partition(":")[2]
            return int(value) if value.isdigit() else None
    return None


def rank_review_pairs(
    pairs: pd.DataFrame,
    groups: pd.DataFrame,
    provisional_split: pd.DataFrame,
    budget: int,
) -> pd.DataFrame:
    """Rank uncertain, consequential image pairs for human review.

    ``pairs`` must satisfy ``CLASSIFIED_PAIR_COLUMNS``; ``groups`` requires one
    ``image_id,duplicate_group_id,split_group_id`` row per image; and
    ``provisional_split`` requires one ``image_id,split`` row per grouped image.
    ``review_priority`` is uncertainty, while component sizes, conservative
    cross-group positive bridges, color traps, and relationships that cross the
    provisional split contribute independent impact terms.

    Selection is stratified between accepted group-derived matches, withheld
    uncertain matches, rejected raw bridges, and color traps. Within each stratum,
    distinct duplicate-component relationships are visited before repeating one.
    This prevents one family of already-suppressed bridges consuming the budget
    while accepted false merges and crop-sensitive misses never receive review.

    The resulting ``score`` is a review-ranking heuristic in ``[0,1]``, not
    duplicate confidence. Exact byte matches are excluded. Rows are deterministically
    ordered by descending review score and pair ID, capped at ``budget``, and
    returned in the shared ``PAIR_REVIEW_COLUMNS`` schema. Empty inputs/candidate
    sets return a header-only queue.

    Raises:
        ValueError: For malformed tables, inconsistent image coverage, unknown
            pair images, or a nonpositive/noninteger budget.
    """
    if isinstance(budget, bool) or not isinstance(budget, Integral) or int(budget) < 1:
        raise ValueError(f"budget must be a positive integer, got {budget!r}")
    normalized_pairs = _validated_pairs(pairs)
    normalized_groups = _validated_groups(groups)
    image_ids = set(normalized_groups.image_id)
    normalized_split = _validated_split(provisional_split, image_ids)
    pair_images = set(normalized_pairs.img_a.astype(str)) | set(normalized_pairs.img_b.astype(str))
    unknown = sorted(pair_images - image_ids)
    if unknown:
        raise ValueError(f"pairs references images absent from groups: {unknown[:5]}")

    group_by_image = normalized_groups.set_index("image_id")
    split_by_image = normalized_split.set_index("image_id").split.to_dict()
    duplicate_sizes = normalized_groups.groupby("duplicate_group_id").size().to_dict()

    rows: list[dict[str, Any]] = []
    impact_values: list[float] = []
    features: list[dict[str, Any]] = []
    for row in normalized_pairs.itertuples(index=False):
        duplicate_a = group_by_image.at[row.img_a, "duplicate_group_id"]
        duplicate_b = group_by_image.at[row.img_b, "duplicate_group_id"]
        split_group_a = group_by_image.at[row.img_a, "split_group_id"]
        split_group_b = group_by_image.at[row.img_b, "split_group_id"]
        same_duplicate_group = duplicate_a == duplicate_b
        same_split_group = split_group_a == split_group_b
        size_a = int(duplicate_sizes[duplicate_a])
        size_b = int(duplicate_sizes[duplicate_b])
        # Product estimates how many pair relationships a cross-component bridge
        # could imply; within-component review impact uses that component's size.
        impact_raw = float(size_a if same_duplicate_group else size_a * size_b)
        impact_values.append(impact_raw)
        visual_count = _visual_signal_count(row.reason_codes)
        color_trap = bool(
            "histogram_within_limit" in str(row.reason_codes).split(";")
            and visual_count is not None
            and visual_count < 2
        )
        bridge = bool(row.classification == "positive" and not same_duplicate_group)
        cross_split = split_by_image[row.img_a] != split_by_image[row.img_b]
        split_relationship_change = not same_split_group
        features.append(
            {
                "row": row,
                "impact_raw": impact_raw,
                "color_trap": color_trap,
                "bridge": bridge,
                "cross_split": cross_split,
                "split_relationship_change": split_relationship_change,
                "size_a": size_a,
                "size_b": size_b,
                "same_duplicate_group": same_duplicate_group,
                "family": tuple(sorted((str(duplicate_a), str(duplicate_b)))),
            }
        )

    max_impact = max(impact_values, default=1.0)
    impact_denominator = math.log1p(max_impact) if max_impact > 0 else 1.0
    for feature in features:
        row = feature["row"]
        if bool(row.exact_match):
            continue
        uncertainty = float(row.review_priority)
        impact = math.log1p(feature["impact_raw"]) / impact_denominator
        split_score = 1.0 if feature["cross_split"] else (
            0.5 if feature["split_relationship_change"] else 0.0
        )
        color_score = 1.0 if feature["color_trap"] else 0.0
        accepted = feature["same_duplicate_group"]
        verification_conflict = "verification_conflict" in str(row.reason_codes).split(";")

        # Avoid filling a fixed budget with certain, ordinary negatives. A pair
        # is reviewable when uncertainty or one of the declared risk slices fires.
        if not (
            uncertainty > 0
            or row.classification == "ambiguous"
            or feature["bridge"]
            or feature["color_trap"]
            or accepted
        ):
            continue
        review_score = (
            0.50 * uncertainty
            + 0.20 * impact
            + 0.10 * split_score
            + 0.05 * color_score
            + 0.10 * float(accepted)
            + 0.10 * float(verification_conflict)
        )
        flags = []
        if row.classification == "ambiguous":
            flags.append("threshold_uncertain")
        if feature["bridge"]:
            flags.append("cross_duplicate_group_bridge")
        if accepted:
            flags.append("accepted_match_audit")
        if verification_conflict:
            flags.append("verification_conflict")
        if feature["cross_split"]:
            flags.append("would_change_split")
        elif feature["split_relationship_change"]:
            flags.append("would_merge_split_groups")
        if feature["color_trap"]:
            flags.append("color_similarity_trap")
        if feature["size_a"] > 1 or feature["size_b"] > 1:
            flags.append(f"component_sizes:{feature['size_a']}x{feature['size_b']}")
        reason = (
            f"classification={row.classification};duplicate_score={float(row.duplicate_score):.6f};"
            f"final_pred_dup={int(accepted)};"
            f"uncertainty={uncertainty:.6f};impact={impact:.6f};"
            f"flags={','.join(flags) if flags else 'borderline'};evidence={row.reason_codes}"
        )
        if feature["bridge"]:
            suggested = "approve bridge, reject, or edit relation"
        elif row.classification == "ambiguous":
            suggested = "approve, reject, or edit relation"
        elif feature["color_trap"]:
            suggested = "reject or edit relation"
        else:
            suggested = "approve or reject relationship"
        slice_value = "|".join(flags) if flags else row.classification
        rows.append(
            {
                "pair_id": row.pair_id,
                "review_score": min(1.0, max(0.0, review_score)),
                "reason": reason,
                "suggested_action": suggested,
                "slice": slice_value,
                "evidence_path": "",
                "lane": "accepted" if accepted else "color" if feature["color_trap"] else "bridge" if feature["bridge"] else "withheld",
                "family": feature["family"],
            }
        )

    candidates = pd.DataFrame(
        rows,
        columns=(
            "pair_id",
            "review_score",
            "reason",
            "suggested_action",
            "slice",
            "evidence_path",
            "lane",
            "family",
        ),
    ).sort_values(["review_score", "pair_id"], ascending=[False, True], kind="mergesort")
    # Balanced risk strata, with relationship-level diversity inside each lane.
    # No labels are used: accepted membership is the final group-derived output.
    remaining = list(candidates.index)
    selected: list[int] = []
    visits: dict[tuple[Any, Any], int] = {}
    while remaining and len(selected) < int(budget):
        for lane in ("accepted", "withheld", "bridge", "color"):
            eligible = [index for index in remaining if candidates.at[index, "lane"] == lane]
            if not eligible or len(selected) >= int(budget):
                continue
            # ``min`` preserves the score/pair ordering on equal visit counts.
            index = min(eligible, key=lambda item: visits.get((lane, candidates.at[item, "family"]), 0))
            key = (lane, candidates.at[index, "family"])
            visits[key] = visits.get(key, 0) + 1
            selected.append(index)
            remaining.remove(index)
    candidates = candidates.loc[selected].sort_values(
        ["review_score", "pair_id"], ascending=[False, True], kind="mergesort"
    )
    queue = make_review_queue(
        candidates,
        id_col="pair_id",
        score_col="review_score",
        budget=int(budget),
        reason_col="reason",
        action_col="suggested_action",
        slice_col="slice",
        evidence_col="evidence_path",
        ascending=False,
    )
    if tuple(REVIEW_COLUMNS) != PAIR_REVIEW_COLUMNS:  # shared-contract invariant
        raise AssertionError("T05 pair review schema diverged from common REVIEW_COLUMNS")
    return queue[list(PAIR_REVIEW_COLUMNS)]


def _validated_queue(queue: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(queue, pd.DataFrame):
        raise ValueError(f"queue must be a pandas DataFrame, got {type(queue).__name__}")
    missing = [column for column in PAIR_REVIEW_COLUMNS if column not in queue.columns]
    if missing:
        raise ValueError(f"queue is missing required columns {missing}; has {list(queue.columns)}")
    out = queue[list(PAIR_REVIEW_COLUMNS)].copy()
    if out.item_id.isna().any() or (out.item_id.astype(str).str.strip() == "").any():
        raise ValueError("queue.item_id values must be nonblank")
    out["item_id"] = out.item_id.astype(str)
    if out.item_id.duplicated().any():
        raise ValueError("queue.item_id values must be unique")
    for item_id in out.item_id:
        _parse_pair_id(item_id)
    if len(out):
        try:
            ranks = pd.to_numeric(out["rank"], errors="raise").astype(int)
        except (TypeError, ValueError) as exc:
            raise ValueError("queue.rank must contain consecutive positive integers") from exc
        if ranks.tolist() != list(range(1, len(out) + 1)):
            raise ValueError("queue.rank must contain consecutive positive integers")
        scores = pd.to_numeric(out.score, errors="raise").astype(float)
        if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
            raise ValueError("queue.score must contain finite values in [0,1]")
        out["rank"] = ranks
        out["score"] = scores
    return out


def render_pair_review(
    queue: pd.DataFrame,
    images: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Render a self-contained offline HTML page showing both images per pair.

    ``queue`` must use ``PAIR_REVIEW_COLUMNS`` and pair IDs from
    :func:`make_pair_id`. ``images`` requires unique ``image_id,path`` rows.
    Relative paths resolve against ``images.attrs['input_dir']`` when available,
    otherwise the current directory. The common contact-sheet renderer embeds
    both thumbnails as base64 JPEG data and exports the shared review CSV schema.
    Empty queues produce a valid zero-card page.

    Returns:
        The written HTML :class:`~pathlib.Path`.

    Raises:
        ValueError: For malformed queue/image tables or unknown image IDs.
        FileNotFoundError: If either evidence image cannot be found.
    """
    normalized_queue = _validated_queue(queue)
    if not isinstance(images, pd.DataFrame):
        raise ValueError(f"images must be a pandas DataFrame, got {type(images).__name__}")
    missing = [column for column in ("image_id", "path") if column not in images.columns]
    if missing:
        raise ValueError(f"images is missing required columns {missing}; has {list(images.columns)}")
    normalized_images = images.copy()
    for column in ("image_id", "path"):
        if normalized_images[column].isna().any():
            raise ValueError(f"images.{column} values must be non-null")
        normalized_images[column] = normalized_images[column].astype(str).str.strip()
        if (normalized_images[column] == "").any():
            raise ValueError(f"images.{column} values must be nonblank")
    if normalized_images.image_id.duplicated().any():
        raise ValueError("images must contain exactly one path per image_id")
    path_by_id = normalized_images.set_index("image_id").path.to_dict()
    root = Path(str(images.attrs.get("input_dir", ".")))

    display = normalized_queue.copy()
    path_a: list[str] = []
    path_b: list[str] = []
    pair_labels: list[str] = []
    for item_id in display.item_id:
        img_a, img_b = _parse_pair_id(item_id)
        unknown = sorted({item for item in (img_a, img_b) if item not in path_by_id})
        if unknown:
            raise ValueError(f"queue pair references unknown image IDs: {unknown}")
        resolved = []
        for image_id in (img_a, img_b):
            path = Path(path_by_id[image_id])
            path = path if path.is_absolute() else root / path
            if not path.is_file():
                raise FileNotFoundError(f"review image {image_id!r} not found: {path}")
            resolved.append(str(path))
        path_a.append(resolved[0])
        path_b.append(resolved[1])
        pair_labels.append(f"{img_a} ↔ {img_b}")
    display["image_a_path"] = path_a
    display["image_b_path"] = path_b
    display["pair"] = pair_labels
    return render_contact_sheet(
        display,
        output_path,
        image_col="image_b_path",
        before_image_col="image_a_path",
        fields=("rank", "pair", "item_id", "score", "reason", "suggested_action", "slice"),
        title="T05 pair review queue",
        embed=True,
    )


def _queue_fingerprint(queue: pd.DataFrame) -> str:
    immutable = queue[list(_QUEUE_IDENTITY_COLUMNS)].copy()
    records = [
        {column: str(row[column]) for column in _QUEUE_IDENTITY_COLUMNS}
        for row in immutable.to_dict(orient="records")
    ]
    return sha256_text(json.dumps(records, sort_keys=True, ensure_ascii=False, separators=(",", ":")))


def write_review_context(
    queue: pd.DataFrame,
    input_fingerprint: str,
    config_fingerprint: str,
    output_path: str | Path = "review_context.json",
) -> Path:
    """Write deterministic queue identity and run fingerprints for stale checks.

    The queue must use ``PAIR_REVIEW_COLUMNS``. Input/config fingerprints are
    caller-supplied, nonblank identifiers for the exact input and frozen policy.
    Queue identity hashes only immutable exported fields, so reviewer-entered
    decision columns can change without invalidating the original queue match.

    Returns:
        Path to the written JSON context. Changing the input fingerprint, config
        fingerprint, queue order, or immutable queue content changes
        ``context_fingerprint``.

    Raises:
        ValueError: For a malformed queue or blank fingerprint.
    """
    normalized_queue = _validated_queue(queue)
    fingerprints = {}
    for name, value in (
        ("input_fingerprint", input_fingerprint),
        ("config_fingerprint", config_fingerprint),
    ):
        if value is None or not str(value).strip():
            raise ValueError(f"{name} must be nonblank")
        fingerprints[name] = str(value).strip()
    queue_fingerprint = _queue_fingerprint(normalized_queue)
    identity = {
        **fingerprints,
        "queue_fingerprint": queue_fingerprint,
    }
    context = {
        "schema_version": _REVIEW_CONTEXT_VERSION,
        **identity,
        "context_fingerprint": sha256_text(
            json.dumps(identity, sort_keys=True, separators=(",", ":"))
        ),
        "queue_size": len(normalized_queue),
        "item_ids": normalized_queue.item_id.tolist(),
        "queue_identity_columns": list(_QUEUE_IDENTITY_COLUMNS),
        "review_columns": list(PAIR_REVIEW_COLUMNS),
    }
    return write_json(output_path, context)


def _load_context(context: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(context, Mapping):
        return dict(context)
    if isinstance(context, (str, Path)):
        path = Path(context)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read review context at {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("review context JSON must contain an object")
        return payload
    raise ValueError(
        f"context must be a mapping or JSON path, got {type(context).__name__}"
    )


def _validated_context(queue: pd.DataFrame, context: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    payload = _load_context(context)
    required = {
        "schema_version",
        "input_fingerprint",
        "config_fingerprint",
        "queue_fingerprint",
        "context_fingerprint",
        "queue_size",
        "item_ids",
        "queue_identity_columns",
        "review_columns",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"review context is missing required fields {missing}")
    if payload["schema_version"] != _REVIEW_CONTEXT_VERSION:
        raise ValueError(
            f"unsupported review context schema_version {payload['schema_version']!r}"
        )
    for name in ("input_fingerprint", "config_fingerprint", "queue_fingerprint"):
        if not str(payload[name]).strip():
            raise ValueError(f"review context {name} must be nonblank")
    identity = {
        "input_fingerprint": str(payload["input_fingerprint"]),
        "config_fingerprint": str(payload["config_fingerprint"]),
        "queue_fingerprint": str(payload["queue_fingerprint"]),
    }
    expected_context_fingerprint = sha256_text(
        json.dumps(identity, sort_keys=True, separators=(",", ":"))
    )
    if str(payload["context_fingerprint"]) != expected_context_fingerprint:
        raise ValueError("review context fingerprint is invalid or has been altered")
    actual_queue_fingerprint = _queue_fingerprint(queue)
    if identity["queue_fingerprint"] != actual_queue_fingerprint:
        raise ValueError("stale review context: queue fingerprint does not match current queue")
    if payload["queue_size"] != len(queue):
        raise ValueError("stale review context: queue size does not match current queue")
    if list(payload["item_ids"]) != queue.item_id.tolist():
        raise ValueError("stale review context: ordered item IDs do not match current queue")
    if tuple(payload["queue_identity_columns"]) != _QUEUE_IDENTITY_COLUMNS:
        raise ValueError("review context queue identity schema does not match")
    if tuple(payload["review_columns"]) != PAIR_REVIEW_COLUMNS:
        raise ValueError("review context review-column schema does not match")
    return payload


def _normalized_review_decisions(reviewed: pd.DataFrame) -> pd.DataFrame:
    """Validate reviewer/action/override semantics without queue identity checks."""
    if not isinstance(reviewed, pd.DataFrame):
        raise ValueError(f"reviewed must be a pandas DataFrame, got {type(reviewed).__name__}")
    missing = [column for column in PAIR_REVIEW_COLUMNS if column not in reviewed.columns]
    if missing:
        raise ValueError(f"reviewed is missing required columns {missing}; has {list(reviewed.columns)}")
    out = reviewed[list(PAIR_REVIEW_COLUMNS)].copy().fillna("")
    out["item_id"] = out.item_id.astype(str)
    if out.item_id.duplicated().any():
        raise ValueError("reviewed contains duplicate item_id rows")
    for item_id in out.item_id:
        _parse_pair_id(item_id)

    out["reviewer_decision"] = out.reviewer_decision.astype(str).str.strip().str.lower()
    out = out[out.reviewer_decision != ""].copy()
    unknown = sorted(set(out.reviewer_decision) - set(DEFAULT_ALLOWED_DECISIONS))
    if unknown:
        raise ValueError(
            f"unknown reviewer_decision values {unknown}; allowed={sorted(DEFAULT_ALLOWED_DECISIONS)}"
        )
    out["reviewer"] = out.reviewer.astype(str).str.strip()
    out["review_note"] = out.review_note.astype(str).str.strip()
    out["override_value"] = out.override_value.astype(str).str.strip()
    if (out.reviewer == "").any():
        item_ids = out.loc[out.reviewer == "", "item_id"].tolist()[:5]
        raise ValueError(f"reviewed decisions require a reviewer; items={item_ids}")
    material = out.reviewer_decision != "skip"
    if (out.loc[material, "review_note"] == "").any():
        item_ids = out.loc[material & (out.review_note == ""), "item_id"].tolist()[:5]
        raise ValueError(f"material reviewed decisions require a review_note; items={item_ids}")

    relations: list[str] = []
    for row in out.itertuples(index=False):
        decision = row.reviewer_decision
        override = row.override_value
        if decision == "approve":
            if override not in {"", DUPLICATE_RELATION}:
                raise ValueError(
                    f"approve for {row.item_id!r} allows only blank or {DUPLICATE_RELATION!r} override"
                )
            relation = DUPLICATE_RELATION
        elif decision == "reject":
            if override not in {SCENE_ONLY_RELATION, DIFFERENT_SCENE_RELATION}:
                raise ValueError(
                    f"reject for {row.item_id!r} requires explicit "
                    f"{SCENE_ONLY_RELATION!r} or {DIFFERENT_SCENE_RELATION!r} override"
                )
            relation = override
        elif decision == "edit":
            if override not in RELATION_OVERRIDE_VALUES:
                raise ValueError(
                    f"edit for {row.item_id!r} requires one of {sorted(RELATION_OVERRIDE_VALUES)}"
                )
            relation = override
        else:  # escalate / skip do not silently alter a relationship
            if override:
                raise ValueError(f"{decision} for {row.item_id!r} must not set override_value")
            relation = ""
        relations.append(relation)
    out["review_relation"] = relations
    return out[list(PAIR_REVIEW_COLUMNS) + ["review_relation"]].reset_index(drop=True)


def _same_identity_value(column: str, reviewed_value: Any, queue_value: Any) -> bool:
    if column == "rank":
        try:
            return int(reviewed_value) == int(queue_value)
        except (TypeError, ValueError):
            return False
    if column == "score":
        try:
            return math.isclose(float(reviewed_value), float(queue_value), abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    left = "" if pd.isna(reviewed_value) else str(reviewed_value)
    right = "" if pd.isna(queue_value) else str(queue_value)
    return left == right


def validate_pair_reviews(
    reviewed: pd.DataFrame,
    queue: pd.DataFrame,
    context: Mapping[str, Any] | str | Path,
) -> pd.DataFrame:
    """Validate reviewed rows against the exact queue and fingerprint context.

    The reviewed table may be the full HTML export or only decided rows, but it
    must retain the shared review schema. Immutable queue fields must match the
    original row exactly, unknown/duplicate items are rejected, reviewer identity
    and notes are required for material decisions, and T05 relation overrides
    follow the explicit duplicate/same-scene/different-scene vocabulary.

    ``reject`` must explicitly choose ``same_scene_only`` or ``different_scene``;
    rejecting duplication alone does not establish scene semantics. ``escalate``
    and ``skip`` remain unresolved and cannot carry an override.

    Returns:
        Decided rows only, in shared review columns plus normalized
        ``review_relation``. Input tables are not mutated.

    Raises:
        ValueError: For stale/tampered fingerprints, queue drift, unknown items,
            missing reviewer/note, invalid actions, or invalid overrides.
    """
    normalized_queue = _validated_queue(queue)
    _validated_context(normalized_queue, context)
    if not isinstance(reviewed, pd.DataFrame):
        raise ValueError(f"reviewed must be a pandas DataFrame, got {type(reviewed).__name__}")
    missing = [column for column in PAIR_REVIEW_COLUMNS if column not in reviewed.columns]
    if missing:
        raise ValueError(f"reviewed is missing required columns {missing}; has {list(reviewed.columns)}")
    raw = reviewed[list(PAIR_REVIEW_COLUMNS)].copy().fillna("")
    raw["item_id"] = raw.item_id.astype(str)
    if raw.item_id.duplicated().any():
        raise ValueError("reviewed contains duplicate item_id rows")
    queue_by_id = normalized_queue.set_index("item_id")
    unknown = sorted(set(raw.item_id) - set(queue_by_id.index))
    if unknown:
        raise ValueError(f"reviewed contains unknown queue item IDs: {unknown[:5]}")
    for row in raw.itertuples(index=False):
        source = queue_by_id.loc[row.item_id]
        for column in _QUEUE_IDENTITY_COLUMNS:
            reviewed_value = getattr(row, column)
            queue_value = row.item_id if column == "item_id" else source[column]
            if not _same_identity_value(column, reviewed_value, queue_value):
                raise ValueError(
                    f"stale reviewed row {row.item_id!r}: immutable field {column!r} changed"
                )
    return _normalized_review_decisions(raw)


def apply_pair_reviews(
    classified_pairs: pd.DataFrame,
    reviewed: pd.DataFrame,
    audit: AuditLog | None = None,
) -> pd.DataFrame:
    """Apply validated T05 review decisions to a copy of classified pairs.

    ``approve``/``edit=duplicate`` set a direct pair positive. Nonduplicate
    relations set it negative and are later emitted as grouping constraints.
    ``escalate`` and ``skip`` leave automatic classification unchanged and mark
    the row unresolved. Automatic scores/evidence are retained. Every decided
    row is appended to ``audit`` with reviewer, before/after state, note, and the
    review timestamp; existing audit history is never rewritten.

    Returns:
        ``REVIEWED_PAIR_COLUMNS``. The original automatic table is not mutated.

    Raises:
        ValueError: For malformed decisions, unknown pair IDs, attempts to reject
            exact byte matches, or an invalid audit object.
    """
    automatic = _validated_pairs(classified_pairs)[list(CLASSIFIED_PAIR_COLUMNS)]
    decisions = _normalized_review_decisions(reviewed)
    if audit is not None and not isinstance(audit, AuditLog):
        raise ValueError(f"audit must be AuditLog or None, got {type(audit).__name__}")
    unknown = sorted(set(decisions.item_id) - set(automatic.pair_id.astype(str)))
    if unknown:
        raise ValueError(f"reviewed decisions reference unknown pair IDs: {unknown[:5]}")

    out = automatic.copy()
    out["review_decision"] = ""
    out["review_relation"] = ""
    out["reviewer"] = ""
    out["review_note"] = ""
    out["reviewed_at"] = ""
    out["review_unresolved"] = False
    for review in decisions.itertuples(index=False):
        mask = out.pair_id.astype(str) == review.item_id
        current = out.loc[mask].iloc[0]
        relation = review.review_relation
        if bool(current.exact_match) and relation in {SCENE_ONLY_RELATION, DIFFERENT_SCENE_RELATION}:
            raise ValueError(f"cannot reject exact byte-match pair {review.item_id!r}")
        before = {
            "classification": str(current.classification),
            "pred_dup": int(current.pred_dup),
            "relation": "automatic",
        }
        unresolved = review.reviewer_decision in {"escalate", "skip"}
        if relation == DUPLICATE_RELATION:
            out.loc[mask, "classification"] = "positive"
            out.loc[mask, "pred_dup"] = 1
        elif relation in {SCENE_ONLY_RELATION, DIFFERENT_SCENE_RELATION}:
            out.loc[mask, "classification"] = "negative"
            out.loc[mask, "pred_dup"] = 0
        out.loc[mask, "review_decision"] = review.reviewer_decision
        out.loc[mask, "review_relation"] = relation
        out.loc[mask, "reviewer"] = review.reviewer
        out.loc[mask, "review_note"] = review.review_note
        out.loc[mask, "reviewed_at"] = review.reviewed_at
        out.loc[mask, "review_unresolved"] = unresolved
        suffix = f"human_review:{review.reviewer_decision}"
        if relation:
            suffix += f";human_relation:{relation}"
        existing_reasons = str(current.reason_codes).strip(";")
        out.loc[mask, "reason_codes"] = f"{existing_reasons};{suffix}" if existing_reasons else suffix
        after = {
            "classification": str(out.loc[mask, "classification"].iloc[0]),
            "pred_dup": int(out.loc[mask, "pred_dup"].iloc[0]),
            "relation": relation or "unresolved",
            "unresolved": unresolved,
        }
        if audit is not None:
            audit.log(
                actor=review.reviewer,
                action=f"t05_pair_review:{review.reviewer_decision}",
                item_id=review.item_id,
                before=before,
                after=after,
                reason=review.review_note,
                reviewed_at=review.reviewed_at,
            )
    return out[list(REVIEWED_PAIR_COLUMNS)]


def _constraints_from_reviewed_pairs(reviewed_pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in reviewed_pairs.itertuples(index=False):
        if row.review_relation:
            rows.append(
                {"img_a": row.img_a, "img_b": row.img_b, "relation": row.review_relation}
            )
    return pd.DataFrame(rows, columns=PAIR_CONSTRAINT_COLUMNS).sort_values(
        list(PAIR_CONSTRAINT_COLUMNS), kind="mergesort", ignore_index=True
    )


def _predictions_from_groups(
    classified_pairs: pd.DataFrame,
    expanded_pairs: pd.DataFrame,
) -> pd.DataFrame:
    positives = set(
        expanded_pairs[["img_a", "img_b"]].itertuples(index=False, name=None)
    )
    rows = [
        {
            "img_a": row.img_a,
            "img_b": row.img_b,
            "score": float(row.duplicate_score),
            "pred_dup": int((row.img_a, row.img_b) in positives),
        }
        for row in classified_pairs.itertuples(index=False)
    ]
    return pd.DataFrame(rows, columns=PREDICTION_COLUMNS).sort_values(
        ["img_a", "img_b"], kind="mergesort", ignore_index=True
    )


def rebuild_after_review(
    images: pd.DataFrame,
    classified_pairs: pd.DataFrame,
    reviewed: pd.DataFrame,
    params: Mapping[str, Any] | None,
    rng: np.random.Generator,
    audit: AuditLog | None = None,
) -> dict[str, Any]:
    """Apply reviews and deterministically rebuild duplicate groups and split.

    Automatic classified pairs are copied and retained separately. Reviewed
    duplicate/nonduplicate relations become explicit constraints for both
    conservative duplicate grouping and broader split grouping, preventing a
    rejected pair from returning through another graph path. Final evaluator
    predictions are generated from all within-group pairs, not raw positive
    edges. The full split is then regenerated with the supplied RNG.

    Args:
        images: Public image metadata used by Tasks 7–8.
        classified_pairs: Immutable automatic Task 6 output.
        reviewed: Validated or directly supplied shared-schema decided rows.
        params: Frozen T05 parameters.
        rng: NumPy generator controlling split-group tie breaks.
        audit: Optional append-only shared :class:`AuditLog`.

    Returns:
        A dictionary containing automatic/final classified pairs, automatic/final
        group-derived prediction tables, constraints, duplicate groups, expanded
        positive pairs, split groups/manifest/summary, unresolved rows, and review
        diagnostics. Escalations/skips set review recommendation to ``HOLD``.

    Raises:
        ValueError: For invalid reviews/inputs or contradictory reviewed
            constraints. Grouping failures are never silently bypassed.
    """
    cfg = validate_params(params)
    automatic_pairs = _validated_pairs(classified_pairs)[list(CLASSIFIED_PAIR_COLUMNS)]
    automatic_groups = build_duplicate_groups(images, automatic_pairs)
    automatic_expanded = expand_group_pairs(automatic_groups)
    automatic_predictions = _predictions_from_groups(automatic_pairs, automatic_expanded)

    final_pairs = apply_pair_reviews(automatic_pairs, reviewed, audit)
    constraints = _constraints_from_reviewed_pairs(final_pairs)
    duplicate_groups = build_duplicate_groups(images, final_pairs, constraints)
    expanded_pairs = expand_group_pairs(duplicate_groups)
    predictions = _predictions_from_groups(final_pairs, expanded_pairs)
    split_groups = build_split_groups(
        images,
        duplicate_groups,
        final_pairs,
        overrides=constraints,
        params=cfg,
    )
    split_manifest = assign_group_splits(
        split_groups,
        float(cfg["test_fraction"]),
        rng,
    )
    split_summary = summarize_split(images, split_manifest, split_groups)
    unresolved = final_pairs[final_pairs.review_unresolved].copy().reset_index(drop=True)
    decisions = final_pairs.loc[final_pairs.review_decision != "", "review_decision"]
    decision_counts = {
        decision: int((decisions == decision).sum())
        for decision in sorted(DEFAULT_ALLOWED_DECISIONS)
    }
    changed = int(
        (
            final_pairs.set_index("pair_id").classification
            != automatic_pairs.set_index("pair_id").classification
        ).sum()
    )
    review_summary = {
        "n_reviewed": int(len(decisions)),
        "decision_counts": decision_counts,
        "changed_pair_count": changed,
        "final_prediction_changed_count": int(
            (predictions.pred_dup.to_numpy() != automatic_predictions.pred_dup.to_numpy()).sum()
        ),
        "constraint_count": len(constraints),
        "unresolved_count": len(unresolved),
        "escalation_count": decision_counts.get("escalate", 0),
        "skip_count": decision_counts.get("skip", 0),
        "unresolved_item_ids": unresolved.pair_id.tolist(),
        "recommended_status": "HOLD" if len(unresolved) else "PROVISIONAL",
    }
    combined_summary = dict(split_summary)
    combined_summary["review"] = review_summary
    if len(unresolved):
        combined_summary["recommended_status"] = "HOLD"

    return {
        "automatic_classified_pairs": automatic_pairs,
        "automatic_predictions": automatic_predictions,
        "final_classified_pairs": final_pairs,
        "predictions": predictions,
        "constraints": constraints,
        "duplicate_groups": duplicate_groups,
        "expanded_pairs": expanded_pairs,
        "split_groups": split_groups,
        "split_manifest": split_manifest,
        "split_summary": combined_summary,
        "unresolved_reviews": unresolved,
        "review_summary": review_summary,
    }
