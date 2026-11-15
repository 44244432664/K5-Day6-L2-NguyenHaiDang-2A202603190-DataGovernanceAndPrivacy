"""Interpretable automatic near-duplicate scoring and classification for T05.

The score is a deterministic heuristic, not a calibrated probability. Visual
signals determine classification; camera/time context is deliberately excluded
so capture proximity alone can never prove a duplicate.
"""
from __future__ import annotations

from collections.abc import Mapping
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from .contracts import (
    CLASSIFIED_PAIR_COLUMNS,
    DIFFERENT_SCENE_RELATION,
    DUPLICATE_GROUP_COLUMNS,
    DUPLICATE_RELATION,
    GROUP_CONFLICT_COLUMNS,
    GROUP_PAIR_COLUMNS,
    PAIR_CLASSIFICATIONS,
    PAIR_CONSTRAINT_COLUMNS,
    PAIR_EVIDENCE_COLUMNS,
    RELATION_OVERRIDE_VALUES,
    SCENE_ONLY_RELATION,
    canonical_pair,
    validate_params,
)
from .verification import VERIFICATION_COLUMNS


# Histogram is supporting evidence only. Even a perfect histogram contributes
# at most 0.10, while the three spatial/structural visual signals contribute
# 0.30 each.
_WEIGHTS = {
    "phash": 0.30,
    "dhash": 0.30,
    "thumbnail": 0.30,
    "histogram": 0.10,
}


def _validated_evidence(pair_evidence: pd.DataFrame) -> pd.DataFrame:
    """Return a validated copy of the raw pair table."""
    if not isinstance(pair_evidence, pd.DataFrame):
        raise ValueError(
            f"pair_evidence must be a pandas DataFrame, got {type(pair_evidence).__name__}"
        )
    missing = [column for column in PAIR_EVIDENCE_COLUMNS if column not in pair_evidence.columns]
    if missing:
        raise ValueError(
            f"pair_evidence is missing required columns {missing}; has {list(pair_evidence.columns)}"
        )
    out = pair_evidence.copy()
    if out.pair_id.isna().any() or (out.pair_id.astype(str).str.strip() == "").any():
        raise ValueError("pair_evidence pair_id values must be nonblank")
    if out.pair_id.astype(str).duplicated().any():
        raise ValueError("pair_evidence pair_id values must be unique")

    seen: set[tuple[str, str]] = set()
    for row in out.itertuples(index=False):
        pair = canonical_pair(row.img_a, row.img_b)
        if pair != (str(row.img_a), str(row.img_b)):
            raise ValueError(f"pair_evidence pair is not canonical: {(row.img_a, row.img_b)!r}")
        if pair in seen:
            raise ValueError(f"pair_evidence contains duplicate unordered pair {pair!r}")
        seen.add(pair)
        if not isinstance(row.exact_match, (bool, np.bool_)):
            raise ValueError(f"exact_match must be boolean for pair {pair!r}")

    bounds = {
        "phash_distance": None,
        "dhash_distance": None,
        "histogram_distance": 2.0,
        "thumbnail_distance": 1.0,
        "time_delta_seconds": None,
    }
    for column, maximum in bounds.items():
        try:
            values = pd.to_numeric(out[column], errors="raise").astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"pair_evidence.{column} must be numeric") from exc
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"pair_evidence.{column} must contain finite nonnegative values")
        if maximum is not None and (values > maximum).any():
            raise ValueError(f"pair_evidence.{column} must be <= {maximum}")
        out[column] = values
    present = set(VERIFICATION_COLUMNS) & set(out.columns)
    if present and present != set(VERIFICATION_COLUMNS):
        raise ValueError("verification evidence columns must be supplied together")
    for column in present:
        values = pd.to_numeric(out[column], errors="raise").astype(float)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError(f"pair_evidence.{column} must contain finite values in [0,1]")
        out[column] = values
    return out


def _verified(frame: pd.DataFrame, cfg: Mapping[str, int | float]) -> np.ndarray:
    return (
        (frame.verification_error.to_numpy() <= cfg["verification_error_max"])
        & (frame.verification_texture.to_numpy() >= cfg["verification_min_texture"])
    )


def _similarity(distance: pd.Series, limit: float) -> np.ndarray:
    """Map distance to [0,1], with configured limit mapping to 0.75."""
    values = distance.to_numpy(dtype=float)
    if limit == 0:
        return (values == 0).astype(float)
    return np.clip(1.0 - 0.25 * values / float(limit), 0.0, 1.0)


def _signal_masks(frame: pd.DataFrame, params: Mapping[str, int | float]) -> dict[str, np.ndarray]:
    return {
        "phash": frame.phash_distance.to_numpy() <= params["phash_distance_max"],
        "dhash": frame.dhash_distance.to_numpy() <= params["dhash_distance_max"],
        "thumbnail": frame.thumbnail_distance.to_numpy() <= params["thumbnail_distance_max"],
        "histogram": frame.histogram_distance.to_numpy() <= params["histogram_distance_max"],
    }


def _reason_codes(
    exact: bool,
    masks: Mapping[str, bool],
    visual_count: int,
    suffix: str | None = None,
) -> str:
    codes = ["exact_sha256"] if exact else []
    codes.extend(f"{name}_within_limit" for name in ("phash", "dhash", "thumbnail", "histogram") if masks[name])
    codes.append(f"visual_signals:{visual_count}")
    if suffix:
        codes.append(suffix)
    return ";".join(codes)


def _review_priority(scores: np.ndarray, params: Mapping[str, int | float]) -> np.ndarray:
    """Triangular uncertainty: 1 at decision threshold, 0 at review-band edges."""
    low = float(params["review_score_low"])
    threshold = float(params["duplicate_score_threshold"])
    high = float(params["review_score_high"])
    priority = np.zeros_like(scores, dtype=float)
    left = (scores >= low) & (scores <= threshold)
    right = (scores > threshold) & (scores <= high)
    if threshold == low:
        priority[left] = 1.0
    else:
        priority[left] = (scores[left] - low) / (threshold - low)
    if high == threshold:
        priority[right] = 1.0
    else:
        priority[right] = (high - scores[right]) / (high - threshold)
    return np.clip(priority, 0.0, 1.0)


def score_duplicate_evidence(
    pair_evidence: pd.DataFrame,
    params: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Assign deterministic visual duplicate scores and evidence reason codes.

    The heuristic is ``0.30*pHash + 0.30*dHash + 0.30*thumbnail +
    0.10*histogram``. Each component is a threshold-relative similarity in
    ``[0,1]`` whose configured distance limit maps to 0.75. Exact SHA-256
    equality overrides the score to 1.0. When protocol-v2 RGB verification is
    supplied, a textured verified match also supplies a threshold-relative
    similarity (0.75 at its relative-MAE limit). The original unboosted score is
    retained as ``coarse_duplicate_score`` for development comparisons.

    Camera and timestamp fields are preserved but never enter the score. The
    returned score is not a probability. The input table is not mutated.
    """
    cfg = validate_params(params)
    out = _validated_evidence(pair_evidence)
    if not cfg["verification_enabled"]:
        out = out.drop(columns=[*VERIFICATION_COLUMNS, "coarse_duplicate_score"], errors="ignore")
    similarities = {
        "phash": _similarity(out.phash_distance, float(cfg["phash_distance_max"])),
        "dhash": _similarity(out.dhash_distance, float(cfg["dhash_distance_max"])),
        "thumbnail": _similarity(
            out.thumbnail_distance, float(cfg["thumbnail_distance_max"])
        ),
        "histogram": _similarity(
            out.histogram_distance, float(cfg["histogram_distance_max"])
        ),
    }
    scores = sum(_WEIGHTS[name] * similarities[name] for name in _WEIGHTS)
    if "verification_error" in out:
        out["coarse_duplicate_score"] = scores.copy()
        registered = np.clip(1.0 - 0.25 * out.verification_error.to_numpy() / float(cfg["verification_error_max"]), 0, 1)
        scores = np.where(_verified(out, cfg), np.maximum(scores, registered), scores)
    exact = out.exact_match.to_numpy(dtype=bool)
    # Decimal weights can leave mathematically exact 1.0 as 0.9999999999999999;
    # rounding keeps boundary policy deterministic without changing resolution.
    scores = np.round(np.where(exact, 1.0, np.clip(scores, 0.0, 1.0)), 12)
    masks = _signal_masks(out, cfg)
    visual_counts = masks["phash"].astype(int) + masks["dhash"].astype(int) + masks[
        "thumbnail"
    ].astype(int)

    out["duplicate_score"] = scores
    priority = _review_priority(scores, cfg)
    priority[exact] = 0.0  # exact byte identity is not an uncertain visual match
    out["review_priority"] = priority
    out["reason_codes"] = [
        _reason_codes(
            bool(exact[index]),
            {name: bool(values[index]) for name, values in masks.items()},
            int(visual_counts[index]),
        )
        for index in range(len(out))
    ]
    return out


def classify_pairs(
    scored_pairs: pd.DataFrame,
    params: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Classify scored pairs as positive, negative, or ambiguous.

    Exact SHA-256 matches are positive. A visual near-duplicate is positive only
    when its score reaches ``duplicate_score_threshold`` *and* at least
    ``min_visual_signals`` among pHash, dHash, and thumbnail are within their
    limits. Histogram evidence cannot satisfy this gate. Protocol-v2 evidence
    additionally requires textured RGB verification for nonexact positives;
    registered crops can recover positives despite changed global hashes.
    Legacy tables without verification retain the v1 gate for explicit ablations.

    Non-positive pairs in the configured review band, and high-scoring pairs
    that fail the visual-signal gate, are ambiguous. Only positives receive
    ``pred_dup=1``; ambiguous decisions are conservatively withheld for review.
    """
    cfg = validate_params(params)
    out = _validated_evidence(scored_pairs)
    for column in ("duplicate_score", "review_priority"):
        try:
            values = pd.to_numeric(scored_pairs[column], errors="raise").astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"scored_pairs.{column} must be numeric") from exc
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError(f"scored_pairs.{column} must contain finite values in [0,1]")
        out[column] = values.to_numpy()

    masks = _signal_masks(out, cfg)
    visual_counts = masks["phash"].astype(int) + masks["dhash"].astype(int) + masks[
        "thumbnail"
    ].astype(int)
    scores = out.duplicate_score.to_numpy(dtype=float)
    exact = out.exact_match.to_numpy(dtype=bool)
    enough_visual = visual_counts >= int(cfg["min_visual_signals"])
    positive = exact | ((scores >= float(cfg["duplicate_score_threshold"])) & enough_visual)
    conflicting = (~exact) & (scores >= float(cfg["duplicate_score_threshold"])) & (~enough_visual)
    verification_conflict = np.zeros(len(out), dtype=bool)
    if "verification_error" in out:
        verified = _verified(out, cfg)
        crop = out.verification_crop_fraction.to_numpy() > 0
        coarse_positive = (scores >= float(cfg["duplicate_score_threshold"])) & enough_visual
        verification_conflict = (~exact) & coarse_positive & (~verified)
        positive = exact | (verified & (scores >= float(cfg["duplicate_score_threshold"])) & (enough_visual | crop))
        conflicting = conflicting & (~positive)
    in_review_band = (scores >= float(cfg["review_score_low"])) & (
        scores <= float(cfg["review_score_high"])
    )
    ambiguous = (~positive) & (in_review_band | conflicting | verification_conflict)
    classification = np.where(positive, "positive", np.where(ambiguous, "ambiguous", "negative"))

    # Ambiguous pairs must remain reviewable even at an exact review-band edge;
    # conflicting high scores receive maximum review priority.
    priority = out.review_priority.to_numpy(dtype=float, copy=True)
    priority[ambiguous] = np.maximum(priority[ambiguous], 0.5)
    priority[conflicting] = 1.0
    priority[verification_conflict] = 1.0
    priority[exact] = 0.0
    out["review_priority"] = priority
    out["reason_codes"] = [
        _reason_codes(
            bool(exact[index]),
            {name: bool(values[index]) for name, values in masks.items()},
            int(visual_counts[index]),
            suffix=(
                "auto_positive"
                if classification[index] == "positive"
                else "needs_review"
                if classification[index] == "ambiguous"
                else "auto_negative"
            ),
        )
        for index in range(len(out))
    ]
    out["classification"] = classification
    out["pred_dup"] = positive.astype(int)
    if "verification_error" in out:
        reasons = []
        for index, reason in enumerate(out.reason_codes):
            flag = "rgb_verified" if verified[index] else "rgb_unverified"
            if verification_conflict[index]:
                flag += ";verification_conflict"
            if positive[index] and not exact[index] and crop[index]:
                flag += ";registered_crop"
            reasons.append(reason + ";" + flag)
        out["reason_codes"] = reasons

    if not set(out.classification).issubset(PAIR_CLASSIFICATIONS):  # defensive invariant
        raise AssertionError("internal T05 classification invariant failed")
    extra = [column for column in (*VERIFICATION_COLUMNS, "coarse_duplicate_score") if column in out]
    return out[list(CLASSIFIED_PAIR_COLUMNS) + extra]


class _UnionFind:
    """Small deterministic union-find keyed by image ID."""

    def __init__(self, items: list[str]):
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def union(self, a: str, b: str) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a == root_b:
            return
        # Stable root selection makes group construction independent of input
        # row order when merge decisions are otherwise equal.
        keep, merge = sorted((root_a, root_b))
        self.parent[merge] = keep

    def members(self, item: str) -> set[str]:
        root = self.find(item)
        return {candidate for candidate in self.parent if self.find(candidate) == root}


def _image_ids(images: pd.DataFrame) -> list[str]:
    if not isinstance(images, pd.DataFrame):
        raise ValueError(f"images must be a pandas DataFrame, got {type(images).__name__}")
    if "image_id" not in images.columns:
        raise ValueError(f"images is missing required column 'image_id'; has {list(images.columns)}")
    if images.image_id.isna().any():
        raise ValueError("images has null image_id values")
    ids = images.image_id.astype(str)
    blank = ids.str.strip() == ""
    if blank.any():
        raise ValueError(f"images has blank image_id values at rows {images.index[blank].tolist()[:5]}")
    duplicates = sorted(ids[ids.duplicated(keep=False)].unique().tolist())
    if duplicates:
        raise ValueError(f"images has duplicate image_id values: {duplicates[:5]}")
    return sorted(ids.tolist())


def _validated_classified_pairs(classified_pairs: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(classified_pairs, pd.DataFrame):
        raise ValueError(
            f"classified_pairs must be a pandas DataFrame, got {type(classified_pairs).__name__}"
        )
    missing = [column for column in CLASSIFIED_PAIR_COLUMNS if column not in classified_pairs.columns]
    if missing:
        raise ValueError(
            f"classified_pairs is missing required columns {missing}; has {list(classified_pairs.columns)}"
        )
    out = _validated_evidence(classified_pairs)
    states = out.classification.astype(str)
    unknown = sorted(set(states) - PAIR_CLASSIFICATIONS)
    if unknown:
        raise ValueError(f"classified_pairs has unknown classification values {unknown}")
    try:
        pred_dup = pd.to_numeric(out.pred_dup, errors="raise").astype(int)
    except (TypeError, ValueError) as exc:
        raise ValueError("classified_pairs.pred_dup must contain 0/1 integers") from exc
    if not pred_dup.isin([0, 1]).all():
        raise ValueError("classified_pairs.pred_dup must contain only 0/1")
    expected = (states == "positive").astype(int)
    if not pred_dup.equals(expected):
        raise ValueError("classified_pairs.pred_dup must be 1 exactly for positive classifications")
    if ((out.exact_match.astype(bool)) & (states != "positive")).any():
        raise ValueError("exact SHA-256 pairs must be classified positive")
    for column in ("duplicate_score", "review_priority"):
        values = pd.to_numeric(out[column], errors="raise").astype(float)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError(f"classified_pairs.{column} must contain finite values in [0,1]")
        out[column] = values
    out["classification"] = states
    out["pred_dup"] = pred_dup
    return out[list(CLASSIFIED_PAIR_COLUMNS)]


def _normalized_constraints(
    constraints: pd.DataFrame | None,
    known_ids: set[str] | None = None,
) -> pd.DataFrame:
    if constraints is None:
        return pd.DataFrame(columns=PAIR_CONSTRAINT_COLUMNS)
    if not isinstance(constraints, pd.DataFrame):
        raise ValueError(f"constraints must be a pandas DataFrame, got {type(constraints).__name__}")
    missing = [column for column in PAIR_CONSTRAINT_COLUMNS if column not in constraints.columns]
    if missing:
        raise ValueError(f"constraints is missing required columns {missing}; has {list(constraints.columns)}")
    rows = []
    for row in constraints.itertuples(index=False):
        img_a, img_b = canonical_pair(row.img_a, row.img_b)
        relation = str(row.relation).strip()
        if relation not in RELATION_OVERRIDE_VALUES:
            raise ValueError(
                f"unknown constraint relation {relation!r}; allowed={sorted(RELATION_OVERRIDE_VALUES)}"
            )
        if known_ids is not None and (img_a not in known_ids or img_b not in known_ids):
            unknown = sorted({item for item in (img_a, img_b) if item not in known_ids})
            raise ValueError(f"constraint references unknown image IDs: {unknown}")
        rows.append({"img_a": img_a, "img_b": img_b, "relation": relation})
    return (
        pd.DataFrame(rows, columns=PAIR_CONSTRAINT_COLUMNS)
        .drop_duplicates()
        .sort_values(list(PAIR_CONSTRAINT_COLUMNS), kind="mergesort")
        .reset_index(drop=True)
    )


def _groups_from_union(ids: list[str], union: _UnionFind) -> pd.DataFrame:
    components: dict[str, list[str]] = {}
    for image_id in ids:
        components.setdefault(union.find(image_id), []).append(image_id)
    rows = []
    for members in sorted((sorted(values) for values in components.values()), key=lambda x: x[0]):
        first = members[0]
        group_id = f"dup:{len(first)}:{first}"
        rows.extend({"image_id": image_id, "duplicate_group_id": group_id} for image_id in members)
    return pd.DataFrame(rows, columns=DUPLICATE_GROUP_COLUMNS).sort_values(
        "image_id", kind="mergesort", ignore_index=True
    )


def _constraint_relation_sets(constraints: pd.DataFrame) -> dict[tuple[str, str], set[str]]:
    relations: dict[tuple[str, str], set[str]] = {}
    for row in constraints.itertuples(index=False):
        relations.setdefault((row.img_a, row.img_b), set()).add(row.relation)
    return relations


def find_group_conflicts(
    groups: pd.DataFrame,
    constraints: pd.DataFrame | None,
) -> pd.DataFrame:
    """Report required/forbidden relation conflicts against duplicate groups.

    ``duplicate`` requires both images in one group. ``same_scene_only`` and
    ``different_scene`` both forbid a duplicate-group merge; they remain
    distinct for later scene-aware splitting. Conflicts are returned rather than
    silently resolved.
    """
    if not isinstance(groups, pd.DataFrame):
        raise ValueError(f"groups must be a pandas DataFrame, got {type(groups).__name__}")
    missing = [column for column in DUPLICATE_GROUP_COLUMNS if column not in groups.columns]
    if missing:
        raise ValueError(f"groups is missing required columns {missing}; has {list(groups.columns)}")
    if groups.image_id.isna().any() or groups.duplicate_group_id.isna().any():
        raise ValueError("groups image_id and duplicate_group_id values must be non-null")
    normalized_groups = groups.copy()
    normalized_groups["image_id"] = normalized_groups.image_id.astype(str)
    normalized_groups["duplicate_group_id"] = normalized_groups.duplicate_group_id.astype(str)
    if (normalized_groups.image_id.str.strip() == "").any() or (
        normalized_groups.duplicate_group_id.str.strip() == ""
    ).any():
        raise ValueError("groups image_id and duplicate_group_id values must be nonblank")
    if normalized_groups.image_id.duplicated().any():
        raise ValueError("groups must contain exactly one row per image_id")

    group_of = dict(zip(normalized_groups.image_id, normalized_groups.duplicate_group_id))
    normalized_constraints = _normalized_constraints(constraints)
    rows: list[dict[str, str]] = []
    relations = _constraint_relation_sets(normalized_constraints)
    for pair, relation_set in sorted(relations.items()):
        img_a, img_b = pair
        if img_a not in group_of or img_b not in group_of:
            rows.append(
                {
                    "conflict_type": "unknown_constraint_image",
                    "img_a": img_a,
                    "img_b": img_b,
                    "relation": "|".join(sorted(relation_set)),
                    "duplicate_group_id": "",
                    "detail": "constraint references an image absent from groups",
                }
            )
            continue
        if len(relation_set) > 1:
            rows.append(
                {
                    "conflict_type": "conflicting_pair_relations",
                    "img_a": img_a,
                    "img_b": img_b,
                    "relation": "|".join(sorted(relation_set)),
                    "duplicate_group_id": group_of[img_a] if group_of[img_a] == group_of[img_b] else "",
                    "detail": "one pair has multiple incompatible relation overrides",
                }
            )
        same_group = group_of[img_a] == group_of[img_b]
        if DUPLICATE_RELATION in relation_set and not same_group:
            rows.append(
                {
                    "conflict_type": "required_duplicate_separated",
                    "img_a": img_a,
                    "img_b": img_b,
                    "relation": DUPLICATE_RELATION,
                    "duplicate_group_id": "",
                    "detail": "duplicate constraint requires one group",
                }
            )
        forbidden = relation_set & {SCENE_ONLY_RELATION, DIFFERENT_SCENE_RELATION}
        if forbidden and same_group:
            rows.append(
                {
                    "conflict_type": "forbidden_duplicate_merged",
                    "img_a": img_a,
                    "img_b": img_b,
                    "relation": "|".join(sorted(forbidden)),
                    "duplicate_group_id": group_of[img_a],
                    "detail": "nonduplicate constraint forbids one duplicate group",
                }
            )
    return pd.DataFrame(rows, columns=GROUP_CONFLICT_COLUMNS)


def _raise_conflicts(conflicts: pd.DataFrame, stage: str) -> None:
    if len(conflicts):
        preview = conflicts.head(5).to_dict(orient="records")
        raise ValueError(f"duplicate-group constraint conflicts at {stage}: {preview}")


def build_duplicate_groups(
    images: pd.DataFrame,
    classified_pairs: pd.DataFrame,
    constraints: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build deterministic duplicate groups with complete-link safeguards.

    Exact pairs are grouped first. Explicit ``duplicate`` constraints are then
    applied. Remaining automatic positive edges are considered by descending
    score and stable pair ID, but components merge only when *every* cross-pair
    is positive/required and no nonduplicate constraint crosses the merge.

    Raises:
        ValueError: For malformed inputs, unknown IDs, contradictory constraints,
            or exact/required relations that violate cannot-link constraints.
    """
    ids = _image_ids(images)
    known_ids = set(ids)
    pairs = _validated_classified_pairs(classified_pairs)
    pair_ids = set(pairs.img_a) | set(pairs.img_b)
    unknown_pairs = sorted(pair_ids - known_ids)
    if unknown_pairs:
        raise ValueError(f"classified_pairs references unknown image IDs: {unknown_pairs[:5]}")
    normalized_constraints = _normalized_constraints(constraints, known_ids)
    relation_sets = _constraint_relation_sets(normalized_constraints)
    direct_conflicts = [pair for pair, relations in relation_sets.items() if len(relations) > 1]
    if direct_conflicts:
        raise ValueError(f"pairs have conflicting relation constraints: {direct_conflicts[:5]}")

    cannot_pairs = {
        pair
        for pair, relations in relation_sets.items()
        if relations & {SCENE_ONLY_RELATION, DIFFERENT_SCENE_RELATION}
    }
    forced_pairs = {
        pair for pair, relations in relation_sets.items() if DUPLICATE_RELATION in relations
    }
    positive_pairs = {
        (row.img_a, row.img_b)
        for row in pairs.itertuples(index=False)
        if row.classification == "positive"
    }
    exact_pairs = {
        (row.img_a, row.img_b)
        for row in pairs.itertuples(index=False)
        if bool(row.exact_match)
    }

    union = _UnionFind(ids)
    for img_a, img_b in sorted(exact_pairs):
        union.union(img_a, img_b)
    exact_conflicts = find_group_conflicts(_groups_from_union(ids, union), normalized_constraints)
    _raise_conflicts(
        exact_conflicts[exact_conflicts.conflict_type.isin(
            ["conflicting_pair_relations", "forbidden_duplicate_merged", "unknown_constraint_image"]
        )],
        "exact grouping",
    )

    for img_a, img_b in sorted(forced_pairs):
        members_a, members_b = union.members(img_a), union.members(img_b)
        if any(canonical_pair(a, b) in cannot_pairs for a in members_a for b in members_b if a != b):
            raise ValueError(
                f"duplicate constraint {(img_a, img_b)!r} would violate a nonduplicate constraint"
            )
        union.union(img_a, img_b)
    _raise_conflicts(find_group_conflicts(_groups_from_union(ids, union), normalized_constraints), "required grouping")

    accepted_pairs = positive_pairs | forced_pairs
    candidates = pairs[pairs.classification == "positive"].sort_values(
        ["duplicate_score", "pair_id"], ascending=[False, True], kind="mergesort"
    )
    for row in candidates.itertuples(index=False):
        if bool(row.exact_match):
            continue
        members_a, members_b = union.members(row.img_a), union.members(row.img_b)
        if members_a == members_b:
            continue
        cross_pairs = {
            canonical_pair(a, b) for a in members_a for b in members_b if a != b
        }
        if cross_pairs & cannot_pairs:
            continue
        if cross_pairs and cross_pairs.issubset(accepted_pairs):
            union.union(row.img_a, row.img_b)

    groups = _groups_from_union(ids, union)
    _raise_conflicts(find_group_conflicts(groups, normalized_constraints), "automatic grouping")
    return groups


def expand_group_pairs(groups: pd.DataFrame) -> pd.DataFrame:
    """Expand duplicate groups into all unordered within-group pairs.

    This is the evaluator-facing unit: a group of size ``m`` contributes
    ``m(m-1)/2`` pairs. Singleton groups contribute none.
    """
    if not isinstance(groups, pd.DataFrame):
        raise ValueError(f"groups must be a pandas DataFrame, got {type(groups).__name__}")
    missing = [column for column in DUPLICATE_GROUP_COLUMNS if column not in groups.columns]
    if missing:
        raise ValueError(f"groups is missing required columns {missing}; has {list(groups.columns)}")
    if groups.image_id.isna().any() or groups.duplicate_group_id.isna().any():
        raise ValueError("groups image_id and duplicate_group_id values must be non-null")
    normalized = groups.copy()
    normalized["image_id"] = normalized.image_id.astype(str)
    normalized["duplicate_group_id"] = normalized.duplicate_group_id.astype(str)
    if (normalized.image_id.str.strip() == "").any() or (
        normalized.duplicate_group_id.str.strip() == ""
    ).any():
        raise ValueError("groups image_id and duplicate_group_id values must be nonblank")
    if normalized.image_id.duplicated().any():
        raise ValueError("groups must contain exactly one row per image_id")

    rows = []
    for group_id, subset in normalized.groupby("duplicate_group_id", sort=True):
        members = sorted(subset.image_id.tolist())
        rows.extend(
            {"img_a": img_a, "img_b": img_b, "duplicate_group_id": group_id}
            for img_a, img_b in combinations(members, 2)
        )
    return pd.DataFrame(rows, columns=GROUP_PAIR_COLUMNS)
