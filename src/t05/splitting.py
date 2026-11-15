"""Scene-aware split constraints and deterministic group assignment for T05.

Duplicate groups remain distinct from broader split groups. A split group may
join duplicate components, visually supported same-camera bursts, and explicit
same-scene links, but it never creates a duplicate prediction.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

from .contracts import (
    DIFFERENT_SCENE_RELATION,
    DUPLICATE_GROUP_COLUMNS,
    DUPLICATE_RELATION,
    PAIR_CONSTRAINT_COLUMNS,
    RELATION_OVERRIDE_VALUES,
    SCENE_ONLY_RELATION,
    SPLIT_GROUP_COLUMNS,
    SPLIT_MANIFEST_COLUMNS,
    canonical_pair,
    validate_params,
)


class _UnionFind:
    """Deterministic union-find keyed by normalized image ID."""

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
        keep, merge = sorted((root_a, root_b))
        self.parent[merge] = keep

    def members(self, item: str) -> set[str]:
        root = self.find(item)
        return {candidate for candidate in self.parent if self.find(candidate) == root}


def _validated_images(images: pd.DataFrame) -> pd.DataFrame:
    """Validate split-relevant public metadata and return canonical ID order."""
    if not isinstance(images, pd.DataFrame):
        raise ValueError(f"images must be a pandas DataFrame, got {type(images).__name__}")
    required = ("image_id", "camera_id", "captured_at")
    missing = [column for column in required if column not in images.columns]
    if missing:
        raise ValueError(f"images is missing required columns {missing}; has {list(images.columns)}")
    if images.empty:
        raise ValueError("images must contain at least one row")

    out = images.copy()
    for column in ("image_id", "camera_id"):
        if out[column].isna().any():
            raise ValueError(f"images.{column} values must be non-null")
        out[column] = out[column].astype(str).str.strip()
        if (out[column] == "").any():
            raise ValueError(f"images.{column} values must be nonblank")
    duplicates = sorted(out.loc[out.image_id.duplicated(keep=False), "image_id"].unique())
    if duplicates:
        raise ValueError(f"images has duplicate image_id values: {duplicates[:5]}")
    timestamps = pd.to_datetime(out.captured_at, format="ISO8601", errors="coerce", utc=True)
    if timestamps.isna().any():
        rows = out.index[timestamps.isna()].tolist()[:5]
        raise ValueError(f"images.captured_at has invalid timestamps at rows {rows}")
    out["captured_at"] = timestamps
    return out.sort_values("image_id", kind="mergesort", ignore_index=True)


def _validated_duplicate_groups(
    duplicate_groups: pd.DataFrame,
    image_ids: set[str],
) -> pd.DataFrame:
    if not isinstance(duplicate_groups, pd.DataFrame):
        raise ValueError(
            f"duplicate_groups must be a pandas DataFrame, got {type(duplicate_groups).__name__}"
        )
    missing = [column for column in DUPLICATE_GROUP_COLUMNS if column not in duplicate_groups.columns]
    if missing:
        raise ValueError(
            f"duplicate_groups is missing required columns {missing}; has {list(duplicate_groups.columns)}"
        )
    out = duplicate_groups.copy()
    for column in DUPLICATE_GROUP_COLUMNS:
        if out[column].isna().any():
            raise ValueError(f"duplicate_groups.{column} values must be non-null")
        out[column] = out[column].astype(str).str.strip()
        if (out[column] == "").any():
            raise ValueError(f"duplicate_groups.{column} values must be nonblank")
    if out.image_id.duplicated().any():
        raise ValueError("duplicate_groups must contain exactly one row per image_id")
    actual = set(out.image_id)
    if actual != image_ids:
        missing_ids = sorted(image_ids - actual)
        extra_ids = sorted(actual - image_ids)
        raise ValueError(
            "duplicate_groups must cover images exactly; "
            f"missing={missing_ids[:5]}, extra={extra_ids[:5]}"
        )
    return out.sort_values("image_id", kind="mergesort", ignore_index=True)


def _validated_overrides(
    overrides: pd.DataFrame | None,
    image_ids: set[str],
) -> pd.DataFrame:
    if overrides is None:
        return pd.DataFrame(columns=PAIR_CONSTRAINT_COLUMNS)
    if not isinstance(overrides, pd.DataFrame):
        raise ValueError(f"overrides must be a pandas DataFrame, got {type(overrides).__name__}")
    missing = [column for column in PAIR_CONSTRAINT_COLUMNS if column not in overrides.columns]
    if missing:
        raise ValueError(f"overrides is missing required columns {missing}; has {list(overrides.columns)}")
    rows: list[dict[str, str]] = []
    for row in overrides.itertuples(index=False):
        img_a, img_b = canonical_pair(row.img_a, row.img_b)
        relation = str(row.relation).strip()
        if relation not in RELATION_OVERRIDE_VALUES:
            raise ValueError(
                f"unknown override relation {relation!r}; allowed={sorted(RELATION_OVERRIDE_VALUES)}"
            )
        unknown = sorted({item for item in (img_a, img_b) if item not in image_ids})
        if unknown:
            raise ValueError(f"override references unknown image IDs: {unknown}")
        rows.append({"img_a": img_a, "img_b": img_b, "relation": relation})
    out = pd.DataFrame(rows, columns=PAIR_CONSTRAINT_COLUMNS).drop_duplicates()
    relation_counts = out.groupby(["img_a", "img_b"]).relation.nunique()
    conflicts = relation_counts[relation_counts > 1]
    if len(conflicts):
        raise ValueError(f"pairs have conflicting relation overrides: {list(conflicts.index)[:5]}")
    return out.sort_values(list(PAIR_CONSTRAINT_COLUMNS), kind="mergesort", ignore_index=True)


def _validated_pair_evidence(
    pair_evidence: pd.DataFrame,
    images: pd.DataFrame,
) -> pd.DataFrame:
    if not isinstance(pair_evidence, pd.DataFrame):
        raise ValueError(
            f"pair_evidence must be a pandas DataFrame, got {type(pair_evidence).__name__}"
        )
    required = (
        "pair_id",
        "img_a",
        "img_b",
        "phash_distance",
        "dhash_distance",
        "thumbnail_distance",
        "same_camera",
        "time_delta_seconds",
    )
    missing = [column for column in required if column not in pair_evidence.columns]
    if missing:
        raise ValueError(
            f"pair_evidence is missing required columns {missing}; has {list(pair_evidence.columns)}"
        )
    out = pair_evidence.copy()
    if out.pair_id.isna().any() or (out.pair_id.astype(str).str.strip() == "").any():
        raise ValueError("pair_evidence pair_id values must be nonblank")
    if out.pair_id.astype(str).duplicated().any():
        raise ValueError("pair_evidence pair_id values must be unique")

    metadata = images.set_index("image_id")
    known_ids = set(metadata.index)
    seen: set[tuple[str, str]] = set()
    normalized_pairs: list[tuple[str, str]] = []
    for row in out.itertuples(index=False):
        pair = canonical_pair(row.img_a, row.img_b)
        if pair != (str(row.img_a), str(row.img_b)):
            raise ValueError(f"pair_evidence pair is not canonical: {(row.img_a, row.img_b)!r}")
        if pair in seen:
            raise ValueError(f"pair_evidence contains duplicate unordered pair {pair!r}")
        unknown = sorted(set(pair) - known_ids)
        if unknown:
            raise ValueError(f"pair_evidence references unknown image IDs: {unknown}")
        seen.add(pair)
        normalized_pairs.append(pair)
    expected_pair_count = len(known_ids) * (len(known_ids) - 1) // 2
    if len(seen) != expected_pair_count:
        raise ValueError(
            "pair_evidence must cover every unordered image pair; "
            f"expected={expected_pair_count}, actual={len(seen)}"
        )

    for column in ("phash_distance", "dhash_distance", "thumbnail_distance", "time_delta_seconds"):
        try:
            values = pd.to_numeric(out[column], errors="raise").astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"pair_evidence.{column} must be numeric") from exc
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"pair_evidence.{column} must contain finite nonnegative values")
        out[column] = values

    for index, (img_a, img_b) in enumerate(normalized_pairs):
        same_camera = metadata.at[img_a, "camera_id"] == metadata.at[img_b, "camera_id"]
        recorded_same = out.iloc[index].same_camera
        if not isinstance(recorded_same, (bool, np.bool_)) or bool(recorded_same) != same_camera:
            raise ValueError(f"pair_evidence.same_camera disagrees with images for pair {(img_a, img_b)!r}")
        delta = abs(
            (metadata.at[img_a, "captured_at"] - metadata.at[img_b, "captured_at"]).total_seconds()
        )
        if not math.isclose(float(out.iloc[index].time_delta_seconds), delta, abs_tol=1e-6):
            raise ValueError(
                f"pair_evidence.time_delta_seconds disagrees with images for pair {(img_a, img_b)!r}"
            )
    return out


def _crosses_cannot_link(
    union: _UnionFind,
    img_a: str,
    img_b: str,
    cannot_pairs: set[tuple[str, str]],
) -> bool:
    members_a, members_b = union.members(img_a), union.members(img_b)
    return any(
        canonical_pair(a, b) in cannot_pairs
        for a in members_a
        for b in members_b
        if a != b
    )


def _visual_scene_compatible(row: Any, params: Mapping[str, int | float]) -> bool:
    """Require one spatial/structural signal; histogram or time alone is unsafe."""
    return bool(
        row.phash_distance <= params["phash_distance_max"]
        or row.dhash_distance <= params["dhash_distance_max"]
        or row.thumbnail_distance <= params["thumbnail_distance_max"]
    )


def build_split_groups(
    images: pd.DataFrame,
    duplicate_groups: pd.DataFrame,
    pair_evidence: pd.DataFrame,
    overrides: pd.DataFrame | None = None,
    params: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Build broader scene/split groups without changing duplicate groups.

    Inputs require public image metadata (``image_id``, ``camera_id``, and UTC-
    parseable ``captured_at``), one duplicate-group row per image, canonical pair
    evidence with distances/time in seconds, and optional relation overrides.

    Duplicate groups and ``duplicate``/``same_scene_only`` overrides are must-link
    constraints. ``different_scene`` is a cannot-link. Automatic scene links
    require the same camera, a time delta no larger than
    ``burst_window_seconds``, and at least one compatible pHash, dHash, or
    thumbnail signal. Histogram/time alone never links scenes. Pure temporal
    burst components are constrained to ``burst_max_span_seconds`` before they
    are combined with duplicate groups, preventing unbounded timestamp chaining.

    Returns:
        ``SPLIT_GROUP_COLUMNS`` sorted by image ID. Component-size columns are
        diagnostics only; oversized groups remain intact and must produce HOLD
        risk in :func:`summarize_split` rather than silent image drops.

    Raises:
        ValueError: On malformed/incomplete inputs or contradictory overrides.
    """
    cfg = validate_params(params)
    normalized_images = _validated_images(images)
    ids = normalized_images.image_id.tolist()
    known_ids = set(ids)
    groups = _validated_duplicate_groups(duplicate_groups, known_ids)
    evidence = _validated_pair_evidence(pair_evidence, normalized_images)
    normalized_overrides = _validated_overrides(overrides, known_ids)

    union = _UnionFind(ids)
    duplicate_group_of = dict(zip(groups.image_id, groups.duplicate_group_id))
    for _, subset in groups.groupby("duplicate_group_id", sort=True):
        members = sorted(subset.image_id.tolist())
        for member in members[1:]:
            union.union(members[0], member)

    relations = {
        (row.img_a, row.img_b): row.relation
        for row in normalized_overrides.itertuples(index=False)
    }
    # Both nonduplicate relations contradict an already accepted duplicate group.
    for pair, relation in sorted(relations.items()):
        if relation in {SCENE_ONLY_RELATION, DIFFERENT_SCENE_RELATION}:
            if duplicate_group_of[pair[0]] == duplicate_group_of[pair[1]]:
                raise ValueError(
                    f"override {relation!r} contradicts duplicate group for pair {pair!r}"
                )

    cannot_pairs = {
        pair for pair, relation in relations.items() if relation == DIFFERENT_SCENE_RELATION
    }
    must_pairs = {
        pair
        for pair, relation in relations.items()
        if relation in {DUPLICATE_RELATION, SCENE_ONLY_RELATION}
    }
    for img_a, img_b in sorted(must_pairs):
        if _crosses_cannot_link(union, img_a, img_b, cannot_pairs):
            raise ValueError(
                f"required split relation {(img_a, img_b)!r} would violate different_scene"
            )
        union.union(img_a, img_b)
    for img_a, img_b in sorted(cannot_pairs):
        if union.find(img_a) == union.find(img_b):
            raise ValueError(
                f"different_scene override {(img_a, img_b)!r} conflicts with required split links"
            )

    # Select bounded burst edges in a separate graph. Duplicate edges can span
    # capture dates and intentionally do not inflate a local burst's time span.
    burst_union = _UnionFind(ids)
    timestamp_ns = {
        row.image_id: int(row.captured_at.value)
        for row in normalized_images.itertuples(index=False)
    }
    window = float(cfg["burst_window_seconds"])
    max_span = float(cfg["burst_max_span_seconds"])
    candidates = evidence[
        evidence.same_camera.astype(bool) & (evidence.time_delta_seconds <= window)
    ].sort_values(["time_delta_seconds", "pair_id"], kind="mergesort")
    accepted_burst_edges: list[tuple[str, str]] = []
    for row in candidates.itertuples(index=False):
        if not _visual_scene_compatible(row, cfg):
            continue
        members = burst_union.members(row.img_a) | burst_union.members(row.img_b)
        span_seconds = (max(timestamp_ns[item] for item in members) - min(
            timestamp_ns[item] for item in members
        )) / 1_000_000_000
        if span_seconds <= max_span:
            burst_union.union(row.img_a, row.img_b)
            accepted_burst_edges.append((row.img_a, row.img_b))

    for img_a, img_b in accepted_burst_edges:
        if union.find(img_a) == union.find(img_b):
            continue
        if _crosses_cannot_link(union, img_a, img_b, cannot_pairs):
            continue
        union.union(img_a, img_b)

    components: dict[str, list[str]] = {}
    for image_id in ids:
        components.setdefault(union.find(image_id), []).append(image_id)
    split_group_of: dict[str, str] = {}
    sizes: dict[str, int] = {}
    for members in sorted((sorted(values) for values in components.values()), key=lambda x: x[0]):
        first = members[0]
        group_id = f"split:{len(first)}:{first}"
        sizes[group_id] = len(members)
        split_group_of.update({member: group_id for member in members})

    maximum = int(cfg["max_component_size"])
    rows = [
        {
            "image_id": image_id,
            "duplicate_group_id": duplicate_group_of[image_id],
            "split_group_id": split_group_of[image_id],
            "split_group_size": sizes[split_group_of[image_id]],
            "exceeds_max_component_size": sizes[split_group_of[image_id]] > maximum,
        }
        for image_id in ids
    ]
    result = pd.DataFrame(rows, columns=SPLIT_GROUP_COLUMNS)
    result.attrs["max_component_size"] = maximum
    result.attrs["accepted_burst_edge_count"] = len(accepted_burst_edges)
    return result


def _validated_fraction(test_fraction: float) -> float:
    if isinstance(test_fraction, bool) or not isinstance(test_fraction, Real):
        raise ValueError(f"test_fraction must be a finite number in (0,1), got {test_fraction!r}")
    value = float(test_fraction)
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError(f"test_fraction must be a finite number in (0,1), got {test_fraction!r}")
    return value


def assign_group_splits(
    split_groups: pd.DataFrame,
    test_fraction: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Assign whole split groups to train/test with deterministic greedy balance.

    Groups are ordered by descending size with RNG-controlled tie-breaking based
    on canonical group IDs. A group is added to test when doing so improves the
    absolute image-count distance from the requested test fraction. If at least
    two groups exist, a final deterministic correction guarantees nonempty train
    and test sets. One indivisible group is assigned to train and later reported
    as an impossible two-way split. No rows are silently dropped.

    Returns:
        ``SPLIT_MANIFEST_COLUMNS`` sorted by image ID.
    """
    fraction = _validated_fraction(test_fraction)
    if not isinstance(rng, np.random.Generator):
        raise ValueError(f"rng must be numpy.random.Generator, got {type(rng).__name__}")
    if not isinstance(split_groups, pd.DataFrame):
        raise ValueError(f"split_groups must be a pandas DataFrame, got {type(split_groups).__name__}")
    missing = [column for column in ("image_id", "split_group_id") if column not in split_groups.columns]
    if missing:
        raise ValueError(f"split_groups is missing required columns {missing}; has {list(split_groups.columns)}")
    if split_groups.image_id.isna().any() or split_groups.split_group_id.isna().any():
        raise ValueError("split_groups image_id and split_group_id values must be non-null")
    normalized = split_groups.copy()
    normalized["image_id"] = normalized.image_id.astype(str).str.strip()
    normalized["split_group_id"] = normalized.split_group_id.astype(str).str.strip()
    if (normalized.image_id == "").any() or (normalized.split_group_id == "").any():
        raise ValueError("split_groups image_id and split_group_id values must be nonblank")
    if normalized.image_id.duplicated().any():
        raise ValueError("split_groups must contain exactly one row per image_id")

    sizes = normalized.groupby("split_group_id", sort=True).size().to_dict()
    group_ids = sorted(sizes)
    permutation = rng.permutation(len(group_ids))
    tie_rank = {group_ids[int(index)]: rank for rank, index in enumerate(permutation)}
    ordered = sorted(group_ids, key=lambda group_id: (-sizes[group_id], tie_rank[group_id], group_id))
    target = len(normalized) * fraction
    test_groups: set[str] = set()
    n_test = 0
    for group_id in ordered:
        proposed = n_test + sizes[group_id]
        if abs(proposed - target) < abs(n_test - target):
            test_groups.add(group_id)
            n_test = proposed

    if len(group_ids) >= 2 and not test_groups:
        chosen = min(group_ids, key=lambda group_id: (abs(sizes[group_id] - target), tie_rank[group_id], group_id))
        test_groups.add(chosen)
        n_test = sizes[chosen]
    if len(group_ids) >= 2 and len(test_groups) == len(group_ids):
        chosen = min(
            test_groups,
            key=lambda group_id: (abs((n_test - sizes[group_id]) - target), tie_rank[group_id], group_id),
        )
        test_groups.remove(chosen)

    rows = [
        {
            "image_id": row.image_id,
            "split": "test" if row.split_group_id in test_groups else "train",
        }
        for row in normalized.sort_values("image_id", kind="mergesort").itertuples(index=False)
    ]
    result = pd.DataFrame(rows, columns=SPLIT_MANIFEST_COLUMNS)
    result.attrs["target_test_fraction"] = fraction
    return result


def summarize_split(
    images: pd.DataFrame,
    assignments: pd.DataFrame,
    split_groups: pd.DataFrame,
) -> dict[str, Any]:
    """Summarize observable split balance, coverage, and release-blocking risks.

    This function uses no ground truth. It validates exact image coverage, counts
    explicit drops, verifies that split groups remain intact, and reports target
    deviation (fractions are in ``[0,1]``; counts are image/group counts).
    Oversized components or an impossible/unusable two-way split are returned as
    blocking risks with ``recommended_status='HOLD'``; images are never silently
    dropped to manufacture a successful split. Results and risk ordering are
    deterministic for the supplied tables.

    Returns:
        A diagnostics dictionary containing coverage, balance, component-size,
        crossing, risk, and provisional status fields.

    Raises:
        ValueError: If tables are malformed or do not cover the same images.
    """
    if not isinstance(images, pd.DataFrame) or "image_id" not in images.columns:
        raise ValueError("images must be a DataFrame with image_id")
    if images.image_id.isna().any():
        raise ValueError("images.image_id values must be unique and nonblank")
    image_ids = images.image_id.astype(str).str.strip()
    if (image_ids == "").any() or image_ids.duplicated().any():
        raise ValueError("images.image_id values must be unique and nonblank")
    expected = set(image_ids)

    if not isinstance(assignments, pd.DataFrame):
        raise ValueError(f"assignments must be a pandas DataFrame, got {type(assignments).__name__}")
    missing = [column for column in SPLIT_MANIFEST_COLUMNS if column not in assignments.columns]
    if missing:
        raise ValueError(f"assignments is missing required columns {missing}; has {list(assignments.columns)}")
    normalized_assignments = assignments.copy()
    if normalized_assignments.image_id.isna().any() or normalized_assignments.split.isna().any():
        raise ValueError("assignments image_id and split values must be non-null")
    normalized_assignments["image_id"] = normalized_assignments.image_id.astype(str).str.strip()
    normalized_assignments["split"] = normalized_assignments.split.astype(str).str.strip()
    if normalized_assignments.image_id.duplicated().any():
        raise ValueError("assignments must contain exactly one row per image_id")
    actual = set(normalized_assignments.image_id)
    if actual != expected:
        raise ValueError(
            "assignments must cover images exactly; "
            f"missing={sorted(expected - actual)[:5]}, extra={sorted(actual - expected)[:5]}"
        )
    allowed = {"train", "test", "drop"}
    unknown = sorted(set(normalized_assignments.split) - allowed)
    if unknown:
        raise ValueError(f"assignments has unsupported split values {unknown}; allowed={sorted(allowed)}")

    if not isinstance(split_groups, pd.DataFrame):
        raise ValueError(f"split_groups must be a pandas DataFrame, got {type(split_groups).__name__}")
    missing = [column for column in ("image_id", "split_group_id") if column not in split_groups.columns]
    if missing:
        raise ValueError(f"split_groups is missing required columns {missing}; has {list(split_groups.columns)}")
    normalized_groups = split_groups.copy()
    if normalized_groups.image_id.isna().any() or normalized_groups.split_group_id.isna().any():
        raise ValueError("split_groups image_id and split_group_id values must be non-null")
    normalized_groups["image_id"] = normalized_groups.image_id.astype(str).str.strip()
    normalized_groups["split_group_id"] = normalized_groups.split_group_id.astype(str).str.strip()
    if (normalized_groups.image_id == "").any() or (normalized_groups.split_group_id == "").any():
        raise ValueError("split_groups image_id and split_group_id values must be nonblank")
    if normalized_groups.image_id.duplicated().any() or set(normalized_groups.image_id) != expected:
        raise ValueError("split_groups must contain exactly one row for every image")

    merged = normalized_groups[["image_id", "split_group_id"]].merge(
        normalized_assignments[["image_id", "split"]], on="image_id", how="left", validate="one_to_one"
    )
    crossing = merged.groupby("split_group_id").split.nunique()
    crossing_ids = sorted(crossing[crossing > 1].index.tolist())
    group_sizes = normalized_groups.groupby("split_group_id").size().sort_index()
    oversized_ids: list[str] = []
    if "exceeds_max_component_size" in normalized_groups.columns:
        flags = normalized_groups.copy()
        flags["exceeds_max_component_size"] = flags.exceeds_max_component_size.astype(bool)
        oversized_ids = sorted(
            flags.loc[flags.exceeds_max_component_size, "split_group_id"].unique().tolist()
        )

    counts = normalized_assignments.split.value_counts()
    n_train = int(counts.get("train", 0))
    n_test = int(counts.get("test", 0))
    n_drop = int(counts.get("drop", 0))
    retained = n_train + n_test
    target = assignments.attrs.get("target_test_fraction")
    target_fraction = float(target) if target is not None else None
    actual_fraction = n_test / retained if retained else 0.0
    possible = len(group_sizes) >= 2 and len(expected) >= 2
    usable = n_train > 0 and n_test > 0

    risks: list[dict[str, Any]] = []
    if crossing_ids:
        risks.append(
            {
                "code": "SPLIT_GROUP_CROSSES_ASSIGNMENTS",
                "blocking": True,
                "count": len(crossing_ids),
                "group_ids": crossing_ids[:20],
            }
        )
    if oversized_ids:
        risks.append(
            {
                "code": "OVERSIZED_SPLIT_COMPONENT",
                "blocking": True,
                "count": len(oversized_ids),
                "group_ids": oversized_ids[:20],
            }
        )
    if not possible:
        risks.append(
            {
                "code": "IMPOSSIBLE_TWO_WAY_GROUP_SPLIT",
                "blocking": True,
                "count": 1,
                "detail": "fewer than two indivisible split groups",
            }
        )
    elif not usable:
        risks.append(
            {
                "code": "UNUSABLE_TWO_WAY_SPLIT",
                "blocking": True,
                "count": 1,
                "detail": "valid grouping permits two splits but assignment left one empty",
            }
        )

    blocking = [risk for risk in risks if risk.get("blocking")]
    return {
        "n_images": len(expected),
        "n_split_groups": int(len(group_sizes)),
        "n_train": n_train,
        "n_test": n_test,
        "n_drop": n_drop,
        "retained_count": retained,
        "retention_fraction": retained / len(expected) if expected else 0.0,
        "target_test_fraction": target_fraction,
        "actual_test_fraction": actual_fraction,
        "test_fraction_deviation": (
            abs(actual_fraction - target_fraction) if target_fraction is not None else None
        ),
        "largest_split_group_size": int(group_sizes.max()) if len(group_sizes) else 0,
        "split_group_size_distribution": {
            str(int(size)): int(count)
            for size, count in group_sizes.value_counts().sort_index().items()
        },
        "cross_split_group_count": len(crossing_ids),
        "oversized_split_group_count": len(oversized_ids),
        "two_way_split_possible": possible,
        "usable_train_test": usable,
        "risks": risks,
        "blocking_risk_count": len(blocking),
        "recommended_status": "HOLD" if blocking else "PROVISIONAL",
    }
