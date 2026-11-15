"""Artifact validation, observable risk diagnostics, and release policy for T05.

Structural contract violations raise immediately. Structurally valid but unsafe
results remain representable as diagnostics and deterministic GO/HOLD/REWORK
decisions without consulting instructor ground truth.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd

from ..common.decision import Decision
from .contracts import GROUP_ASSIGNMENT_COLUMNS, SPLIT_MANIFEST_COLUMNS, canonical_pair


_MODES = frozenset({"poc", "baseline"})
_ALLOWED_SPLITS = frozenset({"train", "test", "drop"})


def _normalized_image_ids(image_ids: Iterable[Any]) -> list[str]:
    if isinstance(image_ids, (str, bytes)) or not isinstance(image_ids, Iterable):
        raise ValueError("image_ids must be an iterable of unique nonblank IDs")
    raw = list(image_ids)
    for value in raw:
        if value is None:
            raise ValueError("image_ids must not contain null values")
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)) and bool(missing):
            raise ValueError("image_ids must not contain null values")
    normalized = [str(value).strip() for value in raw]
    if any(value == "" for value in normalized):
        raise ValueError("image_ids must not contain blank values")
    duplicates = sorted({value for value in normalized if normalized.count(value) > 1})
    if duplicates:
        raise ValueError(f"image_ids contains duplicate values: {duplicates[:5]}")
    return sorted(normalized)


def _normalized_mode(mode: str) -> str:
    value = str(mode).strip().lower()
    if value not in _MODES:
        raise ValueError(f"mode must be one of {sorted(_MODES)}, got {mode!r}")
    return value


def validate_predictions(
    predictions: pd.DataFrame,
    image_ids: Iterable[Any],
) -> pd.DataFrame:
    """Validate and normalize evaluator-facing T05 pair predictions.

    Required columns are ``img_a,img_b,pred_dup``; optional ``score`` values are
    interpreted as duplicate confidence/evidence scores in ``[0,1]``. Pairs must
    be canonical, unique, non-self, and reference known images. ``pred_dup`` must
    be exactly 0/1. Empty tables with required headers are valid because a run may
    have no candidate positives.

    Returns:
        A copied table sorted by ``img_a,img_b`` with normalized string IDs,
        integer flags, and numeric score when present. Extra audit columns are
        preserved.

    Raises:
        ValueError: For malformed schema, IDs, pairs, flags, scores, or unknown
            images. The caller's table is never mutated.
    """
    known = set(_normalized_image_ids(image_ids))
    if not isinstance(predictions, pd.DataFrame):
        raise ValueError(
            f"predictions must be a pandas DataFrame, got {type(predictions).__name__}"
        )
    required = ("img_a", "img_b", "pred_dup")
    missing = [column for column in required if column not in predictions.columns]
    if missing:
        raise ValueError(
            f"predictions is missing required columns {missing}; has {list(predictions.columns)}"
        )
    out = predictions.copy()
    for column in ("img_a", "img_b"):
        if out[column].isna().any():
            raise ValueError(f"predictions.{column} values must be non-null")
        out[column] = out[column].astype(str).str.strip()
        if (out[column] == "").any():
            raise ValueError(f"predictions.{column} values must be nonblank")

    seen: set[tuple[str, str]] = set()
    for row in out.itertuples(index=False):
        pair = canonical_pair(row.img_a, row.img_b)
        if pair != (row.img_a, row.img_b):
            raise ValueError(f"predictions pair is not canonical: {(row.img_a, row.img_b)!r}")
        if pair in seen:
            raise ValueError(f"predictions contains duplicate unordered pair {pair!r}")
        unknown = sorted(set(pair) - known)
        if unknown:
            raise ValueError(f"predictions references unknown image IDs: {unknown}")
        seen.add(pair)

    try:
        flags = pd.to_numeric(out.pred_dup, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("predictions.pred_dup must contain only 0/1") from exc
    if not np.isfinite(flags.astype(float)).all() or not flags.isin([0, 1]).all():
        raise ValueError("predictions.pred_dup must contain only 0/1")
    out["pred_dup"] = flags.astype(int)
    if "score" in out.columns:
        try:
            scores = pd.to_numeric(out.score, errors="raise").astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError("predictions.score must contain finite values in [0,1]") from exc
        if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
            raise ValueError("predictions.score must contain finite values in [0,1]")
        out["score"] = scores
    return out.sort_values(["img_a", "img_b"], kind="mergesort", ignore_index=True)


def _normalize_boolean_flags(values: pd.Series, label: str) -> pd.Series:
    normalized: list[bool] = []
    for value in values:
        if isinstance(value, (bool, np.bool_)):
            normalized.append(bool(value))
        elif isinstance(value, str) and value.strip().lower() in {"true", "false"}:
            normalized.append(value.strip().lower() == "true")
        elif value in (0, 1) and not pd.isna(value):
            normalized.append(bool(value))
        else:
            raise ValueError(f"{label} must contain only boolean values")
    return pd.Series(normalized, index=values.index, dtype=bool)


def _validated_groups(
    groups: pd.DataFrame | None,
    known: set[str],
    mode: str,
) -> pd.DataFrame | None:
    if groups is None:
        if mode == "baseline":
            return None
        raise ValueError("groups is required in poc mode")
    if not isinstance(groups, pd.DataFrame):
        raise ValueError(f"groups must be a pandas DataFrame, got {type(groups).__name__}")
    missing = [column for column in GROUP_ASSIGNMENT_COLUMNS if column not in groups.columns]
    if missing:
        raise ValueError(f"groups is missing required columns {missing}; has {list(groups.columns)}")
    out = groups.copy()
    for column in GROUP_ASSIGNMENT_COLUMNS:
        if out[column].isna().any():
            raise ValueError(f"groups.{column} values must be non-null")
        out[column] = out[column].astype(str).str.strip()
        if (out[column] == "").any():
            raise ValueError(f"groups.{column} values must be nonblank")
    if out.image_id.duplicated().any():
        raise ValueError("groups must contain exactly one row per image_id")
    actual = set(out.image_id)
    if actual != known:
        raise ValueError(
            "groups must cover image_ids exactly; "
            f"missing={sorted(known - actual)[:5]}, extra={sorted(actual - known)[:5]}"
        )

    actual_sizes = out.groupby("split_group_id").size().to_dict()
    if "split_group_size" in out.columns:
        try:
            declared_sizes = pd.to_numeric(out.split_group_size, errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError("groups.split_group_size must contain positive integers") from exc
        if not np.isfinite(declared_sizes.astype(float)).all() or (
            (declared_sizes < 1) | (declared_sizes % 1 != 0)
        ).any():
            raise ValueError("groups.split_group_size must contain positive integers")
        expected_sizes = out.split_group_id.map(actual_sizes).astype(int)
        if not declared_sizes.astype(int).equals(expected_sizes):
            raise ValueError("groups.split_group_size disagrees with split_group_id membership")
        out["split_group_size"] = declared_sizes.astype(int)
    if "exceeds_max_component_size" in out.columns:
        out["exceeds_max_component_size"] = _normalize_boolean_flags(
            out.exceeds_max_component_size,
            "groups.exceeds_max_component_size",
        )
        inconsistent = out.groupby("split_group_id").exceeds_max_component_size.nunique()
        if (inconsistent > 1).any():
            raise ValueError("groups.exceeds_max_component_size must be consistent within a split group")
    return out.sort_values("image_id", kind="mergesort", ignore_index=True)


def validate_split(
    assignments: pd.DataFrame,
    image_ids: Iterable[Any],
    groups: pd.DataFrame | None,
    mode: str,
) -> pd.DataFrame:
    """Validate complete split coverage and record observable group crossings.

    ``assignments`` requires exactly one ``image_id,split`` row per input image;
    split values are ``train``, ``test``, or explicit ``drop``. PoC mode requires
    complete duplicate/split group mappings. Baseline mode may omit groups because
    its independent split intentionally ignores them. Group crossings and
    all-train/all-drop outcomes are structurally parseable, so they are recorded
    in ``result.attrs['validation']`` for risk/decision policy rather than hidden
    or repaired here.

    Returns:
        A normalized, image-ID-sorted ``SPLIT_MANIFEST_COLUMNS`` table with a
        deterministic validation summary in DataFrame attributes.

    Raises:
        ValueError: For malformed schema, incomplete/duplicate/unknown IDs,
            invalid split labels, malformed groups, or invalid diagnostic flags.
    """
    normalized_mode = _normalized_mode(mode)
    ordered_ids = _normalized_image_ids(image_ids)
    known = set(ordered_ids)
    if not isinstance(assignments, pd.DataFrame):
        raise ValueError(
            f"assignments must be a pandas DataFrame, got {type(assignments).__name__}"
        )
    missing = [column for column in SPLIT_MANIFEST_COLUMNS if column not in assignments.columns]
    if missing:
        raise ValueError(
            f"assignments is missing required columns {missing}; has {list(assignments.columns)}"
        )
    out = assignments.copy()
    for column in SPLIT_MANIFEST_COLUMNS:
        if out[column].isna().any():
            raise ValueError(f"assignments.{column} values must be non-null")
        out[column] = out[column].astype(str).str.strip()
        if (out[column] == "").any():
            raise ValueError(f"assignments.{column} values must be nonblank")
    if out.image_id.duplicated().any():
        raise ValueError("assignments must contain exactly one row per image_id")
    actual = set(out.image_id)
    if actual != known:
        raise ValueError(
            "assignments must cover image_ids exactly; "
            f"missing={sorted(known - actual)[:5]}, extra={sorted(actual - known)[:5]}"
        )
    unknown_splits = sorted(set(out.split) - _ALLOWED_SPLITS)
    if unknown_splits:
        raise ValueError(
            f"assignments has unsupported split values {unknown_splits}; "
            f"allowed={sorted(_ALLOWED_SPLITS)}"
        )
    normalized_groups = _validated_groups(groups, known, normalized_mode)
    counts = out.split.value_counts()
    validation: dict[str, Any] = {
        "mode": normalized_mode,
        "n_images": len(known),
        "n_train": int(counts.get("train", 0)),
        "n_test": int(counts.get("test", 0)),
        "n_drop": int(counts.get("drop", 0)),
        "duplicate_group_crossing_count": 0,
        "duplicate_group_crossing_ids": [],
        "split_group_crossing_count": 0,
        "split_group_crossing_ids": [],
        "group_checks_available": normalized_groups is not None,
    }
    if normalized_groups is not None:
        merged = normalized_groups[list(GROUP_ASSIGNMENT_COLUMNS)].merge(
            out[list(SPLIT_MANIFEST_COLUMNS)], on="image_id", how="left", validate="one_to_one"
        )
        for column, prefix in (
            ("duplicate_group_id", "duplicate_group"),
            ("split_group_id", "split_group"),
        ):
            split_counts = merged.groupby(column).split.nunique()
            crossing_ids = sorted(split_counts[split_counts > 1].index.astype(str).tolist())
            validation[f"{prefix}_crossing_count"] = len(crossing_ids)
            validation[f"{prefix}_crossing_ids"] = crossing_ids
    out = out[list(SPLIT_MANIFEST_COLUMNS)].sort_values(
        "image_id", kind="mergesort", ignore_index=True
    )
    out.attrs["validation"] = validation
    return out


def _conflict_details(conflicts: Any) -> tuple[int, list[Any]]:
    if conflicts is None:
        return 0, []
    if isinstance(conflicts, pd.DataFrame):
        return len(conflicts), conflicts.head(20).to_dict(orient="records")
    if isinstance(conflicts, Integral) and not isinstance(conflicts, bool):
        count = int(conflicts)
        if count < 0:
            raise ValueError("conflicts count must be nonnegative")
        return count, []
    if isinstance(conflicts, (list, tuple)):
        return len(conflicts), list(conflicts[:20])
    raise ValueError("conflicts must be None, a nonnegative count, a sequence, or DataFrame")


def _nonnegative_count(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return int(value)


def summarize_risks(
    predictions: pd.DataFrame,
    assignments: pd.DataFrame,
    groups: pd.DataFrame | None,
    image_ids: Iterable[Any],
    *,
    review_summary: Mapping[str, Any] | None = None,
    conflicts: Any = None,
    split_summary: Mapping[str, Any] | None = None,
    mode: str = "poc",
) -> dict[str, Any]:
    """Build deterministic observable diagnostics without hidden ground truth.

    Structural validation is delegated to :func:`validate_predictions` and
    :func:`validate_split` and therefore fails loudly. This function then reports
    pair/split counts, retention, component sizes, positive cross-split pairs,
    explicit conflicts, unresolved reviews, and structured risks. Baseline mode
    reports expected independent-split limitations as HOLD risks instead of
    pretending release safety.

    Returns:
        A JSON-serializable diagnostics dictionary with ``risks``, counts, and a
        provisional ``recommended_status``. ``PROVISIONAL`` means only that the
        declared observable checks passed; it is not an official leakage score.
    """
    normalized_mode = _normalized_mode(mode)
    ordered_ids = _normalized_image_ids(image_ids)
    normalized_predictions = validate_predictions(predictions, ordered_ids)
    normalized_assignments = validate_split(
        assignments, ordered_ids, groups, normalized_mode
    )
    split_validation = dict(normalized_assignments.attrs["validation"])
    known = set(ordered_ids)
    normalized_groups = _validated_groups(groups, known, normalized_mode)
    split_of = dict(zip(normalized_assignments.image_id, normalized_assignments.split))
    positive = normalized_predictions[normalized_predictions.pred_dup == 1]
    positive_cross_split = []
    positive_with_drop = []
    for row in positive.itertuples(index=False):
        split_a, split_b = split_of[row.img_a], split_of[row.img_b]
        if "drop" in {split_a, split_b}:
            positive_with_drop.append((row.img_a, row.img_b))
        elif split_a != split_b:
            positive_cross_split.append((row.img_a, row.img_b))

    counts = normalized_assignments.split.value_counts()
    n_train = int(counts.get("train", 0))
    n_test = int(counts.get("test", 0))
    n_drop = int(counts.get("drop", 0))
    retained = n_train + n_test
    n_images = len(ordered_ids)
    conflict_count, conflict_examples = _conflict_details(conflicts)
    if review_summary is not None and not isinstance(review_summary, Mapping):
        raise ValueError("review_summary must be a mapping or None")
    review = dict(review_summary or {})
    unresolved_count = _nonnegative_count(
        review.get("unresolved_count", 0), "review_summary.unresolved_count"
    )
    escalation_count = _nonnegative_count(
        review.get("escalation_count", 0), "review_summary.escalation_count"
    )

    duplicate_sizes: dict[str, int] = {}
    split_sizes: dict[str, int] = {}
    oversized_ids: list[str] = []
    if normalized_groups is not None:
        duplicate_sizes = normalized_groups.groupby("duplicate_group_id").size().to_dict()
        split_sizes = normalized_groups.groupby("split_group_id").size().to_dict()
        if "exceeds_max_component_size" in normalized_groups.columns:
            oversized_ids = sorted(
                normalized_groups.loc[
                    normalized_groups.exceeds_max_component_size, "split_group_id"
                ].unique().tolist()
            )
    if split_summary is not None and not isinstance(split_summary, Mapping):
        raise ValueError("split_summary must be a mapping or None")
    supplied_split_summary = dict(split_summary or {})
    reported_oversized = _nonnegative_count(
        supplied_split_summary.get("oversized_split_group_count", 0),
        "split_summary.oversized_split_group_count",
    )
    oversized_count = max(len(oversized_ids), reported_oversized)
    target_test_fraction = supplied_split_summary.get(
        "target_test_fraction", assignments.attrs.get("target_test_fraction")
    )
    if target_test_fraction is not None:
        if (
            isinstance(target_test_fraction, bool)
            or not isinstance(target_test_fraction, Real)
            or not math.isfinite(float(target_test_fraction))
            or not 0 <= float(target_test_fraction) <= 1
        ):
            raise ValueError("target_test_fraction must be a finite value in [0,1]")
        target_test_fraction = float(target_test_fraction)

    risks: list[dict[str, Any]] = []

    def add_risk(code: str, status: str, count: int = 1, detail: str = "", **extra: Any) -> None:
        if any(risk["code"] == code for risk in risks):
            return
        row: dict[str, Any] = {
            "code": code,
            "status": status,
            "blocking": status in {"HOLD", "REWORK"},
            "count": int(count),
        }
        if detail:
            row["detail"] = detail
        row.update(extra)
        risks.append(row)

    unsafe_status = "HOLD" if normalized_mode == "baseline" else "REWORK"
    if normalized_mode == "baseline":
        add_risk(
            "BASELINE_NOT_RELEASE_SAFE",
            "HOLD",
            detail="exact-hash/independent-split baseline is diagnostic only",
        )
    if positive_cross_split:
        add_risk(
            "POSITIVE_PAIR_CROSSES_SPLIT",
            unsafe_status,
            len(positive_cross_split),
            "positive duplicate pairs cross retained train/test assignments",
            examples=[list(pair) for pair in positive_cross_split[:20]],
        )
    for prefix in ("duplicate_group", "split_group"):
        count = int(split_validation[f"{prefix}_crossing_count"])
        if count:
            add_risk(
                f"{prefix.upper()}_CROSSES_ASSIGNMENTS",
                unsafe_status,
                count,
                "one relationship group has multiple split labels",
                group_ids=split_validation[f"{prefix}_crossing_ids"][:20],
            )
    if n_drop == n_images and n_images:
        add_risk(
            "ALL_DATA_DROPPED",
            unsafe_status,
            n_drop,
            "all images are dropped; no usable train/test data remains",
        )
    group_count_for_feasibility = len(split_sizes) if normalized_groups is not None else n_images
    two_way_possible = group_count_for_feasibility >= 2 and n_images >= 2
    usable_train_test = n_train > 0 and n_test > 0
    if not usable_train_test:
        if not two_way_possible:
            add_risk(
                "IMPOSSIBLE_TWO_WAY_GROUP_SPLIT",
                "HOLD",
                detail="fewer than two indivisible groups are available",
            )
        else:
            add_risk(
                "UNUSABLE_TWO_WAY_SPLIT",
                unsafe_status,
                detail="train or test is empty despite a feasible two-way split",
            )
    if oversized_count:
        add_risk(
            "OVERSIZED_SPLIT_COMPONENT",
            "HOLD",
            oversized_count,
            "split components exceed the frozen attention threshold",
            group_ids=oversized_ids[:20],
        )
    if conflict_count:
        add_risk(
            "GROUP_CONFLICT",
            "HOLD",
            conflict_count,
            "duplicate/scene constraints remain contradictory",
            examples=conflict_examples,
        )
    if unresolved_count:
        add_risk(
            "UNRESOLVED_REVIEW",
            "HOLD",
            unresolved_count,
            "review escalation/skip remains unresolved",
            item_ids=list(review.get("unresolved_item_ids", []))[:20],
        )
    if n_drop and n_drop < n_images:
        add_risk(
            "DATA_DROPPED",
            "WARN",
            n_drop,
            "retention loss requires explicit release-policy acceptance",
        )

    statuses = {risk["status"] for risk in risks}
    recommended = "HOLD" if "HOLD" in statuses else "REWORK" if "REWORK" in statuses else "PROVISIONAL"
    actual_test_fraction = n_test / retained if retained else 0.0
    return {
        "mode": normalized_mode,
        "n_images": n_images,
        "n_prediction_rows": len(normalized_predictions),
        "n_positive_pairs": len(positive),
        "n_train": n_train,
        "n_test": n_test,
        "n_drop": n_drop,
        "retained_count": retained,
        "retention_fraction": retained / n_images if n_images else 0.0,
        "target_test_fraction": target_test_fraction,
        "actual_test_fraction": actual_test_fraction,
        "test_fraction_deviation": (
            abs(actual_test_fraction - target_test_fraction)
            if target_test_fraction is not None
            else None
        ),
        "n_duplicate_groups": len(duplicate_sizes),
        "n_split_groups": len(split_sizes),
        "largest_duplicate_group_size": max(duplicate_sizes.values(), default=0),
        "largest_split_group_size": max(split_sizes.values(), default=0),
        "duplicate_group_size_distribution": {
            str(size): sum(group_size == size for group_size in duplicate_sizes.values())
            for size in sorted(set(duplicate_sizes.values()))
        },
        "split_group_size_distribution": {
            str(size): sum(group_size == size for group_size in split_sizes.values())
            for size in sorted(set(split_sizes.values()))
        },
        "positive_cross_split_count": len(positive_cross_split),
        "positive_with_drop_count": len(positive_with_drop),
        "duplicate_group_crossing_count": split_validation["duplicate_group_crossing_count"],
        "split_group_crossing_count": split_validation["split_group_crossing_count"],
        "oversized_split_group_count": oversized_count,
        "conflict_count": conflict_count,
        "unresolved_count": unresolved_count,
        "escalation_count": escalation_count,
        "two_way_split_possible": two_way_possible,
        "usable_train_test": usable_train_test,
        "group_checks_available": split_validation["group_checks_available"],
        "risks": risks,
        "blocking_risk_count": sum(bool(risk["blocking"]) for risk in risks),
        "recommended_status": recommended,
        "metric_claim": "observable structural diagnostics only; no hidden-GT metric",
    }


def _policy_number(
    policy: Mapping[str, Any],
    name: str,
    default: float,
    minimum: float,
    maximum: float | None = None,
) -> float:
    value = policy.get(name, default)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"policy.{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < minimum or (
        maximum is not None and result > maximum
    ):
        raise ValueError(f"policy.{name} is outside the allowed range")
    return result


def make_release_decision(
    diagnostics: Mapping[str, Any],
    owner: str,
    policy: Mapping[str, Any] | None,
) -> Decision:
    """Convert validated observable diagnostics into a T05 release decision.

    Policy recognizes ``decision_max_unresolved`` (default 0),
    ``max_drop_fraction`` (default 0), optional ``evidence`` paths, and ``scope``.
    Baseline mode always HOLDs. Correctable pair/group/split violations REWORK;
    unresolved/conflicting/oversized/impossible evidence HOLDs; otherwise GO.

    GO means the declared evidence-based structural policy passed. It never
    claims verified zero true leakage without instructor ground truth.

    Raises:
        ValueError: For malformed diagnostics, owner, or policy values.
    """
    if not isinstance(diagnostics, Mapping):
        raise ValueError("diagnostics must be a mapping")
    if owner is None or not str(owner).strip():
        raise ValueError("owner must be nonblank")
    if policy is None:
        cfg: dict[str, Any] = {}
    elif isinstance(policy, Mapping):
        cfg = dict(policy)
    else:
        raise ValueError(f"policy must be a mapping, got {type(policy).__name__}")
    required = {
        "mode",
        "n_images",
        "n_positive_pairs",
        "n_train",
        "n_test",
        "n_drop",
        "retention_fraction",
        "actual_test_fraction",
        "positive_cross_split_count",
        "conflict_count",
        "unresolved_count",
        "blocking_risk_count",
        "risks",
        "recommended_status",
    }
    missing = sorted(required - set(diagnostics))
    if missing:
        raise ValueError(f"diagnostics is missing required fields {missing}")
    mode = _normalized_mode(str(diagnostics["mode"]))
    unresolved_limit_raw = cfg.get("decision_max_unresolved", 0)
    if (
        isinstance(unresolved_limit_raw, bool)
        or not isinstance(unresolved_limit_raw, Integral)
        or int(unresolved_limit_raw) < 0
    ):
        raise ValueError("policy.decision_max_unresolved must be a nonnegative integer")
    unresolved_limit = int(unresolved_limit_raw)
    max_drop_fraction = _policy_number(cfg, "max_drop_fraction", 0.0, 0.0, 1.0)
    risks = diagnostics["risks"]
    if not isinstance(risks, list) or any(not isinstance(risk, Mapping) for risk in risks):
        raise ValueError("diagnostics.risks must be a list of mappings")
    normalized_risks: list[Mapping[str, Any]] = []
    for index, risk in enumerate(risks):
        code = risk.get("code")
        status_value = risk.get("status")
        if code is None or not str(code).strip():
            raise ValueError(f"diagnostics.risks[{index}].code must be nonblank")
        if status_value not in {"WARN", "HOLD", "REWORK"}:
            raise ValueError(
                f"diagnostics.risks[{index}].status must be WARN, HOLD, or REWORK"
            )
        _nonnegative_count(risk.get("count", 0), f"diagnostics.risks[{index}].count")
        normalized_risks.append(risk)
    counts = {
        name: _nonnegative_count(diagnostics[name], f"diagnostics.{name}")
        for name in (
            "n_images",
            "n_positive_pairs",
            "n_train",
            "n_test",
            "n_drop",
            "positive_cross_split_count",
            "conflict_count",
            "unresolved_count",
            "blocking_risk_count",
        )
    }
    n_images = counts["n_images"]
    n_drop = counts["n_drop"]
    unresolved = counts["unresolved_count"]
    conflicts = counts["conflict_count"]
    if counts["n_train"] + counts["n_test"] + n_drop != n_images:
        raise ValueError("diagnostics train/test/drop counts must sum to n_images")
    for name in ("retention_fraction", "actual_test_fraction"):
        value = diagnostics[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            or not 0 <= float(value) <= 1
        ):
            raise ValueError(f"diagnostics.{name} must be a finite value in [0,1]")
    expected_retention = (counts["n_train"] + counts["n_test"]) / n_images if n_images else 0.0
    retained = counts["n_train"] + counts["n_test"]
    expected_test_fraction = counts["n_test"] / retained if retained else 0.0
    if not math.isclose(float(diagnostics["retention_fraction"]), expected_retention, abs_tol=1e-12):
        raise ValueError("diagnostics.retention_fraction disagrees with split counts")
    if not math.isclose(
        float(diagnostics["actual_test_fraction"]), expected_test_fraction, abs_tol=1e-12
    ):
        raise ValueError("diagnostics.actual_test_fraction disagrees with split counts")
    blocking_count = sum(risk["status"] in {"HOLD", "REWORK"} for risk in normalized_risks)
    if counts["blocking_risk_count"] != blocking_count:
        raise ValueError("diagnostics.blocking_risk_count disagrees with risks")
    risk_statuses = {risk["status"] for risk in normalized_risks}
    expected_recommendation = (
        "HOLD"
        if "HOLD" in risk_statuses
        else "REWORK"
        if "REWORK" in risk_statuses
        else "PROVISIONAL"
    )
    if diagnostics["recommended_status"] != expected_recommendation:
        raise ValueError("diagnostics.recommended_status disagrees with risks")
    drop_fraction = n_drop / n_images if n_images else 0.0
    evidence = cfg.get(
        "evidence",
        ["diagnostics.json", "suspect_pairs.csv", "split_manifest.csv"],
    )
    if isinstance(evidence, (str, bytes)) or not isinstance(evidence, Iterable):
        raise ValueError("policy.evidence must be an iterable of paths")
    evidence_paths = [str(value).strip() for value in evidence]
    if not evidence_paths or any(not value for value in evidence_paths):
        raise ValueError("policy.evidence must contain at least one nonblank path")

    status = "GO"
    reason_code = "POLICY_CHECKS_PASSED"
    reason = (
        "Declared T05 structural checks passed with no blocking observable risk; "
        "this does not claim verified zero leakage without ground truth."
    )
    if mode == "baseline":
        status = "HOLD"
        reason_code = "BASELINE_ONLY_NOT_RELEASE_SAFE"
        reason = "Exact-hash independent-split baseline is diagnostic and not release-safe."
    else:
        rework_risks = [risk for risk in normalized_risks if risk.get("status") == "REWORK"]
        hold_risks = [
            risk
            for risk in normalized_risks
            if risk.get("status") == "HOLD"
            and not (
                risk.get("code") == "UNRESOLVED_REVIEW"
                and unresolved <= unresolved_limit
            )
        ]
        if unresolved > unresolved_limit:
            status = "HOLD"
            reason_code = "UNRESOLVED_REVIEW"
            reason = f"Unresolved reviews {unresolved} exceed policy limit {unresolved_limit}."
        elif conflicts > 0:
            status = "HOLD"
            reason_code = "GROUP_CONFLICT"
            reason = f"{conflicts} grouping/relation conflicts remain unresolved."
        elif hold_risks:
            status = "HOLD"
            reason_code = str(hold_risks[0].get("code") or "OBSERVABLE_RISK_REMAINS")
            reason = str(
                hold_risks[0].get("detail")
                or "Observable material risk remains unresolved."
            )
        elif rework_risks:
            status = "REWORK"
            reason_code = str(rework_risks[0].get("code") or "STRUCTURAL_REWORK_REQUIRED")
            reason = str(
                rework_risks[0].get("detail")
                or "A valid run exposed a pair/group/split problem requiring correction."
            )
        elif drop_fraction > max_drop_fraction:
            status = "REWORK"
            reason_code = "RETENTION_BELOW_POLICY"
            reason = (
                f"Dropped fraction {drop_fraction:.6f} exceeds policy maximum "
                f"{max_drop_fraction:.6f}."
            )

    metrics = {
        key: diagnostics.get(key)
        for key in (
            "n_images",
            "n_positive_pairs",
            "n_train",
            "n_test",
            "n_drop",
            "retention_fraction",
            "actual_test_fraction",
            "positive_cross_split_count",
            "conflict_count",
            "unresolved_count",
            "blocking_risk_count",
        )
    }
    return Decision(
        status=status,
        reason_code=reason_code,
        reason=reason,
        owner=str(owner).strip(),
        scope=str(cfg.get("scope", "t05")),
        evidence=evidence_paths,
        metrics=metrics,
    )
