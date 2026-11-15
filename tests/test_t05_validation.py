"""Task 11 tests for T05 artifact validation and release decisions."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from src.common.decision import Decision
from src.t05 import (
    make_release_decision,
    summarize_risks,
    validate_predictions,
    validate_split,
)


IMAGE_IDS = ["a", "b", "c", "d"]


def _predictions(*rows):
    return pd.DataFrame(rows, columns=("img_a", "img_b", "score", "pred_dup"))


def _safe_predictions():
    return _predictions(
        ("a", "b", 0.95, 1),
        ("a", "c", 0.10, 0),
        ("a", "d", 0.10, 0),
        ("b", "c", 0.10, 0),
        ("b", "d", 0.10, 0),
        ("c", "d", 0.20, 0),
    )


def _groups(*, oversized=False):
    return pd.DataFrame(
        [
            ("a", "dup:ab", "split:ab", 2, oversized),
            ("b", "dup:ab", "split:ab", 2, oversized),
            ("c", "dup:c", "split:c", 1, False),
            ("d", "dup:d", "split:d", 1, False),
        ],
        columns=(
            "image_id",
            "duplicate_group_id",
            "split_group_id",
            "split_group_size",
            "exceeds_max_component_size",
        ),
    )


def _assignments(a="train", b="train", c="test", d="test"):
    return pd.DataFrame(
        [("a", a), ("b", b), ("c", c), ("d", d)],
        columns=("image_id", "split"),
    )


def _diagnostics(**kwargs):
    return summarize_risks(
        kwargs.pop("predictions", _safe_predictions()),
        kwargs.pop("assignments", _assignments()),
        kwargs.pop("groups", _groups()),
        kwargs.pop("image_ids", IMAGE_IDS),
        **kwargs,
    )


def test_validate_predictions_normalizes_sorts_and_does_not_mutate_input():
    predictions = _safe_predictions().iloc[::-1].reset_index(drop=True)
    predictions["source"] = "automatic"
    original = predictions.copy(deep=True)

    validated = validate_predictions(predictions, IMAGE_IDS)

    pd.testing.assert_frame_equal(predictions, original)
    assert validated[["img_a", "img_b"]].values.tolist() == sorted(
        validated[["img_a", "img_b"]].values.tolist()
    )
    assert validated.pred_dup.dtype.kind in "iu"
    assert validated.score.dtype.kind == "f"
    assert set(validated.source) == {"automatic"}


@pytest.mark.parametrize(
    ("predictions", "message"),
    [
        (pd.DataFrame({"img_a": ["a"], "img_b": ["b"]}), "missing required"),
        (_predictions(("b", "a", 0.9, 1)), "not canonical"),
        (_predictions(("a", "a", 0.9, 1)), "self-pair"),
        (_predictions(("a", "b", 0.9, 1), ("a", "b", 0.8, 1)), "duplicate unordered"),
        (_predictions(("a", "unknown", 0.9, 1)), "unknown image"),
        (_predictions(("a", "b", 0.9, 2)), "only 0/1"),
        (_predictions(("a", "b", 0.9, 0.5)), "only 0/1"),
        (_predictions(("a", "b", np.nan, 1)), "finite values"),
        (_predictions(("a", "b", 1.1, 1)), "finite values"),
    ],
)
def test_validate_predictions_rejects_malformed_artifacts(predictions, message):
    with pytest.raises(ValueError, match=message):
        validate_predictions(predictions, IMAGE_IDS)


def test_validate_predictions_accepts_empty_table_with_contract_headers():
    empty = pd.DataFrame(columns=("img_a", "img_b", "pred_dup"))
    validated = validate_predictions(empty, IMAGE_IDS)
    assert validated.empty
    assert tuple(validated.columns) == ("img_a", "img_b", "pred_dup")


@pytest.mark.parametrize("bad_id", [None, np.nan, pd.NA, ""])
def test_image_id_universe_rejects_null_or_blank_values(bad_id):
    with pytest.raises(ValueError, match="null|blank"):
        validate_predictions(_safe_predictions(), ["a", "b", "c", bad_id])


def test_validate_split_records_crossings_without_silently_repairing_them():
    assignments = _assignments(b="test")
    original = assignments.copy(deep=True)

    validated = validate_split(assignments, IMAGE_IDS, _groups(), "poc")

    pd.testing.assert_frame_equal(assignments, original)
    assert validated.attrs["validation"]["duplicate_group_crossing_count"] == 1
    assert validated.attrs["validation"]["split_group_crossing_count"] == 1
    assert validated.attrs["validation"]["duplicate_group_crossing_ids"] == ["dup:ab"]


@pytest.mark.parametrize(
    ("assignments", "message"),
    [
        (pd.DataFrame({"image_id": IMAGE_IDS}), "missing required"),
        (_assignments().iloc[:-1], "cover image_ids exactly"),
        (
            pd.concat([_assignments(), pd.DataFrame([("a", "train")], columns=("image_id", "split"))]),
            "exactly one row",
        ),
        (_assignments(d="validation"), "unsupported split"),
        (_assignments(d=""), "nonblank"),
    ],
)
def test_validate_split_rejects_malformed_manifest(assignments, message):
    with pytest.raises(ValueError, match=message):
        validate_split(assignments, IMAGE_IDS, _groups(), "poc")


def test_validate_split_requires_complete_valid_groups_only_for_poc():
    with pytest.raises(ValueError, match="groups is required"):
        validate_split(_assignments(), IMAGE_IDS, None, "poc")

    baseline = validate_split(_assignments(c="train", d="train"), IMAGE_IDS, None, "baseline")
    assert baseline.attrs["validation"]["group_checks_available"] is False

    incomplete = _groups().iloc[:-1]
    with pytest.raises(ValueError, match="cover image_ids exactly"):
        validate_split(_assignments(), IMAGE_IDS, incomplete, "poc")


def test_validate_split_rejects_inconsistent_group_diagnostics():
    wrong_size = _groups()
    wrong_size.loc[0, "split_group_size"] = 99
    with pytest.raises(ValueError, match="disagrees"):
        validate_split(_assignments(), IMAGE_IDS, wrong_size, "poc")

    inconsistent_flag = _groups()
    inconsistent_flag.loc[0, "exceeds_max_component_size"] = True
    with pytest.raises(ValueError, match="consistent"):
        validate_split(_assignments(), IMAGE_IDS, inconsistent_flag, "poc")


def test_safe_poc_diagnostics_are_observable_and_provisional():
    diagnostics = _diagnostics()

    assert diagnostics["recommended_status"] == "PROVISIONAL"
    assert diagnostics["blocking_risk_count"] == 0
    assert diagnostics["n_images"] == 4
    assert diagnostics["n_positive_pairs"] == 1
    assert diagnostics["n_train"] == 2
    assert diagnostics["n_test"] == 2
    assert diagnostics["n_drop"] == 0
    assert diagnostics["retention_fraction"] == 1.0
    assert diagnostics["positive_cross_split_count"] == 0
    assert "no hidden-GT metric" in diagnostics["metric_claim"]


def test_positive_or_group_crossing_requires_rework():
    diagnostics = _diagnostics(assignments=_assignments(b="test"))
    codes = {risk["code"] for risk in diagnostics["risks"]}

    assert diagnostics["recommended_status"] == "REWORK"
    assert diagnostics["positive_cross_split_count"] == 1
    assert "POSITIVE_PAIR_CROSSES_SPLIT" in codes
    assert "DUPLICATE_GROUP_CROSSES_ASSIGNMENTS" in codes
    assert "SPLIT_GROUP_CROSSES_ASSIGNMENTS" in codes
    assert make_release_decision(diagnostics, "data owner", {}).canonical == "REWORK"


@pytest.mark.parametrize(
    ("assignments", "expected_code"),
    [
        (_assignments("train", "train", "train", "train"), "UNUSABLE_TWO_WAY_SPLIT"),
        (_assignments("drop", "drop", "drop", "drop"), "ALL_DATA_DROPPED"),
    ],
)
def test_all_train_and_all_drop_are_not_successful_poc_splits(assignments, expected_code):
    diagnostics = _diagnostics(assignments=assignments)
    assert diagnostics["recommended_status"] == "REWORK"
    assert expected_code in {risk["code"] for risk in diagnostics["risks"]}


def test_single_indivisible_group_holds_as_impossible_not_successful():
    groups = _groups()
    groups["split_group_id"] = "split:all"
    groups["split_group_size"] = 4
    diagnostics = _diagnostics(
        assignments=_assignments("train", "train", "train", "train"),
        groups=groups,
    )

    assert diagnostics["two_way_split_possible"] is False
    assert diagnostics["recommended_status"] == "HOLD"
    assert "IMPOSSIBLE_TWO_WAY_GROUP_SPLIT" in {
        risk["code"] for risk in diagnostics["risks"]
    }


@pytest.mark.parametrize(
    ("kwargs", "expected_code"),
    [
        ({"review_summary": {"unresolved_count": 1}}, "UNRESOLVED_REVIEW"),
        ({"conflicts": [{"img_a": "a", "img_b": "b"}]}, "GROUP_CONFLICT"),
        ({"groups": _groups(oversized=True)}, "OVERSIZED_SPLIT_COMPONENT"),
    ],
)
def test_unresolved_conflict_and_oversized_component_hold(kwargs, expected_code):
    diagnostics = _diagnostics(**kwargs)
    decision = make_release_decision(diagnostics, "data owner", {})
    assert diagnostics["recommended_status"] == "HOLD"
    assert expected_code in {risk["code"] for risk in diagnostics["risks"]}
    assert decision.canonical == "HOLD"


def test_hold_risk_takes_precedence_over_simultaneous_rework_risk():
    diagnostics = _diagnostics(
        assignments=_assignments(b="test"),
        conflicts=[{"detail": "reviewed cannot-link conflicts with required path"}],
    )
    decision = make_release_decision(diagnostics, "data owner", {})

    assert diagnostics["recommended_status"] == "HOLD"
    assert decision.canonical == "HOLD"
    assert decision.reason_code == "GROUP_CONFLICT"


def test_baseline_expected_violations_are_reported_not_schema_failures():
    diagnostics = summarize_risks(
        _safe_predictions(),
        _assignments("train", "train", "train", "train"),
        None,
        IMAGE_IDS,
        mode="baseline",
    )
    decision = make_release_decision(diagnostics, "data owner", {})

    assert diagnostics["mode"] == "baseline"
    assert diagnostics["group_checks_available"] is False
    assert diagnostics["recommended_status"] == "HOLD"
    assert "BASELINE_NOT_RELEASE_SAFE" in {risk["code"] for risk in diagnostics["risks"]}
    assert decision.canonical == "HOLD"
    assert decision.reason_code == "BASELINE_ONLY_NOT_RELEASE_SAFE"


def test_drop_is_warn_diagnostic_but_default_policy_requires_rework():
    singleton_groups = _groups()
    singleton_groups.loc[singleton_groups.image_id == "b", "duplicate_group_id"] = "dup:b"
    singleton_groups.loc[singleton_groups.image_id == "b", "split_group_id"] = "split:b"
    singleton_groups["split_group_size"] = 1
    predictions = _safe_predictions()
    predictions.loc[(predictions.img_a == "a") & (predictions.img_b == "b"), "pred_dup"] = 0
    diagnostics = _diagnostics(
        predictions=predictions,
        assignments=_assignments(b="drop"),
        groups=singleton_groups,
    )

    assert diagnostics["recommended_status"] == "PROVISIONAL"
    assert "DATA_DROPPED" in {risk["code"] for risk in diagnostics["risks"]}
    assert make_release_decision(diagnostics, "owner", {}).canonical == "REWORK"
    assert make_release_decision(
        diagnostics, "owner", {"max_drop_fraction": 0.25}
    ).canonical == "GO"


def test_make_release_decision_returns_shared_decision_and_scoped_evidence():
    diagnostics = _diagnostics()
    decision = make_release_decision(
        diagnostics,
        "Data Steward",
        {
            "scope": "fixture-v1",
            "evidence": ["diagnostics.json", "split_manifest.csv"],
        },
    )

    assert isinstance(decision, Decision)
    assert decision.canonical == "GO"
    assert decision.reason_code == "POLICY_CHECKS_PASSED"
    assert decision.owner == "Data Steward"
    assert decision.scope == "fixture-v1"
    assert decision.evidence == ["diagnostics.json", "split_manifest.csv"]
    assert "does not claim verified zero leakage" in decision.reason


def test_make_release_decision_rejects_incomplete_or_inconsistent_diagnostics():
    diagnostics = _diagnostics()
    incomplete = dict(diagnostics)
    incomplete.pop("n_images")
    with pytest.raises(ValueError, match="missing required fields"):
        make_release_decision(incomplete, "owner", {})

    inconsistent = copy.deepcopy(diagnostics)
    inconsistent["blocking_risk_count"] = 1
    with pytest.raises(ValueError, match="disagrees with risks"):
        make_release_decision(inconsistent, "owner", {})


@pytest.mark.parametrize(
    ("owner", "policy", "message"),
    [
        ("", {}, "owner"),
        ("owner", {"decision_max_unresolved": -1}, "nonnegative integer"),
        ("owner", {"decision_max_unresolved": 0.5}, "nonnegative integer"),
        ("owner", {"max_drop_fraction": 2}, "outside"),
        ("owner", {"evidence": "diagnostics.json"}, "iterable of paths"),
    ],
)
def test_make_release_decision_rejects_invalid_policy(owner, policy, message):
    diagnostics = _diagnostics()
    original = copy.deepcopy(diagnostics)
    with pytest.raises(ValueError, match=message):
        make_release_decision(diagnostics, owner, policy)
    assert diagnostics == original
