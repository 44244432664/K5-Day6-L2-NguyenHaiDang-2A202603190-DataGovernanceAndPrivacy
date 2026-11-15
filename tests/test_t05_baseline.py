"""Task 5 tests for the runnable T05 exact-hash/random-split baseline."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src import main as runner
from src.t05.baseline import assign_random_splits, predict_exact_pairs
from src.t05.contracts import PAIR_EVIDENCE_COLUMNS, PREDICTION_COLUMNS, SPLIT_MANIFEST_COLUMNS


def _evidence(rows) -> pd.DataFrame:
    records = []
    for index, (img_a, img_b, exact) in enumerate(rows):
        records.append(
            {
                "pair_id": f"pair-{index}",
                "img_a": img_a,
                "img_b": img_b,
                "exact_match": exact,
                "phash_distance": 0,
                "dhash_distance": 0,
                "histogram_distance": 0.0,
                "thumbnail_distance": 0.0,
                "same_camera": True,
                "time_delta_seconds": 0.0,
                "duplicate_score": np.nan,
                "review_priority": np.nan,
                "reason_codes": "",
            }
        )
    return pd.DataFrame(records, columns=PAIR_EVIDENCE_COLUMNS)


def test_predict_exact_pairs_keeps_explicit_negatives():
    result = predict_exact_pairs(_evidence([("a", "b", True), ("a", "c", False)]))
    assert tuple(result.columns) == PREDICTION_COLUMNS
    assert result.to_dict(orient="records") == [
        {"img_a": "a", "img_b": "b", "score": 1.0, "pred_dup": 1},
        {"img_a": "a", "img_b": "c", "score": 0.0, "pred_dup": 0},
    ]


def test_exact_group_of_three_emits_all_three_positive_pairs():
    result = predict_exact_pairs(
        _evidence([("a", "b", True), ("a", "c", True), ("b", "c", True)])
    )
    assert len(result) == 3
    assert result.pred_dup.tolist() == [1, 1, 1]


def test_predict_exact_pairs_rejects_noncanonical_or_duplicate_pairs():
    with pytest.raises(ValueError, match="not canonical"):
        predict_exact_pairs(_evidence([("b", "a", True)]))
    duplicate = _evidence([("a", "b", True), ("a", "b", False)])
    with pytest.raises(ValueError, match="duplicate unordered pair"):
        predict_exact_pairs(duplicate.assign(pair_id=["p1", "p2"]))


def test_random_split_is_seeded_order_independent_and_nontrivial():
    images = pd.DataFrame({"image_id": [f"im_{i:03d}" for i in reversed(range(10))]})
    first = assign_random_splits(images, 0.3, np.random.default_rng(42))
    second = assign_random_splits(
        images.sample(frac=1, random_state=7), 0.3, np.random.default_rng(42)
    )
    assert tuple(first.columns) == SPLIT_MANIFEST_COLUMNS
    pd.testing.assert_frame_equal(first, second)
    assert first.image_id.is_monotonic_increasing
    assert (first.split == "test").sum() == 3
    assert (first.split == "train").sum() == 7


def test_random_split_tiny_dataset_policy():
    empty = assign_random_splits(
        pd.DataFrame(columns=["image_id"]), 0.3, np.random.default_rng(42)
    )
    one = assign_random_splits(
        pd.DataFrame({"image_id": ["only"]}), 0.3, np.random.default_rng(42)
    )
    two = assign_random_splits(
        pd.DataFrame({"image_id": ["a", "b"]}), 0.3, np.random.default_rng(42)
    )
    assert empty.empty and tuple(empty.columns) == SPLIT_MANIFEST_COLUMNS
    assert one.to_dict(orient="records") == [{"image_id": "only", "split": "train"}]
    assert sorted(two.split.tolist()) == ["test", "train"]


@pytest.mark.parametrize("fraction", [0, 1, -0.1, float("nan"), True])
def test_random_split_rejects_invalid_fraction(fraction):
    with pytest.raises(ValueError, match="test_fraction"):
        assign_random_splits(
            pd.DataFrame({"image_id": ["a", "b"]}),
            fraction,
            np.random.default_rng(42),
        )


def _run_baseline(output) -> int:
    return runner.main(
        [
            "--track",
            "t05",
            "--pipeline",
            "src.t05.baseline",
            "--output",
            str(output),
            "--run-name",
            "baseline",
            "--offline",
            "--seed",
            "42",
        ]
    )


def test_baseline_cli_outputs_are_complete_held_and_reproducible(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    assert _run_baseline(first) == 0
    assert _run_baseline(second) == 0

    expected = {
        "suspect_pairs.csv",
        "automatic_pairs.csv",
        "pair_evidence.csv",
        "split_manifest.csv",
        "diagnostics.json",
        "decision.json",
        "decision_log.csv",
        "run_info.json",
    }
    assert expected <= {path.name for path in first.iterdir()}

    predictions = pd.read_csv(first / "suspect_pairs.csv")
    split = pd.read_csv(first / "split_manifest.csv")
    diagnostics = json.loads((first / "diagnostics.json").read_text(encoding="utf-8"))
    decision = json.loads((first / "decision.json").read_text(encoding="utf-8"))
    run_info = json.loads((first / "run_info.json").read_text(encoding="utf-8"))

    assert len(predictions) == 7140
    assert int(predictions.pred_dup.sum()) == 20
    assert len(split) == 120 and (split.split == "test").sum() == 36
    assert set(split.split) == {"train", "test"}
    assert diagnostics["release_safe"] is False
    assert "exact_positive_cross_split_pairs" in diagnostics
    assert decision["status"] == "HOLD"
    assert decision["reason_code"] == "BASELINE_ONLY_NOT_GROUP_SAFE"
    assert run_info["status"] == 0 and run_info["stats"]["n_items"] == 120

    assert (first / "suspect_pairs.csv").read_bytes() == (second / "suspect_pairs.csv").read_bytes()
    assert (first / "split_manifest.csv").read_bytes() == (second / "split_manifest.csv").read_bytes()
