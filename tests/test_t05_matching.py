"""Task 6 tests for interpretable near-duplicate scoring/classification."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PIL import Image, ImageEnhance

from src.common.io import load_config
from src.t05.contracts import (
    CLASSIFIED_PAIR_COLUMNS,
    DUPLICATE_GROUP_COLUMNS,
    GROUP_CONFLICT_COLUMNS,
    GROUP_PAIR_COLUMNS,
    PAIR_CONSTRAINT_COLUMNS,
    PAIR_EVIDENCE_COLUMNS,
)
from src.t05.data import IMAGE_TABLE_COLUMNS, load_inputs
from src.t05.features import build_pair_evidence, extract_features
from src.t05.matching import (
    build_duplicate_groups,
    classify_pairs,
    expand_group_pairs,
    find_group_conflicts,
    score_duplicate_evidence,
)


def _row(
    pair_id: str,
    img_a: str,
    img_b: str,
    *,
    exact=False,
    phash=32,
    dhash=32,
    histogram=1.0,
    thumbnail=0.5,
    same_camera=False,
    seconds=1000,
) -> dict:
    return {
        "pair_id": pair_id,
        "img_a": img_a,
        "img_b": img_b,
        "exact_match": exact,
        "phash_distance": phash,
        "dhash_distance": dhash,
        "histogram_distance": histogram,
        "thumbnail_distance": thumbnail,
        "same_camera": same_camera,
        "time_delta_seconds": seconds,
        "duplicate_score": np.nan,
        "review_priority": np.nan,
        "reason_codes": "",
    }


def _frame(*rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=PAIR_EVIDENCE_COLUMNS)


def test_exact_and_strong_visual_pairs_are_positive():
    evidence = _frame(
        _row("p1", "a", "b", exact=True),
        _row("p2", "a", "c", phash=0, dhash=0, histogram=0, thumbnail=0),
    )
    scored = score_duplicate_evidence(evidence)
    classified = classify_pairs(scored)

    assert scored.duplicate_score.tolist() == [1.0, 1.0]
    assert classified.classification.tolist() == ["positive", "positive"]
    assert classified.pred_dup.tolist() == [1, 1]
    assert classified.review_priority.iloc[0] == 0.0
    assert "exact_sha256" in classified.reason_codes.iloc[0]
    assert "auto_positive" in classified.reason_codes.iloc[1]


def test_configured_distance_boundary_maps_to_threshold_and_is_reviewable():
    evidence = _frame(
        _row("p1", "a", "b", phash=8, dhash=8, histogram=0.08, thumbnail=0.10)
    )
    classified = classify_pairs(score_duplicate_evidence(evidence))
    assert classified.duplicate_score.iloc[0] == pytest.approx(0.75)
    assert classified.classification.iloc[0] == "positive"
    assert classified.pred_dup.iloc[0] == 1
    assert classified.review_priority.iloc[0] == pytest.approx(1.0)


def test_histogram_alone_cannot_create_positive_prediction():
    evidence = _frame(
        _row("p1", "a", "b", phash=64, dhash=64, histogram=0, thumbnail=1.0)
    )
    classified = classify_pairs(score_duplicate_evidence(evidence))
    assert classified.duplicate_score.iloc[0] == pytest.approx(0.10)
    assert classified.classification.iloc[0] == "negative"
    assert classified.pred_dup.iloc[0] == 0
    assert "visual_signals:0" in classified.reason_codes.iloc[0]


def test_high_score_with_insufficient_visual_agreement_is_ambiguous():
    # Only pHash is inside its hard limit, despite a high continuous score.
    evidence = _frame(
        _row("p1", "a", "b", phash=0, dhash=9, histogram=0, thumbnail=0.11)
    )
    classified = classify_pairs(score_duplicate_evidence(evidence))
    assert classified.duplicate_score.iloc[0] > 0.75
    assert classified.classification.iloc[0] == "ambiguous"
    assert classified.pred_dup.iloc[0] == 0
    assert classified.review_priority.iloc[0] == 1.0
    assert "needs_review" in classified.reason_codes.iloc[0]


def test_score_does_not_use_camera_or_timestamp_context():
    rows = [
        _row("p1", "a", "b", phash=4, dhash=4, histogram=0.04, thumbnail=0.05, same_camera=True, seconds=0.1),
        _row("p2", "c", "d", phash=4, dhash=4, histogram=0.04, thumbnail=0.05, same_camera=False, seconds=999999),
    ]
    scored = score_duplicate_evidence(_frame(*rows))
    assert scored.duplicate_score.iloc[0] == scored.duplicate_score.iloc[1]


def test_scoring_is_reproducible_and_does_not_mutate_input():
    evidence = _frame(_row("p1", "a", "b", phash=10, dhash=10, histogram=0.1, thumbnail=0.125))
    original = evidence.copy(deep=True)
    first = classify_pairs(score_duplicate_evidence(evidence))
    second = classify_pairs(score_duplicate_evidence(evidence))
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(evidence, original)
    assert tuple(first.columns) == CLASSIFIED_PAIR_COLUMNS
    assert first.classification.iloc[0] == "ambiguous"


def _write_development_images(root):
    # Synthetic, separately labelled development cases; no fixture GT is used.
    y, x = np.mgrid[:64, :96]
    base = np.zeros((64, 96, 3), dtype=np.uint8)
    base[..., 0] = (x * 2 + y) % 256
    base[..., 1] = (y * 3) % 256
    base[..., 2] = ((x // 12 + y // 12) % 2) * 180 + 30
    original = Image.fromarray(base)
    paths = root / "images"
    paths.mkdir(parents=True)
    original.save(paths / "original.jpg", quality=95)
    original.save(paths / "jpeg.jpg", quality=35)
    ImageEnhance.Brightness(original).enhance(1.12).save(paths / "bright.jpg", quality=95)

    # Same red/blue pixel counts but different spatial layout: a color-only trap.
    vertical = np.zeros_like(base)
    vertical[:, :48] = (220, 20, 20)
    vertical[:, 48:] = (20, 20, 220)
    colors = np.array([(220, 20, 20)] * (64 * 48) + [(20, 20, 220)] * (64 * 48), dtype=np.uint8)
    np.random.default_rng(42).shuffle(colors)
    shuffled = colors.reshape(64, 96, 3)
    Image.fromarray(vertical).save(paths / "vertical.png")
    Image.fromarray(shuffled).save(paths / "shuffled.png")

    ids = ["original", "jpeg", "bright", "vertical", "shuffled"]
    return pd.DataFrame(
        {
            "image_id": ids,
            "path": [f"images/{image_id}.jpg" if image_id in {"original", "jpeg", "bright"} else f"images/{image_id}.png" for image_id in ids],
            "camera_id": ["dev_cam"] * len(ids),
            "captured_at": pd.to_datetime(
                [f"2026-01-01T00:00:0{i}Z" for i in range(len(ids))], utc=True
            ),
        },
        columns=IMAGE_TABLE_COLUMNS,
    )


def test_synthetic_jpeg_and_brightness_cases_and_color_layout_trap(tmp_path):
    images = _write_development_images(tmp_path)
    features = extract_features(images, tmp_path)
    pairs = build_pair_evidence(images, features)
    classified = classify_pairs(score_duplicate_evidence(pairs)).set_index(["img_a", "img_b"])

    for pair in (("jpeg", "original"), ("bright", "original")):
        assert classified.loc[pair, "classification"] == "positive"
        assert classified.loc[pair, "pred_dup"] == 1
    trap = classified.loc[("shuffled", "vertical")]
    assert trap.pred_dup == 0


def test_invalid_or_nonfinite_evidence_is_rejected():
    bad = _frame(_row("p1", "a", "b", phash=np.nan))
    with pytest.raises(ValueError, match="finite nonnegative"):
        score_duplicate_evidence(bad)
    unscored = _frame(_row("p1", "a", "b"))
    with pytest.raises(ValueError, match="duplicate_score"):
        classify_pairs(unscored)


def test_empty_pair_table_keeps_scored_and_classified_headers():
    empty = pd.DataFrame(columns=PAIR_EVIDENCE_COLUMNS)
    scored = score_duplicate_evidence(empty)
    classified = classify_pairs(scored)
    assert scored.empty and tuple(scored.columns) == PAIR_EVIDENCE_COLUMNS
    assert classified.empty and tuple(classified.columns) == CLASSIFIED_PAIR_COLUMNS


def test_repository_fixture_classification_is_finite_and_well_formed():
    loaded = load_inputs("fixtures/t05_near_duplicate", strict_provenance=True)
    config = load_config("configs/lab.yaml")
    features = extract_features(
        loaded["images"], "fixtures/t05_near_duplicate", config["params"]
    )
    evidence = build_pair_evidence(loaded["images"], features)
    classified = classify_pairs(score_duplicate_evidence(evidence, config["params"]), config["params"])

    assert len(classified) == 7140
    assert np.isfinite(classified.duplicate_score).all()
    assert np.isfinite(classified.review_priority).all()
    assert set(classified.classification) <= {"positive", "negative", "ambiguous"}
    exact = classified[classified.exact_match]
    assert len(exact) == 20
    assert (exact.classification == "positive").all()
    assert (exact.pred_dup == 1).all()

    groups = build_duplicate_groups(loaded["images"], classified)
    expanded = expand_group_pairs(groups)
    assert len(groups) == 120 and groups.image_id.is_unique
    positive_pairs = set(
        classified.loc[classified.classification == "positive", ["img_a", "img_b"]].itertuples(
            index=False, name=None
        )
    )
    expanded_pairs = set(expanded[["img_a", "img_b"]].itertuples(index=False, name=None))
    exact_pairs = set(exact[["img_a", "img_b"]].itertuples(index=False, name=None))
    assert expanded_pairs <= positive_pairs
    assert exact_pairs <= expanded_pairs


def _classified_specs(specs):
    rows = []
    for index, spec in enumerate(specs):
        img_a, img_b, state = spec[:3]
        if state == "exact":
            rows.append(_row(f"g{index}", img_a, img_b, exact=True))
        elif state == "positive":
            rows.append(_row(f"g{index}", img_a, img_b, phash=0, dhash=0, histogram=0, thumbnail=0))
        elif state == "ambiguous":
            rows.append(_row(f"g{index}", img_a, img_b, phash=10, dhash=10, histogram=0.1, thumbnail=0.125))
        elif state == "negative":
            rows.append(_row(f"g{index}", img_a, img_b))
        else:
            raise AssertionError(state)
    return classify_pairs(score_duplicate_evidence(_frame(*rows)))


def _images(*ids):
    return pd.DataFrame({"image_id": list(ids)})


def _constraints(*rows):
    return pd.DataFrame(rows, columns=PAIR_CONSTRAINT_COLUMNS)


def test_exact_group_and_singleton_are_assigned_deterministically():
    classified = _classified_specs(
        [("a", "b", "exact"), ("a", "c", "negative"), ("b", "c", "negative")]
    )
    groups = build_duplicate_groups(_images("c", "b", "a"), classified)
    group_of = dict(zip(groups.image_id, groups.duplicate_group_id))
    assert tuple(groups.columns) == DUPLICATE_GROUP_COLUMNS
    assert groups.image_id.tolist() == ["a", "b", "c"]
    assert group_of["a"] == group_of["b"]
    assert group_of["a"] != group_of["c"]


def test_complete_positive_clique_expands_to_all_three_pairs():
    classified = _classified_specs(
        [("a", "b", "positive"), ("a", "c", "positive"), ("b", "c", "positive")]
    )
    groups = build_duplicate_groups(_images("a", "b", "c"), classified)
    expanded = expand_group_pairs(groups)
    assert tuple(expanded.columns) == GROUP_PAIR_COLUMNS
    assert set(expanded[["img_a", "img_b"]].itertuples(index=False, name=None)) == {
        ("a", "b"),
        ("a", "c"),
        ("b", "c"),
    }
    assert expanded.duplicate_group_id.nunique() == 1


def test_weak_bridge_does_not_merge_incompatible_components():
    classified = _classified_specs(
        [("a", "b", "positive"), ("a", "c", "negative"), ("b", "c", "positive")]
    )
    classified.loc[(classified.img_a == "a") & (classified.img_b == "b"), "duplicate_score"] = 0.99
    classified.loc[(classified.img_a == "b") & (classified.img_b == "c"), "duplicate_score"] = 0.90
    groups = build_duplicate_groups(_images("a", "b", "c"), classified)
    group_of = dict(zip(groups.image_id, groups.duplicate_group_id))
    assert group_of["a"] == group_of["b"]
    assert group_of["c"] != group_of["a"]


def test_grouping_is_independent_of_input_row_order():
    classified = _classified_specs(
        [("a", "b", "positive"), ("a", "c", "positive"), ("b", "c", "positive")]
    )
    first = build_duplicate_groups(_images("a", "b", "c"), classified)
    second = build_duplicate_groups(
        _images("c", "a", "b"), classified.sample(frac=1, random_state=42)
    )
    pd.testing.assert_frame_equal(first, second)


def test_nonduplicate_constraint_blocks_direct_and_transitive_merge():
    classified = _classified_specs(
        [("a", "b", "positive"), ("a", "c", "positive"), ("b", "c", "positive")]
    )
    constraints = _constraints(("a", "c", "same_scene_only"))
    groups = build_duplicate_groups(_images("a", "b", "c"), classified, constraints)
    group_of = dict(zip(groups.image_id, groups.duplicate_group_id))
    assert group_of["a"] == group_of["b"]
    assert group_of["a"] != group_of["c"]
    assert find_group_conflicts(groups, constraints).empty


def test_duplicate_constraint_can_force_merge_of_automatic_negatives():
    classified = _classified_specs([("a", "b", "negative")])
    constraints = _constraints(("a", "b", "duplicate"))
    groups = build_duplicate_groups(_images("a", "b"), classified, constraints)
    assert groups.duplicate_group_id.nunique() == 1
    assert find_group_conflicts(groups, constraints).empty


def test_find_group_conflicts_reports_required_forbidden_and_relation_conflicts():
    groups = pd.DataFrame(
        [("a", "g1"), ("b", "g1"), ("c", "g2")], columns=DUPLICATE_GROUP_COLUMNS
    )
    constraints = _constraints(
        ("a", "c", "duplicate"),
        ("a", "b", "same_scene_only"),
        ("a", "b", "duplicate"),
    )
    conflicts = find_group_conflicts(groups, constraints)
    assert tuple(conflicts.columns) == GROUP_CONFLICT_COLUMNS
    assert {
        "required_duplicate_separated",
        "forbidden_duplicate_merged",
        "conflicting_pair_relations",
    } <= set(conflicts.conflict_type)


def test_contradictory_exact_and_nonduplicate_constraint_fails_loudly():
    classified = _classified_specs([("a", "b", "exact")])
    constraints = _constraints(("a", "b", "different_scene"))
    with pytest.raises(ValueError, match="constraint conflicts at exact grouping"):
        build_duplicate_groups(_images("a", "b"), classified, constraints)


def test_transitive_forced_duplicate_chain_cannot_bypass_rejection():
    classified = _classified_specs(
        [("a", "b", "negative"), ("a", "c", "negative"), ("b", "c", "negative")]
    )
    constraints = _constraints(
        ("a", "b", "duplicate"),
        ("b", "c", "duplicate"),
        ("a", "c", "same_scene_only"),
    )
    with pytest.raises(ValueError, match="would violate a nonduplicate constraint"):
        build_duplicate_groups(_images("a", "b", "c"), classified, constraints)


def test_singletons_expand_to_header_only_pair_table():
    groups = pd.DataFrame([("a", "g1"), ("b", "g2")], columns=DUPLICATE_GROUP_COLUMNS)
    expanded = expand_group_pairs(groups)
    assert expanded.empty
    assert tuple(expanded.columns) == GROUP_PAIR_COLUMNS
