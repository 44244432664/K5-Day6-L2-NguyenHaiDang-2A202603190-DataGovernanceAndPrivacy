"""Task 8 tests for scene constraints and deterministic group-aware splitting."""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from src.common.io import load_config
from src.t05 import (
    PAIR_CONSTRAINT_COLUMNS,
    SPLIT_GROUP_COLUMNS,
    SPLIT_MANIFEST_COLUMNS,
    assign_group_splits,
    build_duplicate_groups,
    build_pair_evidence,
    build_split_groups,
    classify_pairs,
    expand_group_pairs,
    extract_features,
    load_inputs,
    make_pair_id,
    score_duplicate_evidence,
    summarize_split,
)
from src.t05.contracts import DUPLICATE_GROUP_COLUMNS


EVIDENCE_COLUMNS = (
    "pair_id",
    "img_a",
    "img_b",
    "phash_distance",
    "dhash_distance",
    "thumbnail_distance",
    "same_camera",
    "time_delta_seconds",
)


def _images(specs):
    return pd.DataFrame(
        [
            {
                "image_id": image_id,
                "camera_id": camera,
                "captured_at": pd.Timestamp(seconds, unit="s", tz="UTC"),
            }
            for image_id, camera, seconds in specs
        ]
    )


def _singleton_duplicate_groups(images):
    return pd.DataFrame(
        [(image_id, f"dup:{image_id}") for image_id in images.image_id],
        columns=DUPLICATE_GROUP_COLUMNS,
    )


def _evidence(images, compatible_pairs=()):
    compatible = {tuple(sorted(pair)) for pair in compatible_pairs}
    metadata = images.set_index("image_id")
    rows = []
    for img_a, img_b in combinations(sorted(images.image_id), 2):
        visual = (img_a, img_b) in compatible
        same_camera = metadata.at[img_a, "camera_id"] == metadata.at[img_b, "camera_id"]
        delta = abs(
            (metadata.at[img_a, "captured_at"] - metadata.at[img_b, "captured_at"]).total_seconds()
        )
        rows.append(
            {
                "pair_id": make_pair_id(img_a, img_b),
                "img_a": img_a,
                "img_b": img_b,
                "phash_distance": 4 if visual else 32,
                "dhash_distance": 32,
                "thumbnail_distance": 0.5,
                "same_camera": same_camera,
                "time_delta_seconds": delta,
            }
        )
    return pd.DataFrame(rows, columns=EVIDENCE_COLUMNS)


def _overrides(*rows):
    return pd.DataFrame(rows, columns=PAIR_CONSTRAINT_COLUMNS)


def test_duplicate_and_scene_only_links_share_split_but_not_duplicate_output():
    images = _images([("a", "cam", 0), ("b", "cam", 1000), ("c", "cam", 2000)])
    duplicate_groups = pd.DataFrame(
        [("a", "dup:a"), ("b", "dup:b"), ("c", "dup:c")],
        columns=DUPLICATE_GROUP_COLUMNS,
    )
    before_pairs = expand_group_pairs(duplicate_groups)
    groups = build_split_groups(
        images,
        duplicate_groups,
        _evidence(images),
        _overrides(("a", "b", "same_scene_only")),
    )
    group_of = dict(zip(groups.image_id, groups.split_group_id))

    assert tuple(groups.columns) == SPLIT_GROUP_COLUMNS
    assert group_of["a"] == group_of["b"]
    assert group_of["a"] != group_of["c"]
    assert before_pairs.empty  # scene-only relation did not create a duplicate pair
    assert dict(zip(groups.image_id, groups.duplicate_group_id)) == dict(
        zip(duplicate_groups.image_id, duplicate_groups.duplicate_group_id)
    )


def test_same_camera_time_requires_visual_support_for_automatic_scene_link():
    images = _images([("a", "cam", 0), ("b", "cam", 40), ("c", "cam", 50)])
    groups = build_split_groups(
        images,
        _singleton_duplicate_groups(images),
        _evidence(images, compatible_pairs=[("a", "b")]),
    )
    group_of = dict(zip(groups.image_id, groups.split_group_id))
    assert group_of["a"] == group_of["b"]
    assert group_of["c"] != group_of["a"]


def test_burst_max_span_prevents_timestamp_chain():
    images = _images([("a", "cam", 0), ("b", "cam", 50), ("c", "cam", 100)])
    groups = build_split_groups(
        images,
        _singleton_duplicate_groups(images),
        _evidence(images, compatible_pairs=[("a", "b"), ("b", "c")]),
        params={"burst_window_seconds": 60, "burst_max_span_seconds": 60},
    )
    group_of = dict(zip(groups.image_id, groups.split_group_id))
    assert group_of["a"] == group_of["b"]
    assert group_of["c"] != group_of["a"]


def test_different_scene_blocks_automatic_and_transitive_scene_merge():
    images = _images([("a", "cam", 0), ("b", "cam", 20), ("c", "cam", 40)])
    groups = build_split_groups(
        images,
        _singleton_duplicate_groups(images),
        _evidence(images, compatible_pairs=[("a", "b"), ("b", "c"), ("a", "c")]),
        _overrides(("a", "c", "different_scene")),
    )
    group_of = dict(zip(groups.image_id, groups.split_group_id))
    assert group_of["a"] == group_of["b"]
    assert group_of["c"] != group_of["a"]


def test_contradictory_required_and_different_scene_paths_fail_loudly():
    images = _images([("a", "cam", 0), ("b", "cam", 100), ("c", "cam", 200)])
    overrides = _overrides(
        ("a", "b", "same_scene_only"),
        ("b", "c", "same_scene_only"),
        ("a", "c", "different_scene"),
    )
    with pytest.raises(ValueError, match="would violate different_scene"):
        build_split_groups(
            images, _singleton_duplicate_groups(images), _evidence(images), overrides
        )


def _split_groups_from_sizes(sizes):
    rows = []
    index = 0
    for group_index, size in enumerate(sizes):
        for _ in range(size):
            image_id = f"im_{index:03d}"
            rows.append(
                {
                    "image_id": image_id,
                    "duplicate_group_id": f"dup:{image_id}",
                    "split_group_id": f"scene:{group_index}",
                    "split_group_size": size,
                    "exceeds_max_component_size": False,
                }
            )
            index += 1
    return pd.DataFrame(rows, columns=SPLIT_GROUP_COLUMNS)


def test_group_assignment_is_full_coverage_group_safe_and_row_order_independent():
    groups = _split_groups_from_sizes([6, 6, 6, 6, 6])
    first = assign_group_splits(groups, 0.3, np.random.default_rng(42))
    second = assign_group_splits(
        groups.sample(frac=1, random_state=7), 0.3, np.random.default_rng(42)
    )
    pd.testing.assert_frame_equal(first, second)
    assert tuple(first.columns) == SPLIT_MANIFEST_COLUMNS
    assert len(first) == len(groups)
    merged = groups.merge(first, on="image_id")
    assert (merged.groupby("split_group_id").split.nunique() == 1).all()
    assert set(first.split) == {"train", "test"}
    assert "drop" not in set(first.split)


def test_unequal_groups_approach_target_without_splitting_or_dropping():
    groups = _split_groups_from_sizes([8, 2])
    assignments = assign_group_splits(groups, 0.3, np.random.default_rng(42))
    summary = summarize_split(groups[["image_id"]], assignments, groups)
    assert summary["n_test"] == 2
    assert summary["n_train"] == 8
    assert summary["n_drop"] == 0
    assert summary["actual_test_fraction"] == pytest.approx(0.2)
    assert summary["test_fraction_deviation"] == pytest.approx(0.1)
    assert summary["recommended_status"] == "PROVISIONAL"


def test_single_indivisible_group_is_reported_as_impossible_hold():
    groups = _split_groups_from_sizes([5])
    assignments = assign_group_splits(groups, 0.3, np.random.default_rng(42))
    summary = summarize_split(groups[["image_id"]], assignments, groups)
    assert set(assignments.split) == {"train"}
    assert summary["two_way_split_possible"] is False
    assert summary["usable_train_test"] is False
    assert summary["recommended_status"] == "HOLD"
    assert "IMPOSSIBLE_TWO_WAY_GROUP_SPLIT" in {risk["code"] for risk in summary["risks"]}


def test_oversized_component_is_retained_and_reported_as_hold():
    images = _images([(f"i{i}", "cam", i * 1000) for i in range(4)])
    duplicate_groups = pd.DataFrame(
        [(image_id, "dup:all") for image_id in images.image_id],
        columns=DUPLICATE_GROUP_COLUMNS,
    )
    groups = build_split_groups(
        images,
        duplicate_groups,
        _evidence(images),
        params={"max_component_size": 3},
    )
    assignments = assign_group_splits(groups, 0.3, np.random.default_rng(42))
    summary = summarize_split(images, assignments, groups)
    assert groups.exceeds_max_component_size.all()
    assert summary["retained_count"] == 4
    assert summary["n_drop"] == 0
    assert summary["oversized_split_group_count"] == 1
    assert summary["recommended_status"] == "HOLD"


def test_repository_fixture_produces_group_safe_full_coverage_split():
    loaded = load_inputs("fixtures/t05_near_duplicate", strict_provenance=True)
    params = load_config("configs/lab.yaml")["params"]
    features = extract_features(
        loaded["images"], "fixtures/t05_near_duplicate", params
    )
    evidence = build_pair_evidence(loaded["images"], features)
    classified = classify_pairs(score_duplicate_evidence(evidence, params), params)
    duplicate_groups = build_duplicate_groups(loaded["images"], classified)
    split_groups = build_split_groups(
        loaded["images"], duplicate_groups, classified, params=params
    )
    assignments = assign_group_splits(
        split_groups, params["test_fraction"], np.random.default_rng(42)
    )
    summary = summarize_split(loaded["images"], assignments, split_groups)

    assert len(assignments) == 120
    assert assignments.image_id.is_unique
    assert set(assignments.split) == {"train", "test"}
    assert summary["n_drop"] == 0
    assert summary["cross_split_group_count"] == 0
    assert summary["usable_train_test"] is True
    merged = split_groups.merge(assignments, on="image_id")
    assert (merged.groupby("split_group_id").split.nunique() == 1).all()
