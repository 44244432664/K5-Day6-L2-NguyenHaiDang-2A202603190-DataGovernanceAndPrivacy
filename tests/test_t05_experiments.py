"""Task 13 tests for internal T05 development examples and ablations."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.metrics.tracks import eval_t05
from tools.t05_experiments import (
    build_development_examples,
    evaluate_development_run,
    run_ablations,
)


def _source_table(root: Path) -> pd.DataFrame:
    root.mkdir(parents=True, exist_ok=True)
    specs = [
        ("a", "scene-1", (30, 60, 100), "cam-1"),
        ("b", "scene-1", (45, 75, 115), "cam-1"),
        ("c", "scene-2", (35, 65, 105), "cam-2"),
        ("d", "scene-3", (180, 30, 50), "cam-3"),
    ]
    rows = []
    for index, (image_id, scene_id, color, camera) in enumerate(specs):
        array = np.zeros((36, 48, 3), dtype=np.uint8)
        array[:, :] = color
        array[4 + index : 12 + index, 8:32] = tuple(255 - value for value in color)
        array[20:28, 10 + index : 20 + index] = (index * 40, 220, 30)
        path = root / f"{image_id}.png"
        Image.fromarray(array).save(path)
        rows.append(
            {
                "image_id": image_id,
                "path": path.name,
                "scene_id": scene_id,
                "camera_id": camera,
                "captured_at": f"2026-01-01T00:{index:02d}:00Z",
            }
        )
    return pd.DataFrame(rows)


def _build(tmp_path: Path, name: str):
    source_root = tmp_path / "sources"
    sources = _source_table(source_root)
    return build_development_examples(
        sources,
        {
            "source_root": source_root,
            "output_dir": tmp_path / name,
            "seed": 42,
            "heldout_fraction": 0.34,
            "evaluation_partition": "all",
            "bootstrap_iterations": 20,
        },
    )


def test_build_development_examples_is_reproducible_grouped_and_logged(tmp_path):
    first = _build(tmp_path, "first")
    second = _build(tmp_path, "second")

    pd.testing.assert_frame_equal(first["images"], second["images"])
    pd.testing.assert_frame_equal(first["labels"], second["labels"])
    pd.testing.assert_frame_equal(first["case_summary"], second["case_summary"])
    assert len(first["images"]) == 20
    assert len(first["labels"]) == 190
    assert len(first["pair_evidence"]) == 190

    for row in first["images"].itertuples(index=False):
        first_bytes = (first["output_dir"] / row.path).read_bytes()
        second_bytes = (second["output_dir"] / row.path).read_bytes()
        assert first_bytes == second_bytes
    partition_counts = first["images"].groupby("scene_id").partition.nunique()
    assert (partition_counts == 1).all()
    source_partition_counts = first["images"].groupby("source_id").partition.nunique()
    assert (source_partition_counts == 1).all()

    labels = first["labels"]
    case_types = set(labels.case_type)
    assert {
        "exact_copy",
        "jpeg_recompression",
        "brightness_change",
        "modest_crop",
        "same_scene_nonduplicate",
        "visually_similar_different_scene",
    } <= case_types
    exact = labels[labels.case_type == "exact_copy"]
    assert len(exact) and exact.is_duplicate.eq(1).all()
    same_scene_only = labels[labels.case_type == "same_scene_nonduplicate"]
    assert len(same_scene_only)
    assert same_scene_only.is_duplicate.eq(0).all() and same_scene_only.same_scene.eq(1).all()

    frozen = json.loads(
        (first["output_dir"] / "frozen_thresholds.json").read_text(encoding="utf-8")
    )
    assert frozen["scope"] == "internal_development_only_not_official"
    assert frozen["seed"] == 42
    assert (first["output_dir"] / "development_cases.csv").exists()


def _toy_labels():
    return pd.DataFrame(
        [
            ("a", "b", 1, 1, "s1", "s1", "dev", "jpeg_recompression"),
            ("a", "c", 0, 0, "s1", "s2", "dev", "visually_similar_different_scene"),
            ("b", "c", 0, 0, "s1", "s2", "dev", "different_scene_negative"),
        ],
        columns=(
            "img_a",
            "img_b",
            "is_duplicate",
            "same_scene",
            "scene_id_a",
            "scene_id_b",
            "partition",
            "case_type",
        ),
    )


def test_evaluate_development_run_has_known_pair_leakage_and_cluster_answers():
    labels = _toy_labels()
    # Without pred_dup, listed rows are positive, matching the official evaluator.
    predictions = pd.DataFrame([("a", "b"), ("a", "c")], columns=("img_a", "img_b"))
    assignments = pd.DataFrame(
        [("a", "train"), ("b", "test"), ("c", "train")],
        columns=("image_id", "split"),
    )
    options = {
        "assignments": assignments,
        "partition": "dev",
        "bootstrap_iterations": 100,
        "seed": 7,
        "runtime_seconds": 3.0,
    }

    first = evaluate_development_run(predictions, labels, options)
    second = evaluate_development_run(predictions, labels, options)

    assert first["scope"] == "internal_development_only_not_official"
    assert first["tp"] == 1 and first["fp"] == 1 and first["fn"] == 0
    assert first["precision"] == pytest.approx(0.5)
    assert first["recall"] == pytest.approx(1.0)
    assert first["f1"] == pytest.approx(2 / 3)
    assert first["false_merge_count"] == 1
    assert first["false_merge_rate"] == pytest.approx(0.5)
    assert first["residual_duplicate_leak_rate"] == 1.0
    assert first["residual_scene_leak_rate"] == 1.0
    assert first["retention_fraction"] == 1.0
    assert first["actual_test_fraction"] == pytest.approx(1 / 3)
    assert first["runtime_per_image_seconds"] == 1.0
    assert first["uncertainty"]["unit"] == "scene_group"
    assert first["uncertainty"] == second["uncertainty"]


def test_review_effect_compares_automatic_and_final_on_same_pool():
    predictions = pd.DataFrame(
        [
            ("a", "b", 1, 0),
            ("a", "c", 0, 1),
            ("b", "c", 0, 0),
        ],
        columns=("img_a", "img_b", "pred_dup", "automatic_pred_dup"),
    )
    metrics = evaluate_development_run(
        predictions,
        _toy_labels(),
        {"partition": "dev", "bootstrap_iterations": 10, "seed": 42},
    )

    assert metrics["f1"] == 1.0
    assert metrics["review_effect"] == {
        "changed_pair_count": 2,
        "automatic_f1": 0.0,
        "final_f1": 1.0,
        "f1_delta": 1.0,
    }


def test_run_ablations_uses_fixed_pool_and_writes_failure_evidence(tmp_path):
    dataset = _build(tmp_path, "development")
    reviewed_predictions = dataset["labels"][["img_a", "img_b"]].copy()
    reviewed_predictions["pred_dup"] = dataset["labels"].is_duplicate.astype(int)
    reviewed_predictions["automatic_pred_dup"] = 0
    supplied_split = dataset["images"][["image_id"]].copy()
    supplied_split["split"] = [
        "test" if index % 3 == 0 else "train" for index in range(len(supplied_split))
    ]
    table = run_ablations(
        [
            {"name": "sha", "method": "exact_hash"},
            {"name": "phash", "method": "phash"},
            {"name": "hist", "method": "histogram"},
            {"name": "visual", "method": "combined_visual"},
            {"name": "visual_context", "method": "combined_visual_metadata"},
            {
                "name": "reviewed",
                "method": "supplied",
                "predictions": reviewed_predictions,
                "assignments": supplied_split,
            },
        ],
        dataset,
    )

    assert table.name.tolist() == [
        "sha",
        "phash",
        "hist",
        "visual",
        "visual_context",
        "reviewed",
    ]
    assert table.n_candidate_pairs.nunique() == 1
    assert set(table.scope) == {"internal_development_only_not_official"}
    assert set(table.split_strategy) == {
        "independent",
        "duplicate_group",
        "scene_group",
        "supplied",
    }
    assert table.f1.between(0, 1).all()
    assert table.retention_fraction.eq(1.0).all()
    reviewed_row = table.set_index("name").loc["reviewed"]
    assert reviewed_row.f1 == 1.0
    assert reviewed_row.review_changed_pair_count > 0
    assert reviewed_row.review_f1_delta > 0
    assert (dataset["output_dir"] / "ablation_results.csv").exists()
    assert (dataset["output_dir"] / "failure_evidence.csv").exists()
    protocol = json.loads(
        (dataset["output_dir"] / "ablation_protocol.json").read_text(encoding="utf-8")
    )
    assert protocol["fixed_candidate_pool"] is True
    assert protocol["bootstrap_unit"] == "scene_group"
    assert protocol["scope"] == "internal_development_only_not_official"


def test_unchanged_official_evaluator_accepts_generated_toy_contract(tmp_path):
    gt = pd.DataFrame(
        [
            ("a", "dup-1", "scene-1", "original"),
            ("b", "dup-1", "scene-1", "jpeg"),
            ("c", "dup-2", "scene-2", "original"),
        ],
        columns=("image_id", "dup_group_id", "scene_id", "transform"),
    )
    pred = pd.DataFrame(
        [("a", "b", 0.9, 1), ("a", "c", 0.1, 0), ("b", "c", 0.1, 0)],
        columns=("img_a", "img_b", "score", "pred_dup"),
    )
    split = pd.DataFrame(
        [("a", "train"), ("b", "train"), ("c", "test")],
        columns=("image_id", "split"),
    )
    gt_path = tmp_path / "gt.csv"
    pred_path = tmp_path / "suspect_pairs.csv"
    gt.to_csv(gt_path, index=False)
    pred.to_csv(pred_path, index=False)
    split.to_csv(tmp_path / "split_manifest.csv", index=False)

    result = eval_t05(gt_path, pred_path, {})
    primary = [metric for metric in result.metrics if metric.primary][0]
    assert primary.name == "Leakage Detection F1 (pair)"
    assert primary.value == 1.0
