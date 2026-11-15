"""No instructor labels: regression and independently generated holdout checks."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PIL import Image, ImageDraw

from src.t05 import (
    PAIR_REVIEW_COLUMNS, build_pair_evidence, classify_pairs, extract_features,
    make_pair_id, rank_review_pairs, rebuild_after_review, score_duplicate_evidence,
    verify_visual_pairs,
)
from tools.t05_experiments import build_development_examples, evaluate_development_run


def _regression_sources(root):
    root.mkdir()
    rows = []
    specs = [("a", "s1", (30, 60, 100)), ("b", "s1", (45, 75, 115)),
             ("c", "s2", (35, 65, 105)), ("d", "s3", (180, 30, 50))]
    for index, (name, scene, color) in enumerate(specs):
        array = np.zeros((36, 48, 3), dtype=np.uint8)
        array[:] = color
        array[4 + index:12 + index, 8:32] = tuple(255 - value for value in color)
        array[20:28, 10 + index:20 + index] = (index * 40, 220, 30)
        Image.fromarray(array).save(root / f"{name}.png")
        rows.append(dict(image_id=name, path=f"{name}.png", scene_id=scene,
                         camera_id=f"cam-{scene}", captured_at=f"2026-01-01T00:00:{index:02d}Z"))
    return pd.DataFrame(rows)


def _holdout_sources(root, seed):
    """New pixels/layouts/dimensions, not the four development originals."""
    root.mkdir()
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(6):
        y, x = np.mgrid[:60, :80]
        base = np.zeros((60, 80, 3), dtype=np.uint8)
        color = rng.integers(35, 160, size=3)
        base[:] = color
        base = np.clip(base.astype(float) + (x[..., None] + y[..., None]) / 6, 0, 255).astype(np.uint8)
        image = Image.fromarray(base)
        draw = ImageDraw.Draw(image)
        for _ in range(5):
            left, top = rng.integers(4, 45), rng.integers(4, 30)
            draw.rectangle((int(left), int(top), int(left + 18), int(top + 16)),
                           fill=tuple(int(c) for c in rng.integers(10, 240, size=3)))
        image.save(root / f"h{index}.png")
        rows.append(dict(image_id=f"h{index}", path=f"h{index}.png", scene_id=f"s{index // 2}",
                         camera_id=f"cam{index // 2}", captured_at=f"2026-02-01T00:00:{index:02d}Z"))
    return pd.DataFrame(rows)


def _evaluate(data, enabled):
    params = {**data["t05_params"], "verification_enabled": enabled}
    classified = classify_pairs(score_duplicate_evidence(data["pair_evidence"], params), params)
    rebuilt = rebuild_after_review(data["images"], classified, pd.DataFrame(columns=PAIR_REVIEW_COLUMNS),
                                   params, np.random.default_rng(42))
    metrics = evaluate_development_run(rebuilt["predictions"], data["labels"],
                                      {"assignments": rebuilt["split_manifest"], "partition": "all",
                                       "bootstrap_iterations": 0, "seed": 42})
    return classified, rebuilt, metrics


def test_crop_and_lookalike_regression_without_gt(tmp_path):
    sources = _regression_sources(tmp_path / "sources")
    data = build_development_examples(sources, {"source_root": tmp_path / "sources",
        "output_dir": tmp_path / "examples", "evaluation_partition": "all", "bootstrap_iterations": 0})
    _, old, before = _evaluate(data, 0)
    classified, new, after = _evaluate(data, 1)
    assert before["fp"] > 0 and before["fn"] > 0
    assert after["tp"] == 40 and after["fp"] == after["fn"] == 0
    assert after["residual_duplicate_leak_rate"] == after["residual_scene_leak_rate"] == 0
    assert after["retention_fraction"] == 1
    assert "registered_crop" in ";".join(classified.reason_codes)
    assert {"verification_error", "verification_crop_fraction", "verification_texture"} <= set(classified)


@pytest.mark.parametrize("seed", [2027, 9173])
def test_frozen_verifier_on_independent_synthetic_holdout(tmp_path, seed):
    sources = _holdout_sources(tmp_path / "sources", seed)
    data = build_development_examples(sources, {"source_root": tmp_path / "sources",
        "output_dir": tmp_path / "examples", "evaluation_partition": "all", "bootstrap_iterations": 0,
        "crop_fraction": 0.07, "jpeg_quality": 55, "brightness_factor": 0.9})
    _, _, before = _evaluate(data, 0)
    _, _, after = _evaluate(data, 1)
    assert after["f1"] > before["f1"]
    assert after["fp"] <= before["fp"]
    assert after["recall"] >= before["recall"]
    assert after["retention_fraction"] == 1


def test_verification_is_symmetric_order_independent_and_preserves_inputs(tmp_path):
    sources = _regression_sources(tmp_path / "sources")
    images = sources.rename(columns={"scene_id": "unused"})
    raw = build_pair_evidence(images, extract_features(images, tmp_path / "sources"))
    original = raw.copy(deep=True)
    first = verify_visual_pairs(images, tmp_path / "sources", raw)
    second = verify_visual_pairs(images.iloc[::-1], tmp_path / "sources", raw.iloc[::-1])
    pd.testing.assert_frame_equal(first.sort_index(), second.sort_index())
    reversed_pairs = raw.assign(img_a=raw.img_b, img_b=raw.img_a)
    reverse = verify_visual_pairs(images, tmp_path / "sources", reversed_pairs)
    np.testing.assert_allclose(first.verification_error, reverse.verification_error, atol=1e-7)
    pd.testing.assert_frame_equal(raw, original)
    bad = first.copy()
    bad.loc[0, "verification_error"] = np.nan
    with pytest.raises(ValueError, match="verification_error"):
        score_duplicate_evidence(bad)


def test_flat_nonexact_content_abstains_instead_of_proving_a_duplicate(tmp_path):
    Image.new("RGB", (40, 30), (60, 90, 120)).save(tmp_path / "a.png")
    Image.new("RGB", (40, 30), (60, 90, 120)).save(tmp_path / "b.bmp")
    images = pd.DataFrame(dict(image_id=["a", "b"], path=["a.png", "b.bmp"],
        camera_id=["one", "two"], captured_at=["2026-01-01T00:00:00Z"] * 2))
    evidence = verify_visual_pairs(images, tmp_path, build_pair_evidence(images, extract_features(images, tmp_path)))
    classified = classify_pairs(score_duplicate_evidence(evidence))
    assert not classified.exact_match.iloc[0]
    assert classified.pred_dup.iloc[0] == 0


def test_review_reserves_accepted_and_withheld_lanes_before_repeating_bridges(tmp_path):
    sources = _regression_sources(tmp_path / "sources")
    data = build_development_examples(sources, {"source_root": tmp_path / "sources",
        "output_dir": tmp_path / "examples", "evaluation_partition": "all", "bootstrap_iterations": 0})
    classified, old, _ = _evaluate(data, 0)
    queue = rank_review_pairs(classified, old["split_groups"], old["split_manifest"], 20)
    comparison = data["labels"].merge(old["predictions"], on=["img_a", "img_b"])
    errors = comparison[(comparison.is_duplicate != comparison.pred_dup)
                        & (comparison.partition != "excluded_cross_partition")]
    error_ids = {make_pair_id(row.img_a, row.img_b) for row in errors.itertuples()}
    assert set(queue.item_id) & error_ids
    assert "accepted_match_audit" in "|".join(queue["slice"])
    assert "threshold_uncertain" in "|".join(queue["slice"])
    repeated = rank_review_pairs(classified.sample(frac=1, random_state=8),
                                old["split_groups"], old["split_manifest"], 20)
    pd.testing.assert_frame_equal(queue, repeated)
    assert len(queue) == 20 and queue.score.is_monotonic_decreasing


@pytest.mark.parametrize("params", [{"verification_error_max": 0}, {"verification_crop_max": 0.25},
    {"verification_width": 1}, {"verification_enabled": True}, {"verification_crop_steps": 0}])
def test_verification_rejects_invalid_configuration(params):
    from src.t05 import validate_params
    with pytest.raises(ValueError, match="verification"):
        validate_params(params)
