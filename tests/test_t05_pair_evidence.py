"""Task 4 tests for complete, deterministic T05 pair evidence."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.common.io import load_config
from src.t05.contracts import IMAGE_FEATURE_COLUMNS, PAIR_EVIDENCE_COLUMNS
from src.t05.data import IMAGE_TABLE_COLUMNS, load_inputs
from src.t05.features import build_pair_evidence, compare_pair, extract_features


def _feature(
    image_id: str,
    sha: str,
    phash,
    dhash,
    histogram,
    thumbnail,
    path: str | None = None,
) -> dict:
    return {
        "image_id": image_id,
        "path": path or f"images/{image_id}.png",
        "camera_id": "unused",
        "captured_at": pd.Timestamp("2026-01-01", tz="UTC"),
        "sha256": sha,
        "width": 2,
        "height": 2,
        "phash": np.asarray(phash, dtype=bool),
        "dhash": np.asarray(dhash, dtype=bool),
        "color_histogram": np.asarray(histogram, dtype=float),
        "thumbnail": np.asarray(thumbnail, dtype=np.float32),
    }


def _metadata(image_id: str, camera: str, captured_at: str, path: str | None = None) -> dict:
    return {
        "image_id": image_id,
        "path": path or f"images/{image_id}.png",
        "camera_id": camera,
        "captured_at": captured_at,
    }


def test_compare_pair_uses_documented_symmetric_distance_units():
    a = _feature("z", "sha-a", [0, 1, 0, 1], [0, 0, 1, 1], [1, 0], [[0, 0], [0, 0]])
    b = _feature("a", "sha-b", [1, 1, 0, 0], [1, 0, 0, 1], [0.25, 0.75], [[0.5, 0.5], [0.5, 0.5]])
    evidence = compare_pair(
        a,
        b,
        _metadata("z", "cam_0", "2026-01-01T00:00:02.500Z"),
        _metadata("a", "cam_1", "2026-01-01T00:00:00Z"),
    )

    assert evidence["img_a"] == "a" and evidence["img_b"] == "z"
    assert evidence["exact_match"] is False
    assert evidence["phash_distance"] == 2
    assert evidence["dhash_distance"] == 2
    assert evidence["histogram_distance"] == pytest.approx(1.5)
    assert evidence["thumbnail_distance"] == pytest.approx(0.5)
    assert evidence["same_camera"] is False
    assert evidence["time_delta_seconds"] == pytest.approx(2.5)
    assert np.isnan(evidence["duplicate_score"])
    assert np.isnan(evidence["review_priority"])
    assert evidence["reason_codes"] == ""


def test_compare_pair_exact_copy_has_zero_visual_distances():
    values = _feature("a", "same", [0, 1], [1, 0], [0.4, 0.6], [[0.1, 0.2]])
    copied = {**values, "image_id": "b", "path": "images/b.png"}
    evidence = compare_pair(
        values,
        copied,
        _metadata("a", "cam_0", "2026-01-01T00:00:00Z"),
        _metadata("b", "cam_0", "2026-01-01T00:00:01Z"),
    )
    assert evidence["exact_match"] is True
    assert evidence["phash_distance"] == 0
    assert evidence["dhash_distance"] == 0
    assert evidence["histogram_distance"] == 0.0
    assert evidence["thumbnail_distance"] == 0.0
    assert evidence["same_camera"] is True


def test_build_pair_evidence_is_complete_unique_and_order_independent():
    ids = ["c", "a", "b", "d"]
    images = pd.DataFrame(
        [_metadata(i, f"cam_{n % 2}", f"2026-01-01T00:00:0{n}Z") for n, i in enumerate(ids)],
        columns=IMAGE_TABLE_COLUMNS,
    )
    features = pd.DataFrame(
        [
            _feature(i, f"sha-{i}", [n % 2, 1], [1, n % 2], [0.5, 0.5], [[n / 10]])
            for n, i in enumerate(ids)
        ],
        columns=IMAGE_FEATURE_COLUMNS,
    )

    first = build_pair_evidence(images, features)
    second = build_pair_evidence(
        images.sample(frac=1, random_state=9),
        features.sample(frac=1, random_state=3),
    )

    assert tuple(first.columns) == PAIR_EVIDENCE_COLUMNS
    assert len(first) == 6
    assert first.pair_id.is_unique
    assert (first.img_a < first.img_b).all()
    pd.testing.assert_frame_equal(first, second)


@pytest.mark.parametrize("n", [0, 1])
def test_zero_or_one_image_returns_valid_empty_evidence(n):
    images = pd.DataFrame(
        [_metadata("only", "cam_0", "2026-01-01T00:00:00Z")][:n],
        columns=IMAGE_TABLE_COLUMNS,
    )
    features = pd.DataFrame(
        [_feature("only", "sha", [0], [0], [1.0], [[0.0]])][:n],
        columns=IMAGE_FEATURE_COLUMNS,
    )
    result = build_pair_evidence(images, features)
    assert result.empty
    assert tuple(result.columns) == PAIR_EVIDENCE_COLUMNS


def test_pair_validation_rejects_incompatible_or_nonfinite_features():
    metadata_a = _metadata("a", "cam", "2026-01-01T00:00:00Z")
    metadata_b = _metadata("b", "cam", "2026-01-01T00:00:01Z")
    a = _feature("a", "a", [0, 1], [0], [1.0], [[0.0]])
    wrong_shape = _feature("b", "b", [0], [0], [1.0], [[0.0]])
    with pytest.raises(ValueError, match="phash shapes differ"):
        compare_pair(a, wrong_shape, metadata_a, metadata_b)

    nonfinite = _feature("b", "b", [0, 1], [0], [1.0], [[np.nan]])
    with pytest.raises(ValueError, match="non-finite"):
        compare_pair(a, nonfinite, metadata_a, metadata_b)


def test_build_pair_evidence_rejects_id_or_path_mismatch():
    images = pd.DataFrame(
        [_metadata("a", "cam", "2026-01-01T00:00:00Z")],
        columns=IMAGE_TABLE_COLUMNS,
    )
    missing = pd.DataFrame([], columns=IMAGE_FEATURE_COLUMNS)
    with pytest.raises(ValueError, match="do not match images"):
        build_pair_evidence(images, missing)

    stale = pd.DataFrame(
        [_feature("a", "sha", [0], [0], [1.0], [[0.0]], path="images/stale.png")],
        columns=IMAGE_FEATURE_COLUMNS,
    )
    with pytest.raises(ValueError, match="stale feature path"):
        build_pair_evidence(images, stale)


def test_repository_fixture_produces_all_pairs_with_finite_distances():
    loaded = load_inputs("fixtures/t05_near_duplicate", strict_provenance=True)
    config = load_config("configs/lab.yaml")
    features = extract_features(
        loaded["images"], "fixtures/t05_near_duplicate", config["params"]
    )
    evidence = build_pair_evidence(loaded["images"], features)

    assert len(evidence) == 120 * 119 // 2 == 7140
    assert evidence.pair_id.is_unique
    assert (evidence.img_a < evidence.img_b).all()
    assert int(evidence.exact_match.sum()) == 20
    for column in (
        "phash_distance",
        "dhash_distance",
        "histogram_distance",
        "thumbnail_distance",
        "time_delta_seconds",
    ):
        assert np.isfinite(evidence[column]).all()
    exact = evidence[evidence.exact_match]
    assert (exact.phash_distance == 0).all()
    assert (exact.dhash_distance == 0).all()
    assert (exact.histogram_distance == 0).all()
    assert (exact.thumbnail_distance == 0).all()
