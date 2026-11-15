"""Task 3 tests for deterministic T05 per-image feature extraction."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.common.io import load_config
from src.t05.contracts import IMAGE_FEATURE_COLUMNS
from src.t05.data import load_inputs
from src.t05.features import extract_features, extract_image_features


def _save(path: Path, mode: str = "RGB", color=None, size=(16, 10)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if color is None:
        color = {"RGB": (20, 80, 140), "RGBA": (20, 80, 140, 90), "L": 80}[mode]
    Image.new(mode, size, color).save(path)
    return path


def _params() -> dict:
    return {
        "phash_size": 4,
        "dhash_size": 4,
        "histogram_bins": 8,
        "thumbnail_width": 7,
        "thumbnail_height": 5,
    }


def test_extract_image_features_has_configured_shapes_ranges_and_units(tmp_path):
    path = _save(tmp_path / "image.png")
    features = extract_image_features(path, _params())

    assert features["width"] == 16 and features["height"] == 10
    assert len(features["sha256"]) == 64
    assert features["phash"].shape == (16,) and features["phash"].dtype == bool
    assert features["dhash"].shape == (16,) and features["dhash"].dtype == bool
    assert features["color_histogram"].shape == (24,)
    assert features["color_histogram"].sum() == pytest.approx(1.0)
    assert features["thumbnail"].shape == (5, 7)
    assert features["thumbnail"].dtype == np.float32
    assert 0.0 <= float(features["thumbnail"].min())
    assert float(features["thumbnail"].max()) <= 1.0
    assert all(np.isfinite(features[name]).all() for name in ("phash", "dhash", "color_histogram", "thumbnail"))


def test_feature_extraction_is_deterministic_and_exact_copies_match(tmp_path):
    first_path = _save(tmp_path / "first.png")
    copy_path = tmp_path / "copy.png"
    copy_path.write_bytes(first_path.read_bytes())

    first = extract_image_features(first_path)
    repeated = extract_image_features(first_path)
    copied = extract_image_features(copy_path)
    assert first["sha256"] == repeated["sha256"] == copied["sha256"]
    for name in ("phash", "dhash", "color_histogram", "thumbnail"):
        np.testing.assert_array_equal(first[name], repeated[name])
        np.testing.assert_array_equal(first[name], copied[name])


@pytest.mark.parametrize(
    ("mode", "suffix"),
    [("L", "gray.png"), ("RGBA", "alpha.png")],
)
def test_grayscale_and_rgba_are_converted_to_rgb(tmp_path, mode, suffix):
    features = extract_image_features(_save(tmp_path / suffix, mode=mode))
    assert features["width"] == 16 and features["height"] == 10
    assert np.isfinite(features["thumbnail"]).all()
    assert features["color_histogram"].sum() == pytest.approx(1.0)


def test_uniform_image_features_are_finite(tmp_path):
    features = extract_image_features(_save(tmp_path / "uniform.png", color=(0, 0, 0)))
    for name in ("phash", "dhash", "color_histogram", "thumbnail"):
        assert np.isfinite(features[name]).all()
    assert features["color_histogram"].sum() == pytest.approx(1.0)


def test_extract_features_is_sorted_and_does_not_mutate_input(tmp_path):
    _save(tmp_path / "images/b.png", color=(200, 0, 0))
    _save(tmp_path / "images/a.png", color=(0, 200, 0))
    images = pd.DataFrame(
        {
            "image_id": ["b", "a"],
            "path": ["images/b.png", "images/a.png"],
            "camera_id": ["cam_1", "cam_0"],
            "captured_at": pd.to_datetime(
                ["2026-10-01T08:00:01Z", "2026-10-01T08:00:00Z"], utc=True
            ),
        }
    )
    original = images.copy(deep=True)

    features = extract_features(images, tmp_path, _params())

    assert tuple(features.columns) == IMAGE_FEATURE_COLUMNS
    assert features.image_id.tolist() == ["a", "b"]
    assert features.path.tolist() == ["images/a.png", "images/b.png"]
    pd.testing.assert_frame_equal(images, original)


def test_extract_features_reports_image_id_and_path_for_bad_image(tmp_path):
    bad = tmp_path / "images/bad.png"
    bad.parent.mkdir()
    bad.write_text("not an image", encoding="utf-8")
    images = pd.DataFrame(
        {
            "image_id": ["broken_01"],
            "path": ["images/bad.png"],
            "camera_id": ["cam_0"],
            "captured_at": ["2026-10-01T08:00:00"],
        }
    )
    with pytest.raises(ValueError, match=r"broken_01.*images/bad\.png.*cannot decode"):
        extract_features(images, tmp_path)


def test_extract_features_rejects_bad_table_and_params(tmp_path):
    with pytest.raises(ValueError, match="missing required columns"):
        extract_features(pd.DataFrame({"image_id": ["x"]}), tmp_path)
    path = _save(tmp_path / "image.png")
    with pytest.raises(ValueError, match="phash_size"):
        extract_image_features(path, {"phash_size": 2})


def test_repository_fixture_extracts_one_finite_row_per_image():
    loaded = load_inputs("fixtures/t05_near_duplicate", strict_provenance=True)
    config = load_config("configs/lab.yaml")
    features = extract_features(
        loaded["images"], "fixtures/t05_near_duplicate", config["params"]
    )

    assert len(features) == 120
    assert features.image_id.is_monotonic_increasing
    assert features.image_id.is_unique
    assert all(len(bits) == 64 for bits in features.phash)
    assert all(len(bits) == 64 for bits in features.dhash)
    assert all(np.isfinite(values).all() for values in features.color_histogram)
    assert all(np.isfinite(values).all() for values in features.thumbnail)
