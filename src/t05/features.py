"""Deterministic per-image feature extraction for T05.

This module computes reusable signals for individual images only. It does not
compare image pairs, assign duplicate labels, or construct scene/split groups.
"""
from __future__ import annotations

from collections.abc import Mapping
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

from ..common.image_ops import color_histogram, dhash, hamming, load_rgb, phash, to_gray
from ..common.io import sha256_file
from .contracts import (
    IMAGE_FEATURE_COLUMNS,
    PAIR_EVIDENCE_COLUMNS,
    canonical_pair,
    make_pair_id,
    validate_params,
)
from .data import IMAGE_TABLE_COLUMNS, resolve_image_path


def _thumbnail(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Return a deterministic HxW float32 grayscale thumbnail in [0, 1]."""
    gray = np.clip(to_gray(image), 0, 255).astype(np.uint8)
    resized = Image.fromarray(gray).resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / np.float32(255.0)


def _extract_image_features(path: Path, params: Mapping[str, int | float]) -> dict[str, Any]:
    """Internal extractor for an already normalized parameter mapping."""
    if not path.exists():
        raise FileNotFoundError(f"image file not found: {path}")
    if not path.is_file():
        raise ValueError(f"image path is not a file: {path}")

    try:
        image = load_rgb(path)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(f"cannot decode image at {path}: {exc}") from exc
    if image.ndim != 3 or image.shape[2] != 3 or image.shape[0] < 1 or image.shape[1] < 1:
        raise ValueError(f"decoded image at {path} has invalid RGB shape {image.shape}")

    p_bits = phash(
        image,
        size=int(params["phash_size"]),
        highfreq_factor=int(params["phash_highfreq_factor"]),
    ).astype(bool, copy=False)
    d_bits = dhash(image, size=int(params["dhash_size"])).astype(bool, copy=False)
    histogram = color_histogram(image, bins=int(params["histogram_bins"])).astype(
        np.float64, copy=False
    )
    thumbnail = _thumbnail(
        image,
        width=int(params["thumbnail_width"]),
        height=int(params["thumbnail_height"]),
    )

    expected_shapes = {
        "phash": (int(params["phash_size"]) ** 2,),
        "dhash": (int(params["dhash_size"]) ** 2,),
        "color_histogram": (3 * int(params["histogram_bins"]),),
        "thumbnail": (int(params["thumbnail_height"]), int(params["thumbnail_width"])),
    }
    arrays = {
        "phash": p_bits,
        "dhash": d_bits,
        "color_histogram": histogram,
        "thumbnail": thumbnail,
    }
    for name, values in arrays.items():
        if values.shape != expected_shapes[name]:
            raise ValueError(
                f"feature {name} for {path} has shape {values.shape}, expected {expected_shapes[name]}"
            )
        if not np.isfinite(values).all():
            raise ValueError(f"feature {name} for {path} contains non-finite values")

    height, width = image.shape[:2]
    return {
        "sha256": sha256_file(path),
        "width": int(width),
        "height": int(height),
        "phash": p_bits.copy(),
        "dhash": d_bits.copy(),
        "color_histogram": histogram.copy(),
        "thumbnail": thumbnail.copy(),
    }


def extract_image_features(
    path: str | Path, params: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Decode one image and compute deterministic T05 visual features.

    Args:
        path: Existing image file. Grayscale, RGB, and images with alpha are
            explicitly converted to RGB by the shared image loader.
        params: Flat T05 parameter mapping. Missing values receive Task 1
            defaults and invalid/unknown values are rejected.

    Returns:
        A dictionary containing ``sha256``, source ``width``/``height``, boolean
        flattened ``phash``/``dhash`` arrays, an L1-normalized color histogram,
        and a 2-D float32 grayscale ``thumbnail`` in [0, 1].

    Raises:
        FileNotFoundError: If the image does not exist.
        ValueError: If parameters are invalid, the path is not a file, decoding
            fails, or any feature has an invalid shape/non-finite value.
    """
    normalized = validate_params(params)
    return _extract_image_features(Path(path), normalized)


def extract_features(
    images: pd.DataFrame,
    input_dir: str | Path,
    params: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Extract one feature row per image in stable lexicographic ID order.

    ``images`` must contain the Task 2 columns ``image_id``, ``path``,
    ``camera_id``, and ``captured_at``. The input DataFrame is not mutated.
    Image paths are resolved safely inside ``input_dir``. The returned columns
    follow :data:`src.t05.contracts.IMAGE_FEATURE_COLUMNS`; array-valued fields
    are in-memory NumPy arrays and are not serialized by this task.

    Raises:
        ValueError: For a malformed table, duplicate/blank IDs, unsafe paths,
            invalid parameters, or feature extraction failure. Per-image errors
            include both image ID and path.
    """
    if not isinstance(images, pd.DataFrame):
        raise ValueError(f"images must be a pandas DataFrame, got {type(images).__name__}")
    missing = [column for column in IMAGE_TABLE_COLUMNS if column not in images.columns]
    if missing:
        raise ValueError(f"images table is missing required columns {missing}; has {list(images.columns)}")
    if images.empty:
        raise ValueError("images table must contain at least one image")

    normalized = validate_params(params)
    ordered = images.copy()
    if ordered["image_id"].isna().any():
        raise ValueError("images table has null image_id values")
    ordered["image_id"] = ordered["image_id"].astype(str)
    blank = ordered["image_id"].str.strip() == ""
    if blank.any():
        raise ValueError(f"images table has blank image_id values at rows {ordered.index[blank].tolist()[:5]}")
    duplicates = sorted(
        ordered.loc[ordered.image_id.duplicated(keep=False), "image_id"].unique().tolist()
    )
    if duplicates:
        raise ValueError(f"images table has duplicate image_id values: {duplicates[:5]}")
    ordered = ordered.sort_values("image_id", kind="mergesort")

    rows: list[dict[str, Any]] = []
    for row in ordered.itertuples(index=False):
        try:
            path = resolve_image_path(input_dir, row.path)
            values = _extract_image_features(path, normalized)
        except (FileNotFoundError, ValueError) as exc:
            raise ValueError(
                f"feature extraction failed for image_id={row.image_id!r}, path={row.path!r}: {exc}"
            ) from exc
        rows.append(
            {
                "image_id": str(row.image_id),
                "path": str(row.path),
                "camera_id": str(row.camera_id),
                "captured_at": row.captured_at,
                **values,
            }
        )

    return pd.DataFrame(rows, columns=IMAGE_FEATURE_COLUMNS)


def _record_value(record: Mapping[str, Any] | pd.Series, key: str, label: str) -> Any:
    """Read a required value from a mapping-like feature/metadata record."""
    try:
        return record[key]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{label} is missing required field {key!r}") from exc


def _bool_vector(record: Mapping[str, Any] | pd.Series, key: str, label: str) -> np.ndarray:
    values = np.asarray(_record_value(record, key, label))
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{label}.{key} must be a nonempty 1-D vector, got shape {values.shape}")
    if values.dtype != np.dtype(bool):
        raise ValueError(f"{label}.{key} must contain boolean hash bits, got dtype {values.dtype}")
    return values


def _float_array(
    record: Mapping[str, Any] | pd.Series,
    key: str,
    label: str,
    ndim: int,
) -> np.ndarray:
    try:
        values = np.asarray(_record_value(record, key, label), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}.{key} must be a numeric array") from exc
    if values.ndim != ndim or values.size == 0:
        raise ValueError(f"{label}.{key} must be a nonempty {ndim}-D array, got shape {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"{label}.{key} contains non-finite values")
    return values


def _timestamp(value: Any, label: str) -> pd.Timestamp:
    if value is None or (not isinstance(value, (list, tuple, np.ndarray)) and pd.isna(value)):
        raise ValueError(f"{label}.captured_at must be a valid timestamp")
    try:
        result = pd.to_datetime(value, errors="raise", utc=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label}.captured_at is invalid: {value!r}") from exc
    if not isinstance(result, pd.Timestamp) or pd.isna(result):
        raise ValueError(f"{label}.captured_at must be one timestamp, got {value!r}")
    return result


def compare_pair(
    features_a: Mapping[str, Any] | pd.Series,
    features_b: Mapping[str, Any] | pd.Series,
    metadata_a: Mapping[str, Any] | pd.Series,
    metadata_b: Mapping[str, Any] | pd.Series,
) -> dict[str, Any]:
    """Compute symmetric raw evidence for one unordered image pair.

    Distances are: pHash/dHash Hamming counts (lower is more similar),
    histogram L1 distance in ``[0,2]``, normalized thumbnail mean absolute
    error in ``[0,1]``, and absolute capture-time difference in seconds.

    No duplicate classification is performed. ``duplicate_score`` and
    ``review_priority`` are ``NaN`` and ``reason_codes`` is empty for later
    tasks to fill.

    Raises:
        ValueError: For missing fields, self-pairs, incompatible feature
            shapes/types, non-finite values, invalid normalization/ranges, or
            invalid metadata.
    """
    id_a = _record_value(metadata_a, "image_id", "metadata_a")
    id_b = _record_value(metadata_b, "image_id", "metadata_b")
    img_a, img_b = canonical_pair(id_a, id_b)

    # If feature records carry IDs, reject accidental metadata/feature
    # misalignment before producing a valid-looking pair.
    for features, expected, label in (
        (features_a, str(id_a), "features_a"),
        (features_b, str(id_b), "features_b"),
    ):
        if "image_id" in features and str(features["image_id"]) != expected:
            raise ValueError(
                f"{label}.image_id={features['image_id']!r} does not match metadata image_id={expected!r}"
            )

    sha_a = str(_record_value(features_a, "sha256", "features_a"))
    sha_b = str(_record_value(features_b, "sha256", "features_b"))
    if not sha_a.strip() or not sha_b.strip():
        raise ValueError("features sha256 values must be nonblank")

    phash_a = _bool_vector(features_a, "phash", "features_a")
    phash_b = _bool_vector(features_b, "phash", "features_b")
    dhash_a = _bool_vector(features_a, "dhash", "features_a")
    dhash_b = _bool_vector(features_b, "dhash", "features_b")
    histogram_a = _float_array(features_a, "color_histogram", "features_a", 1)
    histogram_b = _float_array(features_b, "color_histogram", "features_b", 1)
    thumbnail_a = _float_array(features_a, "thumbnail", "features_a", 2)
    thumbnail_b = _float_array(features_b, "thumbnail", "features_b", 2)

    for name, left, right in (
        ("phash", phash_a, phash_b),
        ("dhash", dhash_a, dhash_b),
        ("color_histogram", histogram_a, histogram_b),
        ("thumbnail", thumbnail_a, thumbnail_b),
    ):
        if left.shape != right.shape:
            raise ValueError(
                f"pair feature {name} shapes differ: {left.shape} != {right.shape}"
            )
    for label, histogram in (("features_a", histogram_a), ("features_b", histogram_b)):
        if np.any(histogram < 0) or not np.isclose(histogram.sum(), 1.0, atol=1e-6):
            raise ValueError(f"{label}.color_histogram must be nonnegative and L1-normalized")
    for label, thumbnail in (("features_a", thumbnail_a), ("features_b", thumbnail_b)):
        if np.any(thumbnail < 0) or np.any(thumbnail > 1):
            raise ValueError(f"{label}.thumbnail values must be in [0,1]")

    camera_a = str(_record_value(metadata_a, "camera_id", "metadata_a"))
    camera_b = str(_record_value(metadata_b, "camera_id", "metadata_b"))
    if not camera_a.strip() or not camera_b.strip():
        raise ValueError("metadata camera_id values must be nonblank")
    timestamp_a = _timestamp(_record_value(metadata_a, "captured_at", "metadata_a"), "metadata_a")
    timestamp_b = _timestamp(_record_value(metadata_b, "captured_at", "metadata_b"), "metadata_b")

    return {
        "pair_id": make_pair_id(img_a, img_b),
        "img_a": img_a,
        "img_b": img_b,
        "exact_match": bool(sha_a == sha_b),
        "phash_distance": hamming(phash_a, phash_b),
        "dhash_distance": hamming(dhash_a, dhash_b),
        "histogram_distance": float(np.abs(histogram_a - histogram_b).sum()),
        "thumbnail_distance": float(np.abs(thumbnail_a - thumbnail_b).mean()),
        "same_camera": bool(camera_a == camera_b),
        "time_delta_seconds": float(abs((timestamp_a - timestamp_b).total_seconds())),
        "duplicate_score": np.nan,
        "review_priority": np.nan,
        "reason_codes": "",
    }


def _indexed_rows(frame: pd.DataFrame, label: str, required: tuple[str, ...]) -> dict[str, pd.Series]:
    """Validate unique string IDs and return an ID-to-row mapping."""
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"{label} must be a pandas DataFrame, got {type(frame).__name__}")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns {missing}; has {list(frame.columns)}")
    if frame["image_id"].isna().any():
        raise ValueError(f"{label} has null image_id values")

    copy = frame.copy()
    copy["image_id"] = copy["image_id"].astype(str)
    blank = copy.image_id.str.strip() == ""
    if blank.any():
        raise ValueError(f"{label} has blank image_id values at rows {copy.index[blank].tolist()[:5]}")
    duplicates = sorted(
        copy.loc[copy.image_id.duplicated(keep=False), "image_id"].unique().tolist()
    )
    if duplicates:
        raise ValueError(f"{label} has duplicate image_id values: {duplicates[:5]}")
    return {row.image_id: row for _, row in copy.iterrows()}


def build_pair_evidence(images: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Build evidence for every unordered pair in stable image-ID order.

    Args:
        images: Task 2 image metadata table.
        features: Task 3 feature table with exactly the same image ID set.

    Returns:
        A DataFrame with :data:`PAIR_EVIDENCE_COLUMNS`. For ``N`` images it
        contains exactly ``N(N-1)/2`` rows; zero or one image produces a valid
        empty table with headers. Inputs are not mutated, and row order is
        deterministic regardless of their original order.

    Raises:
        ValueError: For bad schemas/IDs, metadata-feature ID mismatch, stale
            feature paths, or invalid pair features.
    """
    metadata_by_id = _indexed_rows(images, "images", IMAGE_TABLE_COLUMNS)
    features_by_id = _indexed_rows(features, "features", IMAGE_FEATURE_COLUMNS)
    metadata_ids, feature_ids = set(metadata_by_id), set(features_by_id)
    if metadata_ids != feature_ids:
        missing = sorted(metadata_ids - feature_ids)
        extra = sorted(feature_ids - metadata_ids)
        raise ValueError(
            f"features image IDs do not match images: missing={missing[:5]}, extra={extra[:5]}"
        )

    ordered_ids = sorted(metadata_ids)
    for image_id in ordered_ids:
        metadata_path = str(metadata_by_id[image_id]["path"])
        feature_path = str(features_by_id[image_id]["path"])
        if metadata_path != feature_path:
            raise ValueError(
                f"stale feature path for image_id={image_id!r}: "
                f"images path={metadata_path!r}, features path={feature_path!r}"
            )

    rows = [
        compare_pair(
            features_by_id[id_a],
            features_by_id[id_b],
            metadata_by_id[id_a],
            metadata_by_id[id_b],
        )
        for id_a, id_b in combinations(ordered_ids, 2)
    ]
    return pd.DataFrame(rows, columns=PAIR_EVIDENCE_COLUMNS)
