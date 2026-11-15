"""Reproducible internal development examples and ablations for T05.

This module is development tooling only. It is never imported by the production
pipeline, never reads instructor ground truth, and labels every metric as
internal evidence rather than official evaluation.

Example:

    python -m tools.t05_experiments --sources dev_sources.csv --output out/t05_dev
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.io import load_config, sha256_file, sha256_text, write_json  # noqa: E402
from src.metrics.pairs import pair_prf, residual_cross_split_leak  # noqa: E402
from src.t05 import (  # noqa: E402
    DEFAULT_PARAMS,
    PREDICTION_COLUMNS,
    SPLIT_GROUP_COLUMNS,
    assign_group_splits,
    assign_random_splits,
    build_duplicate_groups,
    build_pair_evidence,
    build_split_groups,
    canonical_pair,
    classify_pairs,
    expand_group_pairs,
    extract_features,
    score_duplicate_evidence,
    validate_params,
    verify_visual_pairs,
)


DEVELOPMENT_IMAGE_COLUMNS = (
    "image_id",
    "path",
    "source_id",
    "scene_id",
    "variant",
    "partition",
    "camera_id",
    "captured_at",
)
DEVELOPMENT_LABEL_COLUMNS = (
    "img_a",
    "img_b",
    "is_duplicate",
    "same_scene",
    "scene_id_a",
    "scene_id_b",
    "source_id_a",
    "source_id_b",
    "partition",
    "case_type",
)

_EXPERIMENT_DEFAULTS = {
    "output_dir": "out/t05_development",
    "source_root": ".",
    "seed": 42,
    "heldout_fraction": 0.3,
    "jpeg_quality": 70,
    "brightness_factor": 1.15,
    "crop_fraction": 0.08,
    "evaluation_partition": "dev",
    "bootstrap_iterations": 200,
}
_EXPERIMENT_KEYS = frozenset(_EXPERIMENT_DEFAULTS) | {"t05_params"}
_DEVELOPMENT_OUTPUTS = (
    "development_images.csv",
    "development_labels.csv",
    "development_cases.csv",
    "frozen_thresholds.json",
    "ablation_results.csv",
    "failure_evidence.csv",
    "ablation_protocol.json",
)


def _normalize_build_params(params: Mapping[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]]:
    if params is None:
        supplied: dict[str, Any] = {}
    elif isinstance(params, Mapping):
        supplied = dict(params)
    else:
        raise ValueError(f"params must be a mapping or None, got {type(params).__name__}")
    t05_names = set(DEFAULT_PARAMS)
    unknown = sorted(set(supplied) - _EXPERIMENT_KEYS - t05_names)
    if unknown:
        raise ValueError(f"unknown development parameters: {unknown}")
    cfg = {**_EXPERIMENT_DEFAULTS, **{key: value for key, value in supplied.items() if key in _EXPERIMENT_KEYS}}
    t05_raw = dict(supplied.get("t05_params") or {})
    t05_raw.update({key: supplied[key] for key in t05_names if key in supplied})
    t05_params = validate_params(t05_raw)

    if isinstance(cfg["seed"], bool) or not isinstance(cfg["seed"], (int, np.integer)):
        raise ValueError("seed must be an integer")
    cfg["seed"] = int(cfg["seed"])
    for name in ("heldout_fraction", "brightness_factor", "crop_fraction"):
        value = cfg[name]
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            raise ValueError(f"{name} must be a finite number")
        cfg[name] = float(value)
        if not math.isfinite(cfg[name]):
            raise ValueError(f"{name} must be a finite number")
    if not 0 < cfg["heldout_fraction"] < 1:
        raise ValueError("heldout_fraction must be in (0,1)")
    if not 0.1 <= cfg["brightness_factor"] <= 3.0:
        raise ValueError("brightness_factor must be in [0.1,3.0]")
    if not 0 < cfg["crop_fraction"] < 0.25:
        raise ValueError("crop_fraction must be in (0,0.25)")
    quality = cfg["jpeg_quality"]
    if isinstance(quality, bool) or not isinstance(quality, (int, np.integer)) or not 1 <= int(quality) <= 95:
        raise ValueError("jpeg_quality must be an integer in [1,95]")
    cfg["jpeg_quality"] = int(quality)
    iterations = cfg["bootstrap_iterations"]
    if (
        isinstance(iterations, bool)
        or not isinstance(iterations, (int, np.integer))
        or int(iterations) < 0
    ):
        raise ValueError("bootstrap_iterations must be a nonnegative integer")
    cfg["bootstrap_iterations"] = int(iterations)
    if cfg["evaluation_partition"] not in {"dev", "heldout", "all"}:
        raise ValueError("evaluation_partition must be dev, heldout, or all")
    cfg["output_dir"] = Path(cfg["output_dir"])
    cfg["source_root"] = Path(cfg["source_root"])
    return cfg, t05_params


def _validated_sources(source_images: pd.DataFrame, source_root: Path) -> pd.DataFrame:
    if not isinstance(source_images, pd.DataFrame):
        raise ValueError(
            f"source_images must be a pandas DataFrame, got {type(source_images).__name__}"
        )
    required = ("image_id", "path", "scene_id")
    missing = [column for column in required if column not in source_images.columns]
    if missing:
        raise ValueError(f"source_images is missing required columns {missing}")
    if source_images.empty:
        raise ValueError("source_images must contain at least one row")
    out = source_images.copy()
    for column in required:
        if out[column].isna().any():
            raise ValueError(f"source_images.{column} values must be non-null")
        out[column] = out[column].astype(str).str.strip()
        if (out[column] == "").any():
            raise ValueError(f"source_images.{column} values must be nonblank")
    if out.image_id.duplicated().any():
        raise ValueError("source_images.image_id values must be unique")
    paths = []
    for row in out.itertuples(index=False):
        candidate = Path(row.path)
        candidate = candidate if candidate.is_absolute() else source_root / candidate
        try:
            candidate = candidate.resolve(strict=True)
        except OSError as exc:
            raise ValueError(f"source image {row.image_id!r} is missing: {candidate}") from exc
        if not candidate.is_file():
            raise ValueError(f"source image {row.image_id!r} is not a file: {candidate}")
        try:
            with Image.open(candidate) as image:
                image.convert("RGB").load()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise ValueError(f"source image {row.image_id!r} cannot be decoded: {exc}") from exc
        paths.append(candidate)
    out["resolved_path"] = paths
    if len(set(paths)) != len(paths):
        raise ValueError("source_images must not map multiple IDs to one file")
    if "camera_id" not in out.columns:
        out["camera_id"] = [f"dev-camera-{index}" for index in range(len(out))]
    if "captured_at" not in out.columns:
        out["captured_at"] = pd.date_range("2026-01-01", periods=len(out), freq="10min", tz="UTC")
    else:
        timestamps = pd.to_datetime(out.captured_at, format="ISO8601", errors="coerce", utc=True)
        if timestamps.isna().any():
            raise ValueError("source_images.captured_at contains invalid timestamps")
        out["captured_at"] = timestamps
    out["camera_id"] = out.camera_id.astype(str).str.strip()
    if (out.camera_id == "").any():
        raise ValueError("source_images.camera_id values must be nonblank")
    return out.sort_values("image_id", kind="mergesort", ignore_index=True)


def _scene_partitions(scenes: list[str], heldout_fraction: float, seed: int) -> dict[str, str]:
    ordered = sorted(set(scenes))
    if len(ordered) < 2:
        return {scene: "dev" for scene in ordered}
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(ordered))
    n_heldout = max(1, min(len(ordered) - 1, int(round(len(ordered) * heldout_fraction))))
    heldout = {ordered[int(index)] for index in permutation[:n_heldout]}
    return {scene: "heldout" if scene in heldout else "dev" for scene in ordered}


def _visual_signature(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        resized = image.convert("RGB").resize((16, 16), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / np.float32(255.0)


def _closest_cross_scene_pair(sources: pd.DataFrame) -> frozenset[str] | None:
    signatures = {
        row.image_id: _visual_signature(row.resolved_path)
        for row in sources.itertuples(index=False)
    }
    candidates = []
    for left, right in combinations(sources.itertuples(index=False), 2):
        if left.scene_id == right.scene_id:
            continue
        distance = float(np.mean(np.abs(signatures[left.image_id] - signatures[right.image_id])))
        candidates.append((distance, left.image_id, right.image_id))
    if not candidates:
        return None
    _, img_a, img_b = min(candidates)
    return frozenset((img_a, img_b))


def _variant_image_id(source_id: str, variant: str) -> str:
    return f"dev:{len(source_id)}:{source_id}:{variant}"


def _clear_development_outputs(output_dir: Path, sources: pd.DataFrame) -> Path:
    """Clear only files managed by this development tool."""
    image_dir = output_dir / "images"
    resolved_image_dir = image_dir.resolve(strict=False)
    for source_path in sources.resolved_path:
        if Path(source_path).is_relative_to(resolved_image_dir):
            raise ValueError("output_dir/images must not contain source images")
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    for path in image_dir.iterdir():
        if path.is_dir():
            raise ValueError(f"development image output contains unexpected directory: {path}")
        path.unlink()
    for name in _DEVELOPMENT_OUTPUTS:
        path = output_dir / name
        if path.is_dir():
            raise ValueError(f"managed development output path is a directory: {path}")
        if path.exists():
            path.unlink()
    return image_dir


def build_development_examples(
    source_images: pd.DataFrame,
    params: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Generate deterministic internal T05 examples and explicit pair labels.

    ``source_images`` requires ``image_id,path,scene_id`` and may include
    ``camera_id,captured_at``. ``params`` controls output directory, source root,
    seed, heldout scene fraction, transform strengths, bootstrap count, and a
    nested/flat frozen T05 parameter mapping.

    For every source, the function writes an original copy, byte-exact copy,
    JPEG recompression, brightness variant, and modest crop-resize variant.
    Duplicate truth is based only on common source ID. Distinct sources in one
    declared scene are same-scene nonduplicates; the visually closest declared
    cross-scene source pair is logged as a hard negative. Entire scene groups,
    including all variants, are assigned together to dev or heldout.

    Returns a mapping containing generated ``images``, exhaustive ``labels``,
    ``pair_evidence``, in-memory ``features``, transform/case summaries, frozen
    parameters, and output paths. Metrics from these labels are internal only.

    Raises:
        ValueError: For malformed source metadata/parameters or unsafe outputs.
    """
    cfg, t05_params = _normalize_build_params(params)
    sources = _validated_sources(source_images, cfg["source_root"])
    output_dir = cfg["output_dir"]
    image_dir = _clear_development_outputs(output_dir, sources)
    partitions = _scene_partitions(
        sources.scene_id.tolist(), cfg["heldout_fraction"], cfg["seed"]
    )
    closest_cross_scene = _closest_cross_scene_pair(sources)

    rows: list[dict[str, Any]] = []
    for index, source in enumerate(sources.itertuples(index=False)):
        safe_base = f"src_{index:04d}_{sha256_text(source.image_id)[:10]}"
        with Image.open(source.resolved_path) as opened:
            rgb = opened.convert("RGB")
            rgb.load()
        suffix = source.resolved_path.suffix.lower() or ".img"
        paths = {
            "original": image_dir / f"{safe_base}__original{suffix}",
            "exact": image_dir / f"{safe_base}__exact{suffix}",
            "jpeg": image_dir / f"{safe_base}__jpeg_q{cfg['jpeg_quality']}.jpg",
            "brightness": image_dir / f"{safe_base}__brightness.png",
            "crop": image_dir / f"{safe_base}__crop.png",
        }
        shutil.copyfile(source.resolved_path, paths["original"])
        shutil.copyfile(source.resolved_path, paths["exact"])
        rgb.save(paths["jpeg"], format="JPEG", quality=cfg["jpeg_quality"], subsampling=0)
        ImageEnhance.Brightness(rgb).enhance(cfg["brightness_factor"]).save(paths["brightness"])
        width, height = rgb.size
        crop_x = max(1, int(round(width * cfg["crop_fraction"]))) if width > 2 else 0
        crop_y = max(1, int(round(height * cfg["crop_fraction"]))) if height > 2 else 0
        if width - 2 * crop_x < 1 or height - 2 * crop_y < 1:
            raise ValueError(f"source image {source.image_id!r} is too small for crop transform")
        cropped = rgb.crop((crop_x, crop_y, width - crop_x, height - crop_y)).resize(
            (width, height), Image.Resampling.BICUBIC
        )
        cropped.save(paths["crop"])

        for variant, path in paths.items():
            rows.append(
                {
                    "image_id": _variant_image_id(source.image_id, variant),
                    "path": path.relative_to(output_dir).as_posix(),
                    "source_id": source.image_id,
                    "scene_id": source.scene_id,
                    "variant": variant,
                    "partition": partitions[source.scene_id],
                    "camera_id": source.camera_id,
                    "captured_at": pd.Timestamp(source.captured_at),
                }
            )
    images = pd.DataFrame(rows, columns=DEVELOPMENT_IMAGE_COLUMNS).sort_values(
        "image_id", kind="mergesort", ignore_index=True
    )

    metadata = images.set_index("image_id")
    hash_of = {
        row.image_id: sha256_file(output_dir / row.path)
        for row in images.itertuples(index=False)
    }
    label_rows = []
    for img_a, img_b in combinations(images.image_id.tolist(), 2):
        left, right = metadata.loc[img_a], metadata.loc[img_b]
        duplicate = left.source_id == right.source_id
        same_scene = left.scene_id == right.scene_id
        partition = (
            left.partition if left.partition == right.partition else "excluded_cross_partition"
        )
        if duplicate:
            if hash_of[img_a] == hash_of[img_b]:
                case_type = "exact_copy"
            elif "crop" in {left.variant, right.variant}:
                case_type = "modest_crop"
            elif "brightness" in {left.variant, right.variant}:
                case_type = "brightness_change"
            elif "jpeg" in {left.variant, right.variant}:
                case_type = "jpeg_recompression"
            else:
                case_type = "transformed_duplicate"
        elif same_scene:
            case_type = "same_scene_nonduplicate"
        elif closest_cross_scene == frozenset((left.source_id, right.source_id)):
            case_type = "visually_similar_different_scene"
        else:
            case_type = "different_scene_negative"
        label_rows.append(
            {
                "img_a": img_a,
                "img_b": img_b,
                "is_duplicate": int(duplicate),
                "same_scene": int(same_scene),
                "scene_id_a": left.scene_id,
                "scene_id_b": right.scene_id,
                "source_id_a": left.source_id,
                "source_id_b": right.source_id,
                "partition": partition,
                "case_type": case_type,
            }
        )
    labels = pd.DataFrame(label_rows, columns=DEVELOPMENT_LABEL_COLUMNS)
    feature_input = images[["image_id", "path", "camera_id", "captured_at"]].copy()
    features = extract_features(feature_input, output_dir, t05_params)
    pair_evidence = verify_visual_pairs(
        feature_input, output_dir, build_pair_evidence(feature_input, features), t05_params
    )
    case_summary = (
        labels.assign(eligible=labels.partition != "excluded_cross_partition")
        .groupby("case_type", sort=True)
        .agg(n_pairs=("img_a", "size"), eligible_pairs=("eligible", "sum"))
        .reset_index()
    )

    frozen = {
        "scope": "internal_development_only_not_official",
        "seed": cfg["seed"],
        "heldout_fraction": cfg["heldout_fraction"],
        "jpeg_quality": cfg["jpeg_quality"],
        "brightness_factor": cfg["brightness_factor"],
        "crop_fraction": cfg["crop_fraction"],
        "evaluation_partition": cfg["evaluation_partition"],
        "bootstrap_iterations": cfg["bootstrap_iterations"],
        "t05_params": t05_params,
    }
    images.to_csv(output_dir / "development_images.csv", index=False)
    labels.to_csv(output_dir / "development_labels.csv", index=False)
    case_summary.to_csv(output_dir / "development_cases.csv", index=False)
    write_json(output_dir / "frozen_thresholds.json", frozen)
    return {
        "images": images,
        "labels": labels,
        "features": features,
        "pair_evidence": pair_evidence,
        "case_summary": case_summary,
        "params": frozen,
        "t05_params": t05_params,
        "output_dir": output_dir,
        "evaluation_partition": cfg["evaluation_partition"],
        "bootstrap_iterations": cfg["bootstrap_iterations"],
        "seed": cfg["seed"],
        "scope": "internal_development_only_not_official",
    }


def _binary_series(values: pd.Series, label: str) -> pd.Series:
    try:
        numeric = pd.to_numeric(values, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain only 0/1") from exc
    if numeric.isna().any() or not numeric.isin([0, 1]).all():
        raise ValueError(f"{label} must contain only 0/1")
    return numeric.astype(int)


def _validated_labels(labels: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(labels, pd.DataFrame):
        raise ValueError(f"labels must be a pandas DataFrame, got {type(labels).__name__}")
    required = ("img_a", "img_b", "is_duplicate", "same_scene")
    missing = [column for column in required if column not in labels.columns]
    if missing:
        raise ValueError(f"labels is missing required columns {missing}")
    out = labels.copy()
    seen = set()
    for index, row in out.iterrows():
        pair = canonical_pair(row.img_a, row.img_b)
        if pair != (str(row.img_a), str(row.img_b)):
            raise ValueError(f"labels pair is not canonical at row {index}: {pair!r}")
        if pair in seen:
            raise ValueError(f"labels contains duplicate pair {pair!r}")
        seen.add(pair)
    out["is_duplicate"] = _binary_series(out.is_duplicate, "labels.is_duplicate")
    out["same_scene"] = _binary_series(out.same_scene, "labels.same_scene")
    if ((out.is_duplicate == 1) & (out.same_scene == 0)).any():
        raise ValueError("duplicate labels must also be same-scene labels")
    if {"scene_id_a", "scene_id_b"}.issubset(out.columns):
        for column in ("scene_id_a", "scene_id_b"):
            if out[column].isna().any():
                raise ValueError(f"labels.{column} values must be non-null")
            out[column] = out[column].astype(str).str.strip()
            if (out[column] == "").any():
                raise ValueError(f"labels.{column} values must be nonblank")
    return out


def _validated_development_predictions(
    predictions: pd.DataFrame,
    known_pairs: set[tuple[str, str]],
) -> tuple[pd.DataFrame, set[tuple[str, str]]]:
    if not isinstance(predictions, pd.DataFrame):
        raise ValueError(
            f"predictions must be a pandas DataFrame, got {type(predictions).__name__}"
        )
    missing = [column for column in ("img_a", "img_b") if column not in predictions.columns]
    if missing:
        raise ValueError(f"predictions is missing required columns {missing}")
    out = predictions.copy()
    pairs = []
    seen = set()
    for index, row in out.iterrows():
        pair = canonical_pair(row.img_a, row.img_b)
        if pair != (str(row.img_a), str(row.img_b)):
            raise ValueError(f"predictions pair is not canonical at row {index}: {pair!r}")
        if pair in seen:
            raise ValueError(f"predictions contains duplicate pair {pair!r}")
        if pair not in known_pairs:
            raise ValueError(f"predictions contains pair outside fixed candidate pool: {pair!r}")
        seen.add(pair)
        pairs.append(pair)
    out["_pair"] = pairs
    if "pred_dup" in out.columns:
        out["pred_dup"] = _binary_series(out.pred_dup, "predictions.pred_dup")
        positive = set(out.loc[out.pred_dup == 1, "_pair"])
    else:
        positive = set(out._pair)
    if "automatic_pred_dup" in out.columns:
        out["automatic_pred_dup"] = _binary_series(
            out.automatic_pred_dup, "predictions.automatic_pred_dup"
        )
    return out, positive


def _metric_values(
    frame: pd.DataFrame,
    positive_pairs: set[tuple[str, str]],
    weight: np.ndarray | None = None,
) -> dict[str, Any]:
    pred = np.array(
        [(row.img_a, row.img_b) in positive_pairs for row in frame.itertuples(index=False)],
        dtype=bool,
    )
    truth = frame.is_duplicate.to_numpy(dtype=bool)
    weights = np.ones(len(frame), dtype=float) if weight is None else weight.astype(float)
    tp = float(weights[pred & truth].sum())
    fp = float(weights[pred & ~truth].sum())
    fn = float(weights[~pred & truth].sum())
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def _cluster_bootstrap(
    frame: pd.DataFrame,
    positive_pairs: set[tuple[str, str]],
    iterations: int,
    seed: int,
    confidence: float,
) -> dict[str, Any]:
    required = {"scene_id_a", "scene_id_b"}
    if not required.issubset(frame.columns) or iterations == 0 or frame.empty:
        return {
            "unit": "scene_group",
            "iterations": iterations,
            "available": False,
            "reason": "missing scene IDs, zero iterations, or empty evaluation pool",
            "f1_ci": [None, None],
        }
    scenes = sorted(set(frame.scene_id_a.astype(str)) | set(frame.scene_id_b.astype(str)))
    rng = np.random.default_rng(seed)
    f1_values = []
    for _ in range(iterations):
        counts = Counter(rng.choice(scenes, size=len(scenes), replace=True).tolist())
        weights = np.array(
            [
                counts[str(row.scene_id_a)]
                if str(row.scene_id_a) == str(row.scene_id_b)
                else counts[str(row.scene_id_a)] * counts[str(row.scene_id_b)]
                for row in frame.itertuples(index=False)
            ],
            dtype=float,
        )
        if weights.sum() == 0:
            continue
        f1_values.append(_metric_values(frame, positive_pairs, weights)["f1"])
    if not f1_values:
        return {
            "unit": "scene_group",
            "iterations": iterations,
            "available": False,
            "reason": "all resamples produced an empty candidate pool",
            "f1_ci": [None, None],
        }
    alpha = (1.0 - confidence) / 2.0
    return {
        "unit": "scene_group",
        "iterations": iterations,
        "available": True,
        "confidence": confidence,
        "effective_iterations": len(f1_values),
        "f1_ci": [
            float(np.quantile(f1_values, alpha)),
            float(np.quantile(f1_values, 1.0 - alpha)),
        ],
    }


def evaluate_development_run(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    split: pd.DataFrame | Mapping[str, Any] | str | None,
) -> dict[str, Any]:
    """Evaluate one run against explicit internal labels on a fixed pair pool.

    ``predictions`` requires canonical ``img_a,img_b`` and optionally binary
    ``pred_dup`` (without it, listed rows are positive like the official T05
    evaluator). Missing candidate rows are negative. ``labels`` requires
    canonical unique pairs plus binary ``is_duplicate,same_scene`` and may add
    partition, scene IDs, and case type. ``split`` may be a train/test/drop
    DataFrame, a partition name, or a mapping with ``assignments``, ``partition``,
    ``seed``, ``bootstrap_iterations``, ``confidence``, and runtime fields.

    Reports pair precision/recall/F1, cross-scene false merges, retention, split
    balance, residual labelled duplicate/scene leakage, per-case slices, optional
    automatic-to-reviewed effect, and scene-cluster bootstrap uncertainty. It
    never resamples correlated pairs independently and always marks results as
    internal development evidence.
    """
    normalized_labels = _validated_labels(labels)
    known_pairs = set(
        normalized_labels[["img_a", "img_b"]].itertuples(index=False, name=None)
    )
    normalized_predictions, positive_pairs = _validated_development_predictions(
        predictions, known_pairs
    )

    options: dict[str, Any] = {}
    assignments: pd.DataFrame | None = None
    if split is None:
        pass
    elif isinstance(split, pd.DataFrame):
        assignments = split
    elif isinstance(split, str):
        options["partition"] = split
    elif isinstance(split, Mapping):
        options = dict(split)
        assignments = options.get("assignments")
    else:
        raise ValueError("split must be a DataFrame, mapping, partition string, or None")
    partition = str(options.get("partition", "all"))
    if partition not in {"dev", "heldout", "all"}:
        raise ValueError("split partition must be dev, heldout, or all")
    pool = normalized_labels.copy()
    if "partition" in pool.columns:
        pool = pool[pool.partition != "excluded_cross_partition"]
        if partition != "all":
            pool = pool[pool.partition == partition]
    pool = pool.reset_index(drop=True)
    pool_pairs = set(pool[["img_a", "img_b"]].itertuples(index=False, name=None))
    selected_positive = positive_pairs & pool_pairs
    metrics = _metric_values(pool, selected_positive)
    official_style = pair_prf(selected_positive, set(
        pool.loc[pool.is_duplicate == 1, ["img_a", "img_b"]].itertuples(index=False, name=None)
    ))
    if metrics["f1"] != official_style["f1"]:
        raise AssertionError("internal pair metric diverged from shared unordered-pair contract")

    predicted_mask = np.array(
        [(row.img_a, row.img_b) in selected_positive for row in pool.itertuples(index=False)]
    )
    false_merge_count = int((predicted_mask & (pool.same_scene.to_numpy() == 0)).sum())
    n_predicted = len(selected_positive)
    result: dict[str, Any] = {
        "scope": "internal_development_only_not_official",
        "partition": partition,
        "n_candidate_pairs": len(pool),
        "n_labelled_duplicate_pairs": int(pool.is_duplicate.sum()),
        "n_predicted_positive_pairs": n_predicted,
        "tp": int(metrics["tp"]),
        "fp": int(metrics["fp"]),
        "fn": int(metrics["fn"]),
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "false_merge_count": false_merge_count,
        "false_merge_rate": false_merge_count / n_predicted if n_predicted else None,
    }

    image_ids = sorted(set(pool.img_a) | set(pool.img_b))
    if assignments is not None:
        if not isinstance(assignments, pd.DataFrame):
            raise ValueError("split.assignments must be a DataFrame")
        missing = [column for column in ("image_id", "split") if column not in assignments.columns]
        if missing:
            raise ValueError(f"split assignments is missing required columns {missing}")
        normalized_split = assignments[["image_id", "split"]].copy()
        normalized_split["image_id"] = normalized_split.image_id.astype(str)
        normalized_split["split"] = normalized_split.split.astype(str)
        if normalized_split.image_id.duplicated().any():
            raise ValueError("split assignments image_id values must be unique")
        unknown_splits = sorted(set(normalized_split.split) - {"train", "test", "drop"})
        if unknown_splits:
            raise ValueError(f"split assignments has invalid values {unknown_splits}")
        split_of = dict(zip(normalized_split.image_id, normalized_split.split))
        missing_ids = sorted(set(image_ids) - set(split_of))
        if missing_ids:
            raise ValueError(f"split assignments misses evaluated images: {missing_ids[:5]}")
        kept = [image_id for image_id in image_ids if split_of[image_id] != "drop"]
        n_test = sum(split_of[image_id] == "test" for image_id in image_ids)
        n_train = sum(split_of[image_id] == "train" for image_id in image_ids)
        duplicate_pairs = set(
            pool.loc[pool.is_duplicate == 1, ["img_a", "img_b"]].itertuples(
                index=False, name=None
            )
        )
        scene_pairs = set(
            pool.loc[pool.same_scene == 1, ["img_a", "img_b"]].itertuples(
                index=False, name=None
            )
        )
        duplicate_leak = residual_cross_split_leak(split_of, duplicate_pairs)
        scene_leak = residual_cross_split_leak(split_of, scene_pairs)
        result.update(
            {
                "n_images": len(image_ids),
                "n_train": n_train,
                "n_test": n_test,
                "n_drop": len(image_ids) - len(kept),
                "retention_fraction": len(kept) / len(image_ids) if image_ids else 0.0,
                "actual_test_fraction": n_test / len(kept) if kept else 0.0,
                "residual_duplicate_leak_rate": duplicate_leak["value"],
                "residual_duplicate_leak_numerator": duplicate_leak["numerator"],
                "residual_duplicate_leak_denominator": duplicate_leak["denominator"],
                "residual_scene_leak_rate": scene_leak["value"],
                "residual_scene_leak_numerator": scene_leak["numerator"],
                "residual_scene_leak_denominator": scene_leak["denominator"],
            }
        )
    else:
        result.update(
            {
                "n_images": len(image_ids),
                "n_train": None,
                "n_test": None,
                "n_drop": None,
                "retention_fraction": None,
                "actual_test_fraction": None,
                "residual_duplicate_leak_rate": None,
                "residual_scene_leak_rate": None,
            }
        )

    runtime = options.get("runtime_seconds", predictions.attrs.get("runtime_seconds"))
    if runtime is not None:
        runtime = float(runtime)
        if not math.isfinite(runtime) or runtime < 0:
            raise ValueError("runtime_seconds must be finite and nonnegative")
    result["runtime_seconds"] = runtime
    result["runtime_per_image_seconds"] = runtime / len(image_ids) if runtime is not None and image_ids else None
    result["runtime_per_pair_seconds"] = runtime / len(pool) if runtime is not None and len(pool) else None

    if "automatic_pred_dup" in normalized_predictions.columns:
        automatic_positive = set(
            normalized_predictions.loc[
                normalized_predictions.automatic_pred_dup == 1, "_pair"
            ]
        ) & pool_pairs
        automatic_metrics = _metric_values(pool, automatic_positive)
        result["review_effect"] = {
            "changed_pair_count": len(automatic_positive ^ selected_positive),
            "automatic_f1": automatic_metrics["f1"],
            "final_f1": metrics["f1"],
            "f1_delta": metrics["f1"] - automatic_metrics["f1"],
        }
    else:
        result["review_effect"] = None

    slices = []
    if "case_type" in pool.columns:
        for case_type, subset in pool.groupby("case_type", sort=True):
            subset_pairs = set(
                subset[["img_a", "img_b"]].itertuples(index=False, name=None)
            )
            values = _metric_values(subset, selected_positive & subset_pairs)
            slices.append(
                {
                    "case_type": str(case_type),
                    "n_pairs": len(subset),
                    "n_positive_labels": int(subset.is_duplicate.sum()),
                    "precision": values["precision"],
                    "recall": values["recall"],
                    "f1": values["f1"],
                    "fp": int(values["fp"]),
                    "fn": int(values["fn"]),
                }
            )
    result["slices"] = slices
    iterations_value = options.get("bootstrap_iterations", 200)
    if (
        isinstance(iterations_value, bool)
        or not isinstance(iterations_value, (int, np.integer))
        or int(iterations_value) < 0
    ):
        raise ValueError("bootstrap_iterations must be nonnegative")
    iterations = int(iterations_value)
    confidence = float(options.get("confidence", 0.95))
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0,1)")
    result["uncertainty"] = _cluster_bootstrap(
        pool,
        selected_positive,
        iterations,
        int(options.get("seed", 42)),
        confidence,
    )
    return result


def _classified_for_method(
    evidence: pd.DataFrame,
    method: str,
    params: Mapping[str, Any],
) -> pd.DataFrame:
    standard = classify_pairs(score_duplicate_evidence(evidence, params), params)
    if method in {"combined_visual", "combined_visual_metadata"}:
        return standard
    out = standard.copy()
    if method == "exact_hash":
        positive = out.exact_match.astype(bool).to_numpy()
    elif method == "phash":
        positive = out.phash_distance.to_numpy(dtype=float) <= float(params["phash_distance_max"])
    elif method == "histogram":
        positive = out.histogram_distance.to_numpy(dtype=float) <= float(
            params["histogram_distance_max"]
        )
    else:
        raise ValueError(
            "method must be exact_hash, phash, histogram, combined_visual, "
            "combined_visual_metadata, or supplied"
        )
    out["duplicate_score"] = positive.astype(float)
    out["review_priority"] = 0.0
    out["classification"] = np.where(positive, "positive", "negative")
    out["pred_dup"] = positive.astype(int)
    out["reason_codes"] = f"development_ablation:{method}"
    return out


def _predictions_from_groups(classified: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    positives = set(
        expand_group_pairs(groups)[["img_a", "img_b"]].itertuples(index=False, name=None)
    )
    rows = [
        {
            "img_a": row.img_a,
            "img_b": row.img_b,
            "score": float(row.duplicate_score),
            "pred_dup": int((row.img_a, row.img_b) in positives),
        }
        for row in classified.itertuples(index=False)
    ]
    return pd.DataFrame(rows, columns=PREDICTION_COLUMNS)


def _duplicate_only_split_groups(groups: pd.DataFrame, maximum: int) -> pd.DataFrame:
    sizes = groups.groupby("duplicate_group_id").size().to_dict()
    rows = [
        {
            "image_id": row.image_id,
            "duplicate_group_id": row.duplicate_group_id,
            "split_group_id": f"split-from-{row.duplicate_group_id}",
            "split_group_size": sizes[row.duplicate_group_id],
            "exceeds_max_component_size": sizes[row.duplicate_group_id] > maximum,
        }
        for row in groups.itertuples(index=False)
    ]
    return pd.DataFrame(rows, columns=SPLIT_GROUP_COLUMNS)


def _normalize_configurations(
    configurations: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if isinstance(configurations, Mapping):
        rows = [{"name": name, **dict(config)} for name, config in configurations.items()]
    elif isinstance(configurations, Sequence) and not isinstance(configurations, (str, bytes)):
        rows = [dict(config) for config in configurations]
    else:
        raise ValueError("configurations must be a mapping or sequence of mappings")
    if not rows:
        raise ValueError("configurations must not be empty")
    names = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("each ablation configuration must be a mapping")
        name = str(row.get("name", "")).strip()
        if not name:
            raise ValueError("each ablation configuration requires a nonblank name")
        names.append(name)
        row["name"] = name
    if len(set(names)) != len(names):
        raise ValueError("ablation configuration names must be unique")
    return rows


def run_ablations(
    configurations: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
    dataset: Mapping[str, Any],
) -> pd.DataFrame:
    """Run fair internal T05 ablations on one fixed generated dataset.

    ``dataset`` is the mapping returned by :func:`build_development_examples`.
    Configurations require ``name`` and ``method``. Supported methods are
    ``exact_hash``, ``phash``, ``histogram``, ``combined_visual``,
    ``combined_visual_metadata``, and ``supplied`` (requires a predictions
    DataFrame). Optional ``params`` override frozen T05 settings;
    ``split_strategy`` may be ``independent``, ``duplicate_group``,
    ``scene_group``, or ``supplied``.

    Every method uses the same labelled candidate pool, partition, seed, metric
    implementation, and scene-cluster uncertainty protocol. The returned table
    includes pair quality, false merges, leakage/retention, runtime, and CIs.
    When an output directory is present, the function writes ablation results,
    a frozen configuration record, and explicit FP/FN failure evidence.
    """
    if not isinstance(dataset, Mapping):
        raise ValueError("dataset must be a mapping")
    required = {"images", "labels", "pair_evidence"}
    missing = sorted(required - set(dataset))
    if missing:
        raise ValueError(f"dataset is missing required fields {missing}")
    images = dataset["images"]
    labels = _validated_labels(dataset["labels"])
    evidence = dataset["pair_evidence"]
    base_params = validate_params(dataset.get("t05_params"))
    seed = int(dataset.get("seed", 42))
    partition = str(dataset.get("evaluation_partition", "all"))
    iterations = int(dataset.get("bootstrap_iterations", 200))
    configs = _normalize_configurations(configurations)
    image_input = images[["image_id", "camera_id", "captured_at"]].copy()

    rows = []
    failures = []
    frozen_configs = []
    for config in configs:
        started = time.perf_counter()
        method = str(config.get("method", "")).strip()
        parameter_overrides = config.get("params") or {}
        if not isinstance(parameter_overrides, Mapping):
            raise ValueError(f"ablation {config['name']!r} params must be a mapping")
        params = validate_params({**base_params, **dict(parameter_overrides)})
        default_strategy = {
            "exact_hash": "independent",
            "phash": "duplicate_group",
            "histogram": "duplicate_group",
            "combined_visual": "duplicate_group",
            "combined_visual_metadata": "scene_group",
            "supplied": "supplied",
        }.get(method)
        if default_strategy is None:
            raise ValueError(f"unknown ablation method {method!r}")
        split_strategy = str(config.get("split_strategy", default_strategy))

        if method == "supplied":
            if split_strategy != "supplied":
                raise ValueError("supplied ablations must use split_strategy='supplied'")
            predictions = config.get("predictions")
            if not isinstance(predictions, pd.DataFrame):
                raise ValueError(f"supplied ablation {config['name']!r} requires predictions DataFrame")
            assignments = config.get("assignments", dataset.get("split"))
            if not isinstance(assignments, pd.DataFrame):
                raise ValueError(f"supplied ablation {config['name']!r} requires assignments DataFrame")
        else:
            classified = _classified_for_method(evidence, method, params)
            duplicate_groups = build_duplicate_groups(image_input, classified)
            predictions = _predictions_from_groups(classified, duplicate_groups)
            if split_strategy == "independent":
                assignments = assign_random_splits(
                    image_input, float(params["test_fraction"]), np.random.default_rng(seed)
                )
            elif split_strategy == "duplicate_group":
                split_groups = _duplicate_only_split_groups(
                    duplicate_groups, int(params["max_component_size"])
                )
                assignments = assign_group_splits(
                    split_groups, float(params["test_fraction"]), np.random.default_rng(seed)
                )
            elif split_strategy == "scene_group":
                split_groups = build_split_groups(
                    image_input,
                    duplicate_groups,
                    classified,
                    params=params,
                )
                assignments = assign_group_splits(
                    split_groups, float(params["test_fraction"]), np.random.default_rng(seed)
                )
            else:
                raise ValueError(f"unknown split_strategy {split_strategy!r}")
        runtime = time.perf_counter() - started
        predictions = predictions.copy()
        predictions.attrs["runtime_seconds"] = runtime
        metrics = evaluate_development_run(
            predictions,
            labels,
            {
                "assignments": assignments,
                "partition": config.get("partition", partition),
                "seed": seed,
                "bootstrap_iterations": config.get("bootstrap_iterations", iterations),
                "runtime_seconds": runtime,
            },
        )
        ci = metrics["uncertainty"]["f1_ci"]
        rows.append(
            {
                "name": config["name"],
                "method": method,
                "split_strategy": split_strategy,
                "scope": metrics["scope"],
                "partition": metrics["partition"],
                "n_candidate_pairs": metrics["n_candidate_pairs"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "f1_ci_low": ci[0],
                "f1_ci_high": ci[1],
                "false_merge_rate": metrics["false_merge_rate"],
                "retention_fraction": metrics["retention_fraction"],
                "actual_test_fraction": metrics["actual_test_fraction"],
                "residual_duplicate_leak_rate": metrics["residual_duplicate_leak_rate"],
                "residual_scene_leak_rate": metrics["residual_scene_leak_rate"],
                "runtime_seconds": metrics["runtime_seconds"],
                "runtime_per_image_seconds": metrics["runtime_per_image_seconds"],
                "runtime_per_pair_seconds": metrics["runtime_per_pair_seconds"],
                "review_changed_pair_count": (
                    metrics["review_effect"]["changed_pair_count"]
                    if metrics["review_effect"] is not None
                    else None
                ),
                "review_f1_delta": (
                    metrics["review_effect"]["f1_delta"]
                    if metrics["review_effect"] is not None
                    else None
                ),
            }
        )
        predicted_flags = (
            predictions.pred_dup.astype(int)
            if "pred_dup" in predictions.columns
            else pd.Series(np.ones(len(predictions), dtype=int), index=predictions.index)
        )
        pred_map = dict(
            zip(
                predictions[["img_a", "img_b"]].itertuples(index=False, name=None),
                predicted_flags,
            )
        )
        eligible = labels[labels.partition != "excluded_cross_partition"] if "partition" in labels else labels
        selected_partition = str(config.get("partition", partition))
        if selected_partition != "all" and "partition" in eligible:
            eligible = eligible[eligible.partition == selected_partition]
        for label in eligible.itertuples(index=False):
            predicted = int(pred_map.get((label.img_a, label.img_b), 0))
            if predicted != int(label.is_duplicate):
                failures.append(
                    {
                        "name": config["name"],
                        "method": method,
                        "img_a": label.img_a,
                        "img_b": label.img_b,
                        "error_type": "false_positive" if predicted else "false_negative",
                        "case_type": getattr(label, "case_type", ""),
                        "scene_id_a": getattr(label, "scene_id_a", ""),
                        "scene_id_b": getattr(label, "scene_id_b", ""),
                    }
                )
        frozen_configs.append(
            {
                "name": config["name"],
                "method": method,
                "split_strategy": split_strategy,
                "partition": config.get("partition", partition),
                "seed": seed,
                "params": params,
            }
        )

    result = pd.DataFrame(rows)
    output_dir_value = dataset.get("output_dir")
    if output_dir_value is not None:
        output_dir = Path(output_dir_value)
        output_dir.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_dir / "ablation_results.csv", index=False)
        pd.DataFrame(
            failures,
            columns=(
                "name",
                "method",
                "img_a",
                "img_b",
                "error_type",
                "case_type",
                "scene_id_a",
                "scene_id_b",
            ),
        ).to_csv(output_dir / "failure_evidence.csv", index=False)
        write_json(
            output_dir / "ablation_protocol.json",
            {
                "scope": "internal_development_only_not_official",
                "fixed_candidate_pool": True,
                "bootstrap_unit": "scene_group",
                "configurations": frozen_configs,
            },
        )
    return result


def _default_ablations() -> list[dict[str, Any]]:
    return [
        {"name": "sha256_only", "method": "exact_hash"},
        {"name": "phash_only", "method": "phash"},
        {"name": "histogram_only", "method": "histogram"},
        {"name": "combined_visual", "method": "combined_visual"},
        {"name": "combined_visual_metadata", "method": "combined_visual_metadata"},
    ]


def main(argv: list[str] | None = None) -> int:
    """Run the internal-only generated-example and ablation CLI.

    Args:
        argv: Optional command-line arguments without the executable name. When
            omitted, ``argparse`` reads the process command line.

    Returns:
        Process exit code ``0`` after writing development artifacts and printing
        the ablation table. Results are labelled internal and are never official
        instructor-GT metrics.

    Raises:
        ValueError: If source rows, partition settings, configurations, or frozen
            T05 parameters are invalid.
        FileNotFoundError: If a source image, source CSV, or config is missing.

    The command is deterministic for fixed source bytes, configuration, partition,
    and seed. It writes only beneath ``--output`` and does not call production.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", required=True, help="CSV with image_id,path,scene_id")
    parser.add_argument("--output", required=True, help="development output directory")
    parser.add_argument("--source-root", help="base directory for relative source paths")
    parser.add_argument("--config", default="configs/lab.yaml", help="frozen T05 config YAML")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--partition", choices=("dev", "heldout", "all"), default="dev")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    source_path = Path(args.sources)
    source_root = Path(args.source_root) if args.source_root else source_path.parent
    dataset = build_development_examples(
        pd.read_csv(source_path, dtype=str),
        {
            "output_dir": args.output,
            "source_root": source_root,
            "seed": args.seed,
            "evaluation_partition": args.partition,
            "t05_params": config.get("params") or {},
        },
    )
    table = run_ablations(_default_ablations(), dataset)
    print("INTERNAL DEVELOPMENT EVIDENCE — NOT OFFICIAL T05 METRICS")
    print(table.to_string(index=False))
    print(f"wrote development artifacts to {Path(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
