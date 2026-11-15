"""Runnable T05 baseline: exact SHA-256 pairs plus random image split.

This baseline is deliberately not release-safe. It detects byte-identical files
only and assigns individual images independently to train/test, so related
images may cross the split. Its decision is always HOLD.
"""
from __future__ import annotations

import math
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

from ..common.context import RunContext
from ..common.decision import Decision, write_decision
from ..common.io import write_json
from .contracts import PREDICTION_COLUMNS, SPLIT_MANIFEST_COLUMNS, canonical_pair, validate_params
from .data import load_inputs
from .features import build_pair_evidence, extract_features


def predict_exact_pairs(pair_evidence: pd.DataFrame) -> pd.DataFrame:
    """Convert raw pair evidence into exact-hash baseline predictions.

    Every candidate pair is retained. ``pred_dup`` is one only when
    ``exact_match`` is true, and ``score`` is the corresponding 1.0/0.0. This
    explicit-negative representation is accepted by the T05 evaluator, which
    filters to ``pred_dup == 1`` before computing pair precision/recall/F1.

    The input DataFrame is not mutated. Pair IDs must be distinct, canonical,
    and free of self-pairs.
    """
    if not isinstance(pair_evidence, pd.DataFrame):
        raise ValueError(
            f"pair_evidence must be a pandas DataFrame, got {type(pair_evidence).__name__}"
        )
    required = ("pair_id", "img_a", "img_b", "exact_match")
    missing = [column for column in required if column not in pair_evidence.columns]
    if missing:
        raise ValueError(
            f"pair_evidence is missing required columns {missing}; has {list(pair_evidence.columns)}"
        )
    if pair_evidence["pair_id"].isna().any() or pair_evidence["pair_id"].astype(str).duplicated().any():
        raise ValueError("pair_evidence pair_id values must be non-null and unique")

    rows: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for row in pair_evidence.itertuples(index=False):
        pair = canonical_pair(row.img_a, row.img_b)
        if pair != (str(row.img_a), str(row.img_b)):
            raise ValueError(f"pair_evidence pair is not canonical: {(row.img_a, row.img_b)!r}")
        if pair in seen_pairs:
            raise ValueError(f"pair_evidence contains duplicate unordered pair {pair!r}")
        seen_pairs.add(pair)
        if not isinstance(row.exact_match, (bool, np.bool_)):
            raise ValueError(
                f"pair_evidence exact_match must be boolean for pair {pair!r}, got {row.exact_match!r}"
            )
        positive = int(bool(row.exact_match))
        rows.append(
            {
                "img_a": pair[0],
                "img_b": pair[1],
                "score": float(positive),
                "pred_dup": positive,
            }
        )
    return pd.DataFrame(rows, columns=PREDICTION_COLUMNS)


def assign_random_splits(
    images: pd.DataFrame,
    test_fraction: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Assign images independently to train/test using a supplied RNG.

    IDs are sorted before random selection, making results independent of input
    row order. For two or more images, the rounded target test count is clamped
    to ``[1, N-1]``. A singleton is assigned to train; an empty input returns a
    header-only table. No image is dropped and no duplicate/scene group is used.
    """
    if not isinstance(images, pd.DataFrame):
        raise ValueError(f"images must be a pandas DataFrame, got {type(images).__name__}")
    if "image_id" not in images.columns:
        raise ValueError(f"images is missing required column 'image_id'; has {list(images.columns)}")
    if isinstance(test_fraction, bool) or not isinstance(test_fraction, Real):
        raise ValueError(f"test_fraction must be a finite number in (0,1), got {test_fraction!r}")
    fraction = float(test_fraction)
    if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError(f"test_fraction must be a finite number in (0,1), got {test_fraction!r}")
    if not isinstance(rng, np.random.Generator):
        raise ValueError(f"rng must be numpy.random.Generator, got {type(rng).__name__}")
    if images.image_id.isna().any():
        raise ValueError("images has null image_id values")

    ids = images.image_id.astype(str)
    blank = ids.str.strip() == ""
    if blank.any():
        raise ValueError(f"images has blank image_id values at rows {images.index[blank].tolist()[:5]}")
    duplicates = sorted(ids[ids.duplicated(keep=False)].unique().tolist())
    if duplicates:
        raise ValueError(f"images has duplicate image_id values: {duplicates[:5]}")
    ordered_ids = sorted(ids.tolist())

    n_images = len(ordered_ids)
    if n_images < 2:
        n_test = 0
    else:
        n_test = max(1, min(n_images - 1, int(round(n_images * fraction))))
    permutation = rng.permutation(n_images)
    test_indices = set(int(i) for i in permutation[:n_test])
    rows = [
        {"image_id": image_id, "split": "test" if index in test_indices else "train"}
        for index, image_id in enumerate(ordered_ids)
    ]
    return pd.DataFrame(rows, columns=SPLIT_MANIFEST_COLUMNS)


def _diagnostics(
    images: pd.DataFrame,
    predictions: pd.DataFrame,
    split: pd.DataFrame,
    provenance: dict[str, Any],
    test_fraction: float,
    seed: int,
) -> dict[str, Any]:
    """Summarize observable baseline behavior without hidden labels."""
    split_of = dict(zip(split.image_id, split.split))
    positives = predictions[predictions.pred_dup == 1]
    cross_split = sum(
        split_of.get(row.img_a) != split_of.get(row.img_b)
        for row in positives.itertuples(index=False)
    )
    n_images = len(images)
    n_test = int((split.split == "test").sum())
    n_train = int((split.split == "train").sum())
    return {
        "mode": "baseline",
        "method": "exact_sha256_plus_independent_random_image_split",
        "seed": int(seed),
        "n_images": n_images,
        "n_pairs": len(predictions),
        "n_exact_positive_pairs": len(positives),
        "n_train": n_train,
        "n_test": n_test,
        "n_drop": 0,
        "target_test_fraction": float(test_fraction),
        "actual_test_fraction": n_test / n_images if n_images else 0.0,
        "exact_positive_cross_split_pairs": int(cross_split),
        "exact_positive_cross_split_rate": cross_split / len(positives) if len(positives) else 0.0,
        "provenance": provenance,
        "release_safe": False,
        "limitations": [
            "detects byte-identical files only; transformed near-duplicates are missed",
            "splits images independently and intentionally ignores duplicate/scene groups",
            "exact-pair split diagnostics are not official scene-leakage metrics",
        ],
    }


def run(ctx: RunContext) -> None:
    """Run the T05 exact-hash/random-split baseline and write its artifacts."""
    if ctx.track != "t05":
        raise NotImplementedError(f"src.t05.baseline supports t05 only (got {ctx.track})")

    params = validate_params(ctx.config.get("params"))
    loaded = load_inputs(
        ctx.input_dir,
        strict_provenance=ctx.config.get("strict_provenance", True),
    )
    images = loaded["images"]
    features = extract_features(images, ctx.input_dir, params)
    pair_evidence = build_pair_evidence(images, features)
    predictions = predict_exact_pairs(pair_evidence)
    split = assign_random_splits(images, float(params["test_fraction"]), ctx.rng)
    diagnostics = _diagnostics(
        images,
        predictions,
        split,
        loaded["provenance"],
        float(params["test_fraction"]),
        ctx.seed,
    )

    # The baseline exports explicit negatives for auditability; the evaluator
    # selects only pred_dup=1 rows when computing pair F1.
    pair_evidence.to_csv(ctx.out("pair_evidence.csv"), index=False, na_rep="")
    predictions.to_csv(ctx.out("automatic_pairs.csv"), index=False)
    predictions.to_csv(ctx.out("suspect_pairs.csv"), index=False)
    split.to_csv(ctx.out("split_manifest.csv"), index=False)
    write_json(ctx.out("diagnostics.json"), diagnostics)

    ctx.stats.update(
        {
            "n_items": len(images),
            "n_pairs": len(predictions),
            "n_exact_positive_pairs": diagnostics["n_exact_positive_pairs"],
            "n_train": diagnostics["n_train"],
            "n_test": diagnostics["n_test"],
            "n_drop": 0,
            "exact_positive_cross_split_pairs": diagnostics[
                "exact_positive_cross_split_pairs"
            ],
        }
    )
    decision = Decision(
        status="HOLD",
        reason_code="BASELINE_ONLY_NOT_GROUP_SAFE",
        reason=(
            "Exact-hash baseline misses transformed duplicates and uses an independent "
            "image split; do not release this split."
        ),
        owner=str(ctx.config.get("owner") or "baseline-owner"),
        scope=ctx.track,
        evidence=[
            str(ctx.out("suspect_pairs.csv")),
            str(ctx.out("split_manifest.csv")),
            str(ctx.out("diagnostics.json")),
        ],
        metrics={
            "n_exact_positive_pairs": diagnostics["n_exact_positive_pairs"],
            "exact_positive_cross_split_pairs": diagnostics[
                "exact_positive_cross_split_pairs"
            ],
            "actual_test_fraction": diagnostics["actual_test_fraction"],
        },
    )
    write_decision(decision, ctx.out("decision.json"), ctx.out("decision_log.csv"))
