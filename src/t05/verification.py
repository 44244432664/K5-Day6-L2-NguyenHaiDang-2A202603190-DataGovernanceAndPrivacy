"""Deterministic, model-free pixel verification for coarse T05 candidates."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from .contracts import validate_params
from .data import resolve_image_path


VERIFICATION_COLUMNS = ("verification_error", "verification_crop_fraction", "verification_texture")


def _views(path: Path, cfg: Mapping[str, int | float]) -> np.ndarray:
    with Image.open(path) as opened:
        image = opened.convert("RGB")
        width, height = image.size
        bank = []
        for fraction in np.linspace(0, cfg["verification_crop_max"], int(cfg["verification_crop_steps"]) + 1):
            x = min(int(round(width * fraction)), (width - 1) // 2)
            y = min(int(round(height * fraction)), (height - 1) // 2)
            view = image.crop((x, y, width - x, height - y)).resize(
                (int(cfg["verification_width"]), int(cfg["verification_height"])),
                Image.Resampling.BICUBIC,
            )
            bank.append(np.asarray(view, dtype=np.float32) / np.float32(255.0))
    return np.stack(bank)


def _errors(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    # One shared RGB gain tolerates exposure, not arbitrary per-channel color
    # remapping. Symmetrizing avoids image-order-dependent verification.
    gain = np.clip(
        np.sum(left * right, axis=(1, 2, 3))
        / np.maximum(np.sum(left * left, axis=(1, 2, 3)), 1e-6),
        0.625, 1.6,
    )
    return np.mean(np.abs(np.clip(left * gain[:, None, None, None], 0, 1) - right), axis=(1, 2, 3)) / np.maximum(
        np.mean(right, axis=(1, 2, 3)), 0.1
    )


def verify_visual_pairs(
    images: pd.DataFrame,
    input_dir: str | Path,
    pair_evidence: pd.DataFrame,
    params: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Add bounded crop/exposure RGB verification to canonical pair evidence.

    Inputs are validated public ``image_id,path`` rows and pair evidence with
    ``img_a,img_b``. Each image is decoded once into a bank of centered crop
    views. Full-versus-crop comparisons run in both directions; no two arbitrary
    small patches are matched. A single exposure gain is constrained to
    [0.625,1.6], so unrelated colors cannot be independently fitted per channel.

    Returns a copied table with ``verification_error`` (symmetric relative RGB
    MAE clipped to [0,1], lower is better), ``verification_crop_fraction`` (per-
    side centered crop fraction), and ``verification_texture`` (minimum spatial
    RGB standard deviation, [0,0.5]). Flat content cannot independently prove a
    nonexact duplicate. Fixed pixels/configuration give deterministic results,
    independent of table order. Disabled verification returns an unchanged copy
    for the explicitly labelled legacy ablation, not a hidden fallback.

    Raises:
        ValueError: For invalid parameters, table columns, duplicate/unknown IDs,
            or unsafe image paths.
        OSError: If an image cannot be opened or decoded.
    """
    cfg = validate_params(params)
    if not isinstance(images, pd.DataFrame) or not isinstance(pair_evidence, pd.DataFrame):
        raise ValueError("verification images and pair evidence must be DataFrames")
    out = pair_evidence.copy()
    if not cfg["verification_enabled"]:
        return out.drop(columns=list(VERIFICATION_COLUMNS), errors="ignore")
    if not {"image_id", "path"} <= set(images.columns) or not {"img_a", "img_b"} <= set(out.columns):
        raise ValueError("verification requires image_id,path and img_a,img_b columns")
    ids = images.image_id.astype(str)
    if images.image_id.isna().any() or ids.duplicated().any() or ids.str.strip().eq("").any():
        raise ValueError("verification requires unique nonblank image IDs")
    banks = {
        str(row.image_id): _views(resolve_image_path(input_dir, row.path), cfg)
        for row in images.sort_values("image_id").itertuples(index=False)
    }
    fractions = np.linspace(0, cfg["verification_crop_max"], int(cfg["verification_crop_steps"]) + 1)
    candidate_fractions = np.concatenate([fractions, fractions[1:]])
    rows = []
    for row in out.itertuples(index=False):
        if row.img_a not in banks or row.img_b not in banks:
            raise ValueError("verification pair references an unknown image ID")
        a, b = banks[row.img_a], banks[row.img_b]
        left = np.concatenate([a, np.repeat(a[:1], len(b) - 1, axis=0)])
        right = np.concatenate([np.repeat(b[:1], len(a), axis=0), b[1:]])
        errors = (_errors(left, right) + _errors(right, left)) * 0.5
        best = int(np.argmin(errors))
        texture = min(float(a[0].std(axis=(0, 1)).mean()), float(b[0].std(axis=(0, 1)).mean()))
        rows.append((float(np.clip(errors[best], 0, 1)), float(candidate_fractions[best]), texture))
    for index, column in enumerate(VERIFICATION_COLUMNS):
        out[column] = [row[index] for row in rows]
    return out
