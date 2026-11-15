"""Safe input loading and provenance validation for T05.

This module validates only the public input contract. It never discovers or
reads instructor ground truth, and it does not compute image features or make
duplicate predictions.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, UnidentifiedImageError

from ..common.manifest import MANIFEST_COLUMNS, verify_manifest


IMAGE_TABLE_COLUMNS = ("image_id", "path", "camera_id", "captured_at")


def resolve_image_path(input_dir: str | Path, relative_path: Any) -> Path:
    """Resolve one image path and require it to remain inside ``input_dir``.

    Args:
        input_dir: Existing T05 input directory.
        relative_path: Nonblank relative path from ``images.csv``.

    Returns:
        An absolute, normalized :class:`Path`. Symlinks are resolved, so a
        symlink that escapes the input root is rejected as well.

    Raises:
        ValueError: For blank, absolute, drive-qualified, or escaping paths.
        FileNotFoundError: If ``input_dir`` itself does not exist.
    """
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"T05 input directory not found: {root}")
    if not root.is_dir():
        raise ValueError(f"T05 input path is not a directory: {root}")
    root = root.resolve(strict=True)

    if relative_path is None:
        raise ValueError("image path must be a nonblank relative path, got None")
    text = str(relative_path)
    if not text.strip():
        raise ValueError("image path must be a nonblank relative path")
    if "\x00" in text:
        raise ValueError(f"image path contains a null byte: {text!r}")

    rel = Path(text)
    if rel.is_absolute() or rel.drive:
        raise ValueError(f"image path must be relative to {root}: {text!r}")

    candidate = (root / rel).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"image path escapes input directory: {text!r}") from exc
    return candidate


def _decode_image(path: Path, image_id: str) -> None:
    """Fully decode an image to reject truncated/corrupt files early."""
    try:
        with Image.open(path) as image:
            image.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(f"image {image_id!r} cannot be decoded at {path}: {exc}") from exc


def validate_image_table(images: pd.DataFrame, input_dir: str | Path) -> pd.DataFrame:
    """Validate and normalize the public T05 ``images.csv`` table.

    Required columns are ``image_id``, ``path``, ``camera_id``, and
    ``captured_at``. IDs, paths, and camera IDs become stripped strings;
    timestamps become timezone-aware UTC pandas values. ISO-8601 timestamps
    without an offset are deterministically interpreted as UTC.

    The caller's DataFrame is not mutated. Row order is preserved.

    Raises:
        ValueError: On an empty/malformed table, duplicate IDs or paths,
            unsafe/missing/non-file paths, invalid timestamps, or undecodable
            images.
    """
    if not isinstance(images, pd.DataFrame):
        raise ValueError(f"images must be a pandas DataFrame, got {type(images).__name__}")
    missing = [column for column in IMAGE_TABLE_COLUMNS if column not in images.columns]
    if missing:
        raise ValueError(f"images.csv is missing required columns {missing}; has {list(images.columns)}")
    if images.empty:
        raise ValueError("images.csv must contain at least one image")

    out = images.copy()
    for column in IMAGE_TABLE_COLUMNS:
        # read_csv(..., keep_default_na=False) normally supplies strings, but
        # direct callers still receive deterministic validation for null cells.
        null_rows = out.index[out[column].isna()].tolist()
        if null_rows:
            raise ValueError(f"images.csv column {column!r} has null values at rows {null_rows[:5]}")
        out[column] = out[column].astype(str).str.strip()
        blank_rows = out.index[out[column] == ""].tolist()
        if blank_rows:
            raise ValueError(f"images.csv column {column!r} has blank values at rows {blank_rows[:5]}")

    duplicate_ids = sorted(out.loc[out.image_id.duplicated(keep=False), "image_id"].unique())
    if duplicate_ids:
        raise ValueError(f"images.csv has duplicate image_id values: {duplicate_ids[:5]}")

    timestamps = pd.to_datetime(out["captured_at"], format="ISO8601", errors="coerce", utc=True)
    bad_timestamp_rows = out.index[timestamps.isna()].tolist()
    if bad_timestamp_rows:
        values = out.loc[bad_timestamp_rows[:5], "captured_at"].tolist()
        raise ValueError(
            f"images.csv has invalid ISO-8601 captured_at values at rows {bad_timestamp_rows[:5]}: {values}"
        )
    out["captured_at"] = timestamps

    root = Path(input_dir).resolve(strict=True)
    normalized_paths: list[str] = []
    resolved_paths: list[Path] = []
    for row in out.itertuples(index=False):
        path = resolve_image_path(root, row.path)
        if not path.exists():
            raise ValueError(f"image {row.image_id!r} file not found: {row.path!r}")
        if not path.is_file():
            raise ValueError(f"image {row.image_id!r} path is not a file: {row.path!r}")
        _decode_image(path, row.image_id)
        normalized_paths.append(path.relative_to(root).as_posix())
        resolved_paths.append(path)

    if len(set(resolved_paths)) != len(resolved_paths):
        duplicates = pd.Series(normalized_paths)[pd.Series(normalized_paths).duplicated(keep=False)].unique()
        raise ValueError(f"images.csv maps multiple image IDs to the same path: {sorted(duplicates)[:5]}")
    out["path"] = normalized_paths
    return out


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"cannot read {label} at {path}: {exc}") from exc


def _manifest_problems(manifest_path: Path, manifest: pd.DataFrame) -> list[dict[str, Any]]:
    """Validate manifest structure/paths before invoking the shared hash verifier."""
    problems: list[dict[str, Any]] = []
    missing = [column for column in MANIFEST_COLUMNS if column not in manifest.columns]
    if missing:
        return [{"problem": "schema", "detail": f"missing columns {missing}"}]
    if manifest.empty:
        return [{"problem": "empty_manifest"}]

    for column in ("file_id", "path", "source", "license", "sha256"):
        blank = manifest.index[manifest[column].astype(str).str.strip() == ""].tolist()
        if blank:
            problems.append(
                {"problem": "blank_manifest_value", "column": column, "rows": blank[:5]}
            )

    duplicate_ids = manifest.loc[manifest.file_id.duplicated(keep=False), "file_id"].unique().tolist()
    if duplicate_ids:
        problems.append({"problem": "duplicate_file_id", "values": sorted(duplicate_ids)[:5]})
    duplicate_paths = manifest.loc[manifest.path.duplicated(keep=False), "path"].unique().tolist()
    if duplicate_paths:
        problems.append({"problem": "duplicate_manifest_path", "values": sorted(duplicate_paths)[:5]})

    unsafe = []
    non_files = []
    for value in manifest["path"]:
        try:
            resolved = resolve_image_path(manifest_path.parent, value)
            if resolved.exists() and not resolved.is_file():
                non_files.append({"path": str(value), "detail": "manifest path is not a file"})
        except (ValueError, FileNotFoundError) as exc:
            unsafe.append({"path": str(value), "detail": str(exc)})
    if unsafe:
        problems.extend({"problem": "unsafe_manifest_path", **item} for item in unsafe[:20])
    if non_files:
        problems.extend({"problem": "non_file_manifest_path", **item} for item in non_files[:20])
    if unsafe or non_files:
        problems.append(
            {
                "problem": "hash_verification_skipped",
                "detail": "manifest contains unsafe or non-file paths",
            }
        )
    else:
        # The shared helper computes SHA-256 for every declared file. It is safe
        # to call only after all paths have been proven to stay under the root.
        try:
            problems.extend(verify_manifest(manifest_path))
        except OSError as exc:
            problems.append({"problem": "hash_verification_error", "detail": str(exc)})
    return problems


def _coverage_problems(
    images: pd.DataFrame, manifest: pd.DataFrame, input_dir: Path
) -> list[dict[str, Any]]:
    """Require one matching manifest record for images.csv and every image row."""
    if any(column not in manifest.columns for column in MANIFEST_COLUMNS):
        return []  # the manifest schema problem already explains why coverage is unavailable

    declared: set[tuple[str, Path]] = set()
    for row in manifest.itertuples(index=False):
        try:
            declared.add((str(row.file_id), resolve_image_path(input_dir, row.path)))
        except (ValueError, FileNotFoundError):
            continue

    problems: list[dict[str, Any]] = []
    metadata_path = (input_dir / "images.csv").resolve(strict=False)
    if ("images", metadata_path) not in declared:
        problems.append(
            {
                "problem": "manifest_coverage",
                "file_id": "images",
                "path": "images.csv",
            }
        )
    for row in images.itertuples(index=False):
        path = resolve_image_path(input_dir, row.path)
        if (str(row.image_id), path) not in declared:
            problems.append(
                {
                    "problem": "manifest_coverage",
                    "file_id": str(row.image_id),
                    "path": str(row.path),
                }
            )
    return problems


def load_inputs(input_dir: str | Path, strict_provenance: bool = True) -> dict[str, Any]:
    """Load and validate T05 public inputs and provenance diagnostics.

    Structural data errors (unsafe paths, invalid metadata, missing/corrupt image
    files) always raise. Provenance problems such as missing manifest coverage or
    checksum mismatches raise only when ``strict_provenance`` is true; otherwise
    they are returned for an explicit downstream HOLD/REWORK decision.

    Returns:
        ``{"images": DataFrame, "manifest": DataFrame, "provenance": dict}``.
        ``images.captured_at`` is UTC datetime data. ``provenance`` contains
        ``strict``, ``verified``, ``problems``, ``n_images``, and
        ``n_manifest_rows``.
    """
    if not isinstance(strict_provenance, bool):
        raise ValueError("strict_provenance must be a boolean")

    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"T05 input directory not found: {root}")
    if not root.is_dir():
        raise ValueError(f"T05 input path is not a directory: {root}")
    root = root.resolve(strict=True)

    images_path = root / "images.csv"
    if not images_path.is_file():
        raise FileNotFoundError(f"T05 images table not found: {images_path}")
    images = validate_image_table(_read_csv(images_path, "images.csv"), root)
    # Preserve the validated root as non-tabular metadata so offline review
    # rendering can resolve relative image paths without weakening the exported
    # images.csv contract.
    images.attrs["input_dir"] = str(root)

    manifest_path = root / "manifest.csv"
    if manifest_path.is_file():
        manifest = _read_csv(manifest_path, "manifest.csv")
        problems = _manifest_problems(manifest_path, manifest)
        problems.extend(_coverage_problems(images, manifest, root))
    else:
        manifest = pd.DataFrame(columns=MANIFEST_COLUMNS)
        problems = [{"problem": "missing_manifest", "path": str(manifest_path)}]

    provenance = {
        "strict": strict_provenance,
        "verified": not problems,
        "problems": problems,
        "n_images": len(images),
        "n_manifest_rows": len(manifest),
    }
    if strict_provenance and problems:
        raise ValueError(
            f"T05 provenance validation failed ({len(problems)} problems), first: {problems[:3]}"
        )
    return {"images": images, "manifest": manifest, "provenance": provenance}
