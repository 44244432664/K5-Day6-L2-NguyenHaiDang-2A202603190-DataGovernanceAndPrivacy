"""Validated, staged production-artifact writing for the T05 pipeline."""
from __future__ import annotations

import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pandas as pd

from ..common.decision import Decision, write_decision
from ..common.io import write_json
from .contracts import CLASSIFIED_PAIR_COLUMNS, GROUP_ASSIGNMENT_COLUMNS
from .validation import make_release_decision, validate_predictions, validate_split
from .verification import VERIFICATION_COLUMNS


# These files describe the current run and must not survive a failed rerun.
# Append-only history and reviewer input are deliberately excluded.
CURRENT_RUN_ARTIFACTS = (
    "suspect_pairs.csv",
    "split_manifest.csv",
    "pair_evidence.csv",
    "groups.csv",
    "automatic_pairs.csv",
    "review_queue.csv",
    "contact_sheet.html",
    "review_context.json",
    "diagnostics.json",
    "decision.json",
)


def clear_current_artifacts(output_dir: str | Path) -> None:
    """Remove only generated current-run T05 files from ``output_dir``.

    ``audit_log.jsonl`` and ``decision_log.csv`` are append-only histories, and
    ``review_queue_reviewed.csv`` is reviewer input, so none is removed. This is
    called before input processing to prevent a failed rerun from being mistaken
    for a successful run because an old ``suspect_pairs.csv`` still exists.

    Raises:
        ValueError: If a managed artifact path unexpectedly names a directory.
    """
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for name in CURRENT_RUN_ARTIFACTS:
        path = root / name
        if path.is_dir():
            raise ValueError(f"managed T05 artifact path is a directory: {path}")
        if path.exists():
            path.unlink()


def _validated_pair_evidence(
    pair_evidence: pd.DataFrame,
    image_ids: list[str],
) -> pd.DataFrame:
    if not isinstance(pair_evidence, pd.DataFrame):
        raise ValueError(
            f"pair_evidence must be a pandas DataFrame, got {type(pair_evidence).__name__}"
        )
    missing = [column for column in CLASSIFIED_PAIR_COLUMNS if column not in pair_evidence.columns]
    if missing:
        raise ValueError(
            f"pair_evidence is missing required columns {missing}; has {list(pair_evidence.columns)}"
        )
    extras = [column for column in (*VERIFICATION_COLUMNS, "coarse_duplicate_score") if column in pair_evidence]
    out = pair_evidence[list(CLASSIFIED_PAIR_COLUMNS) + extras].copy()
    for column in extras:
        values = pd.to_numeric(out[column], errors="raise").astype(float)
        if not values.between(0, 1).all():
            raise ValueError(f"pair_evidence.{column} must be finite in [0,1]")
    if out.pair_id.isna().any() or out.pair_id.astype(str).duplicated().any():
        raise ValueError("pair_evidence.pair_id values must be non-null and unique")
    # Reuse evaluator-facing pair hygiene for canonical/known IDs, explicit flags,
    # and finite [0,1] duplicate scores without conflating evidence with groups.
    validate_predictions(
        out[["img_a", "img_b", "pred_dup"]].assign(score=out.duplicate_score),
        image_ids,
    )
    return out.sort_values(["img_a", "img_b"], kind="mergesort", ignore_index=True)


def write_artifacts(
    output_dir: str | Path,
    *,
    image_ids: list[str],
    predictions: pd.DataFrame,
    assignments: pd.DataFrame,
    groups: pd.DataFrame,
    pair_evidence: pd.DataFrame,
    automatic_predictions: pd.DataFrame,
    diagnostics: Mapping[str, Any],
    decision: Decision,
) -> dict[str, Path]:
    """Validate, stage, and publish final T05 production artifacts.

    Pair artifacts require canonical known-image pairs and explicit binary flags.
    The split must cover every image exactly and preserve valid PoC group schema.
    Evidence requires the complete classified-pair columns. Empty pair tables are
    accepted when they retain their headers. Inputs are copied and never mutated.

    Files are first written under a temporary directory on the output volume.
    ``suspect_pairs.csv`` is replaced last, so the repository runner cannot see a
    successful prediction artifact until all supporting current-run files have
    been published. Existing decision history is copied and appended atomically;
    audit history is only touched into existence and is never rewritten.

    Returns:
        Mapping from artifact filename to final path.

    Raises:
        ValueError: For malformed artifacts, mismatched automatic/final pair
            pools, invalid diagnostics/decision objects, or unsafe output paths.
    """
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if not isinstance(diagnostics, Mapping):
        raise ValueError("diagnostics must be a mapping")
    if not isinstance(decision, Decision):
        raise ValueError(f"decision must be Decision, got {type(decision).__name__}")
    decision.canonical  # validates status aliases/canonical values
    # Reuse Task 11's complete diagnostics integrity checks. The permissive
    # temporary policy affects only the throwaway validation decision; the
    # caller-supplied signed decision is what gets written.
    make_release_decision(
        diagnostics,
        decision.owner,
        {
            "decision_max_unresolved": diagnostics.get("unresolved_count", 0),
            "max_drop_fraction": 1.0,
            "evidence": ["artifact-validation"],
        },
    )

    final_pairs = validate_predictions(predictions, image_ids)
    automatic_pairs = validate_predictions(automatic_predictions, image_ids)
    final_keys = set(final_pairs[["img_a", "img_b"]].itertuples(index=False, name=None))
    automatic_keys = set(
        automatic_pairs[["img_a", "img_b"]].itertuples(index=False, name=None)
    )
    if final_keys != automatic_keys:
        raise ValueError("automatic and final predictions must cover the same pair pool")
    split = validate_split(assignments, image_ids, groups, "poc")
    evidence = _validated_pair_evidence(pair_evidence, image_ids)
    normalized_groups = groups.copy()
    missing_group_columns = [
        column for column in GROUP_ASSIGNMENT_COLUMNS if column not in normalized_groups.columns
    ]
    if missing_group_columns:
        raise ValueError(f"groups is missing required columns {missing_group_columns}")
    normalized_groups = normalized_groups.sort_values(
        "image_id", kind="mergesort", ignore_index=True
    )

    outputs = {
        name: root / name
        for name in (
            "suspect_pairs.csv",
            "split_manifest.csv",
            "pair_evidence.csv",
            "groups.csv",
            "automatic_pairs.csv",
            "diagnostics.json",
            "decision.json",
            "decision_log.csv",
            "audit_log.jsonl",
        )
    }
    audit_path = root / "audit_log.jsonl"
    if audit_path.is_dir():
        raise ValueError(f"audit log path is a directory: {audit_path}")
    audit_path.touch(exist_ok=True)
    with TemporaryDirectory(prefix=".t05-stage-", dir=root) as temporary:
        stage = Path(temporary)
        final_pairs.to_csv(stage / "suspect_pairs.csv", index=False)
        split.to_csv(stage / "split_manifest.csv", index=False)
        evidence.to_csv(stage / "pair_evidence.csv", index=False, na_rep="")
        normalized_groups.to_csv(stage / "groups.csv", index=False)
        automatic_pairs.to_csv(stage / "automatic_pairs.csv", index=False)
        write_json(stage / "diagnostics.json", dict(diagnostics))

        existing_log = root / "decision_log.csv"
        staged_log = stage / "decision_log.csv"
        if existing_log.is_file():
            shutil.copyfile(existing_log, staged_log)
        elif existing_log.exists():
            raise ValueError(f"decision log path is not a file: {existing_log}")
        write_decision(decision, stage / "decision.json", staged_log)

        # Publish the required prediction last: main.py treats its presence as
        # success, so supporting files must already have been replaced.
        publish_order = (
            "split_manifest.csv",
            "pair_evidence.csv",
            "groups.csv",
            "automatic_pairs.csv",
            "diagnostics.json",
            "decision.json",
            "decision_log.csv",
            "suspect_pairs.csv",
        )
        for name in publish_order:
            os.replace(stage / name, root / name)

    return outputs
