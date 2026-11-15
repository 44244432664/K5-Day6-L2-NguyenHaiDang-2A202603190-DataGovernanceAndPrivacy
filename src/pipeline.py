"""Repository five-stage pipeline, with complete T05 production integration.

T05 runs entirely offline and never reads instructor ground truth. Other tracks
retain the starter behavior and fail clearly at their unimplemented process step.
"""
from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from .common.context import RunContext
from .common.decision import Decision, write_decision
from .common.io import sha256_file, sha256_text
from .common.manifest import verify_manifest
from .common.review import make_review_queue, write_review_queue
from .metrics.tracks import get_track
from .t05 import (
    PAIR_REVIEW_COLUMNS,
    build_pair_evidence,
    classify_pairs,
    clear_current_artifacts,
    extract_features,
    find_group_conflicts,
    load_inputs,
    make_release_decision,
    rank_review_pairs,
    rebuild_after_review,
    render_pair_review,
    score_duplicate_evidence,
    summarize_risks,
    validate_pair_reviews,
    validate_params,
    verify_visual_pairs,
    write_artifacts,
    write_review_context,
)


def _input_fingerprint(images: pd.DataFrame, input_dir: Path) -> str:
    """Hash validated public metadata plus actual public image bytes."""
    records = []
    for row in images.sort_values("image_id", kind="mergesort").itertuples(index=False):
        records.append(
            {
                "image_id": str(row.image_id),
                "path": str(row.path),
                "camera_id": str(row.camera_id),
                "captured_at": pd.Timestamp(row.captured_at).isoformat(),
                "sha256": sha256_file(input_dir / str(row.path)),
            }
        )
    metadata_hash = sha256_file(input_dir / "images.csv")
    return sha256_text(
        json.dumps(
            {"images_csv_sha256": metadata_hash, "images": records},
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


def _config_fingerprint(ctx: RunContext, params: Mapping[str, Any]) -> str:
    payload = {
        "track": ctx.track,
        "seed": int(ctx.seed),
        "strict_provenance": bool(ctx.config.get("strict_provenance", True)),
        "params": dict(params),
    }
    return sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _capture_prior_review_context(ctx: RunContext) -> dict[str, Any] | None:
    """Capture review identity before current-run files are cleared."""
    reviewed_path = ctx.output_dir / "review_queue_reviewed.csv"
    context_path = ctx.output_dir / "review_context.json"
    raw: str | None = None
    read_error: Exception | None = None
    if reviewed_path.exists():
        if not reviewed_path.is_file():
            read_error = ValueError(f"review export path is not a file: {reviewed_path}")
        elif context_path.is_file():
            try:
                raw = context_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                read_error = exc
        elif context_path.exists():
            read_error = ValueError(f"review context path is not a file: {context_path}")
    clear_current_artifacts(ctx.output_dir)
    if read_error is not None:
        raise ValueError(f"cannot preserve prior review context: {read_error}") from read_error
    if reviewed_path.exists() and raw is None:
        raise ValueError(
            "review_queue_reviewed.csv exists but its original review_context.json is missing"
        )
    if raw is None:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"prior review_context.json is invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("prior review_context.json must contain an object")
    # This context is review identity, not a reusable success artifact. Restore
    # it after clearing current-run files so an accidental failed rerun does not
    # orphan the preserved reviewer export; final predictions remain absent.
    context_path.write_text(raw, encoding="utf-8")
    return payload


def _check_prior_fingerprints(
    prior: Mapping[str, Any] | None,
    input_fingerprint: str,
    config_fingerprint: str,
) -> None:
    if prior is None:
        return
    if prior.get("input_fingerprint") != input_fingerprint:
        raise ValueError("stale reviewed export: input fingerprint changed")
    if prior.get("config_fingerprint") != config_fingerprint:
        raise ValueError("stale reviewed export: frozen configuration fingerprint changed")


# ---------------------------------------------------------------- 01 load
def load(ctx: RunContext) -> dict[str, Any]:
    """Load validated public inputs for the configured track.

    T05 clears only stale current-run artifacts, preserves review/audit history,
    validates public images and provenance, freezes parameters, and computes
    review-staleness fingerprints. Non-T05 tracks retain the scaffold manifest
    loader so their unimplemented status remains explicit.
    """
    if ctx.track == "t05":
        prior_context = _capture_prior_review_context(ctx)
        params = validate_params(ctx.config.get("params"))
        loaded = load_inputs(
            ctx.input_dir,
            strict_provenance=ctx.config.get("strict_provenance", True),
        )
        input_fingerprint = _input_fingerprint(loaded["images"], ctx.input_dir)
        config_fingerprint = _config_fingerprint(ctx, params)
        _check_prior_fingerprints(prior_context, input_fingerprint, config_fingerprint)
        loaded.update(
            {
                "params": params,
                "input_fingerprint": input_fingerprint,
                "config_fingerprint": config_fingerprint,
                "prior_review_context": prior_context,
            }
        )
        ctx.stats.update(
            {
                "n_items": len(loaded["images"]),
                "provenance_verified": bool(loaded["provenance"]["verified"]),
                "provenance_problem_count": len(loaded["provenance"]["problems"]),
            }
        )
        if loaded["provenance"]["problems"] and ctx.audit is not None:
            ctx.audit.log(
                actor="pipeline",
                action="provenance_problems",
                item_id=str(ctx.input_dir / "manifest.csv"),
                after=loaded["provenance"]["problems"][:20],
            )
        return loaded

    manifest = ctx.input_dir / "manifest.csv"
    problems = verify_manifest(manifest) if manifest.exists() else [{"problem": "no_manifest"}]
    if problems:
        if ctx.audit is not None:
            ctx.audit.log(
                actor="pipeline",
                action="provenance_problems",
                item_id=str(manifest),
                after=problems[:20],
            )
        if ctx.config.get("strict_provenance", True):
            raise ValueError(
                f"provenance check failed ({len(problems)} problems), first: {problems[:3]}"
            )
    return {"manifest": pd.read_csv(manifest) if manifest.exists() else None}


# ---------------------------------------------------------------- 02 process
def process(ctx: RunContext, data: dict[str, Any]) -> dict[str, Any] | pd.DataFrame:
    """Extract T05 evidence and build automatic duplicate/split results."""
    if ctx.track != "t05":
        spec = get_track(ctx.track)
        raise NotImplementedError(
            f"Implement process() for {spec.track} ({spec.title}). It must produce a DataFrame "
            f"with at least {spec.pred_columns} (saved as out/{spec.pred_file}). See "
            f"docs/tracks/{spec.track.upper()}.md"
        )

    images = data["images"]
    params = data["params"]
    features = extract_features(images, ctx.input_dir, params)
    raw_evidence = build_pair_evidence(images, features)
    raw_evidence = verify_visual_pairs(images, ctx.input_dir, raw_evidence, params)
    classified = classify_pairs(score_duplicate_evidence(raw_evidence, params), params)
    empty_reviews = pd.DataFrame(columns=PAIR_REVIEW_COLUMNS)
    automatic = rebuild_after_review(
        images,
        classified,
        empty_reviews,
        params,
        copy.deepcopy(ctx.rng),
    )
    result = {
        **data,
        **automatic,
        "features": features,
        "pair_evidence": classified,
    }
    ctx.stats.update(
        {
            "n_candidate_pairs": len(classified),
            "n_automatic_positive_pairs": int(automatic["automatic_predictions"].pred_dup.sum()),
            "n_automatic_duplicate_groups": int(
                automatic["duplicate_groups"].duplicate_group_id.nunique()
            ),
            "n_automatic_split_groups": int(
                automatic["split_groups"].split_group_id.nunique()
            ),
        }
    )
    return result


# ---------------------------------------------------------------- 03 rank
def rank(ctx: RunContext, scored: dict[str, Any] | pd.DataFrame) -> pd.DataFrame:
    """Rank T05 pair reviews and write queue, HTML, and fingerprint context."""
    if ctx.track != "t05":
        if not isinstance(scored, pd.DataFrame):
            raise ValueError("rank(): non-T05 process output must be a DataFrame")
        budget = int(ctx.param("budget", 20))
        if "score" not in scored.columns:
            raise ValueError(
                "rank(): process() output needs a numeric `score` column (higher = review first), "
                "or adapt rank() to your track"
            )
        queue = make_review_queue(
            scored,
            id_col=scored.columns[0],
            score_col="score",
            budget=budget,
            reason_col="reason" if "reason" in scored.columns else None,
        )
        write_review_queue(queue, ctx.out("review_queue.csv"))
        return queue

    if not isinstance(scored, dict):
        raise ValueError("T05 rank() requires the structured process result")
    queue = rank_review_pairs(
        scored["automatic_classified_pairs"],
        scored["split_groups"],
        scored["split_manifest"],
        int(scored["params"]["budget"]),
    )
    write_review_queue(queue, ctx.out("review_queue.csv"))
    render_pair_review(queue, scored["images"], ctx.out("contact_sheet.html"))
    write_review_context(
        queue,
        scored["input_fingerprint"],
        scored["config_fingerprint"],
        ctx.out("review_context.json"),
    )
    ctx.stats["review_queue_size"] = len(queue)
    return queue


# ---------------------------------------------------------------- 04 review
def review(
    ctx: RunContext,
    scored: dict[str, Any] | pd.DataFrame,
    queue: pd.DataFrame,
) -> dict[str, Any] | pd.DataFrame:
    """Validate optional T05 reviewed export and rebuild every dependent result."""
    if ctx.track != "t05":
        return scored
    if not isinstance(scored, dict):
        raise ValueError("T05 review() requires the structured process result")

    reviewed_path = ctx.output_dir / "review_queue_reviewed.csv"
    if reviewed_path.exists():
        try:
            reviewed_export = pd.read_csv(reviewed_path, dtype=str, keep_default_na=False)
        except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            raise ValueError(f"cannot read reviewed queue at {reviewed_path}: {exc}") from exc
        prior_context = scored.get("prior_review_context")
        if prior_context is None:
            raise ValueError("reviewed export has no preserved original review context")
        reviewed = validate_pair_reviews(reviewed_export, queue, prior_context)
    else:
        reviewed = pd.DataFrame(columns=PAIR_REVIEW_COLUMNS)

    rebuilt = rebuild_after_review(
        scored["images"],
        scored["automatic_classified_pairs"],
        reviewed,
        scored["params"],
        copy.deepcopy(ctx.rng),
        ctx.audit,
    )
    final = {**scored, **rebuilt}
    ctx.stats.update(
        {
            "n_reviewed": int(rebuilt["review_summary"]["n_reviewed"]),
            "n_review_changes": int(rebuilt["review_summary"]["changed_pair_count"]),
            "n_final_prediction_changes": int(rebuilt["review_summary"]["final_prediction_changed_count"]),
            "n_review_constraints": int(rebuilt["review_summary"]["constraint_count"]),
            "n_unresolved_reviews": int(rebuilt["review_summary"]["unresolved_count"]),
            "n_final_positive_pairs": int(rebuilt["predictions"].pred_dup.sum()),
        }
    )
    return final


def _add_provenance_risk(
    diagnostics: dict[str, Any], provenance: Mapping[str, Any]
) -> None:
    problems = list(provenance.get("problems", []))
    if not problems:
        diagnostics["provenance_verified"] = True
        diagnostics["provenance_problem_count"] = 0
        return
    diagnostics["provenance_verified"] = False
    diagnostics["provenance_problem_count"] = len(problems)
    diagnostics["provenance_problems"] = problems[:20]
    diagnostics["risks"].append(
        {
            "code": "PROVENANCE_UNVERIFIED",
            "status": "HOLD",
            "blocking": True,
            "count": len(problems),
            "detail": "public input provenance has unresolved validation problems",
        }
    )
    diagnostics["blocking_risk_count"] += 1
    diagnostics["recommended_status"] = "HOLD"


# ---------------------------------------------------------------- 05 decide
def decide(ctx: RunContext, final: dict[str, Any] | pd.DataFrame) -> Decision:
    """Validate final T05 structure, summarize risks, and create release decision."""
    if ctx.track != "t05":
        return Decision(
            status="HOLD",
            reason_code="POLICY_NOT_IMPLEMENTED",
            reason="Decision policy not implemented yet",
            owner=ctx.config.get("owner", "TODO-owner"),
            scope=ctx.track,
        )
    if not isinstance(final, dict):
        raise ValueError("T05 decide() requires the structured reviewed result")

    conflicts = find_group_conflicts(final["duplicate_groups"], final["constraints"])
    diagnostics = summarize_risks(
        final["predictions"],
        final["split_manifest"],
        final["split_groups"],
        final["images"].image_id,
        review_summary=final["review_summary"],
        conflicts=conflicts,
        split_summary=final["split_summary"],
        mode="poc",
    )
    _add_provenance_risk(diagnostics, final["provenance"])
    configured_policy = ctx.config.get("release_policy") or {}
    if not isinstance(configured_policy, Mapping):
        raise ValueError("release_policy must be a mapping")
    policy = {
        "decision_max_unresolved": final["params"]["decision_max_unresolved"],
        "max_drop_fraction": 0.0,
        **dict(configured_policy),
        "scope": f"{ctx.track}:{ctx.run_name}",
        "evidence": [
            str(ctx.out("diagnostics.json")),
            str(ctx.out("suspect_pairs.csv")),
            str(ctx.out("split_manifest.csv")),
            str(ctx.out("groups.csv")),
        ],
    }
    decision = make_release_decision(
        diagnostics,
        str(ctx.config.get("owner") or ""),
        policy,
    )
    final["conflicts"] = conflicts
    final["diagnostics"] = diagnostics
    final["decision"] = decision
    ctx.stats.update(
        {
            "decision_status": decision.canonical,
            "decision_reason_code": decision.reason_code,
            "n_train": diagnostics["n_train"],
            "n_test": diagnostics["n_test"],
            "n_drop": diagnostics["n_drop"],
            "retention_fraction": diagnostics["retention_fraction"],
            "actual_test_fraction": diagnostics["actual_test_fraction"],
            "blocking_risk_count": diagnostics["blocking_risk_count"],
        }
    )
    return decision


def run(ctx: RunContext) -> None:
    """Run load → process → rank → review → decide and publish artifacts."""
    spec = get_track(ctx.track)
    data = load(ctx)
    scored = process(ctx, data)
    queue = rank(ctx, scored)
    final = review(ctx, scored, queue)
    decision = decide(ctx, final)

    if ctx.track == "t05":
        if not isinstance(final, dict):
            raise AssertionError("T05 final result unexpectedly lost structured artifacts")
        write_artifacts(
            ctx.output_dir,
            image_ids=final["images"].image_id.astype(str).tolist(),
            predictions=final["predictions"],
            assignments=final["split_manifest"],
            groups=final["split_groups"],
            pair_evidence=final["pair_evidence"],
            automatic_predictions=final["automatic_predictions"],
            diagnostics=final["diagnostics"],
            decision=decision,
        )
        return

    if not isinstance(final, pd.DataFrame):
        raise ValueError("non-T05 pipeline output must be a DataFrame")
    ordered_columns = spec.pred_columns + [
        column for column in final.columns if column not in spec.pred_columns
    ]
    final[ordered_columns].to_csv(ctx.out(spec.pred_file), index=False)
    write_decision(decision, ctx.out("decision.json"), ctx.out("decision_log.csv"))
