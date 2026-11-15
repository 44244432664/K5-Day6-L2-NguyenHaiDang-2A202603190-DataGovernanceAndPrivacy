"""T05 Leakage & Near-Duplicate Guardian.

The package currently exposes contracts/configuration, validated public input,
deterministic features, raw pair evidence, near-duplicate classification, and a
runnable exact-hash/random-split baseline. It also provides conservative
duplicate grouping, scene-aware splitting, and an offline pair-review queue;
validated reviews can rebuild groups and splits with append-only audit evidence.
Artifact validators and observable-risk policy produce shared GO/HOLD/REWORK
decisions without consulting hidden ground truth. Later tasks orchestrate these
modules in the five-stage pipeline.
"""

from .contracts import (
    CLASSIFIED_PAIR_COLUMNS,
    DEFAULT_PARAMS,
    DIFFERENT_SCENE_RELATION,
    DUPLICATE_GROUP_COLUMNS,
    DUPLICATE_RELATION,
    GROUP_ASSIGNMENT_COLUMNS,
    GROUP_CONFLICT_COLUMNS,
    GROUP_PAIR_COLUMNS,
    IMAGE_FEATURE_COLUMNS,
    PAIR_EVIDENCE_COLUMNS,
    PAIR_CLASSIFICATIONS,
    PAIR_CONSTRAINT_COLUMNS,
    PAIR_REVIEW_COLUMNS,
    PREDICTION_COLUMNS,
    RELATION_OVERRIDE_VALUES,
    REVIEWED_PAIR_COLUMNS,
    SCENE_ONLY_RELATION,
    SPLIT_GROUP_COLUMNS,
    SPLIT_MANIFEST_COLUMNS,
    canonical_pair,
    make_pair_id,
    validate_params,
)
from .data import IMAGE_TABLE_COLUMNS, load_inputs, resolve_image_path, validate_image_table
from .features import build_pair_evidence, compare_pair, extract_features, extract_image_features
from .matching import (
    build_duplicate_groups,
    classify_pairs,
    expand_group_pairs,
    find_group_conflicts,
    score_duplicate_evidence,
)
from .baseline import assign_random_splits, predict_exact_pairs
from .splitting import assign_group_splits, build_split_groups, summarize_split
from .review import (
    apply_pair_reviews,
    rank_review_pairs,
    rebuild_after_review,
    render_pair_review,
    validate_pair_reviews,
    write_review_context,
)
from .validation import (
    make_release_decision,
    summarize_risks,
    validate_predictions,
    validate_split,
)
from .artifacts import clear_current_artifacts, write_artifacts
from .verification import verify_visual_pairs

__all__ = [
    "DEFAULT_PARAMS",
    "CLASSIFIED_PAIR_COLUMNS",
    "DIFFERENT_SCENE_RELATION",
    "DUPLICATE_GROUP_COLUMNS",
    "DUPLICATE_RELATION",
    "GROUP_ASSIGNMENT_COLUMNS",
    "GROUP_CONFLICT_COLUMNS",
    "GROUP_PAIR_COLUMNS",
    "IMAGE_TABLE_COLUMNS",
    "IMAGE_FEATURE_COLUMNS",
    "PAIR_EVIDENCE_COLUMNS",
    "PAIR_CLASSIFICATIONS",
    "PAIR_CONSTRAINT_COLUMNS",
    "PAIR_REVIEW_COLUMNS",
    "PREDICTION_COLUMNS",
    "RELATION_OVERRIDE_VALUES",
    "REVIEWED_PAIR_COLUMNS",
    "SCENE_ONLY_RELATION",
    "SPLIT_GROUP_COLUMNS",
    "SPLIT_MANIFEST_COLUMNS",
    "assign_group_splits",
    "apply_pair_reviews",
    "assign_random_splits",
    "canonical_pair",
    "build_pair_evidence",
    "build_duplicate_groups",
    "build_split_groups",
    "clear_current_artifacts",
    "compare_pair",
    "classify_pairs",
    "extract_features",
    "extract_image_features",
    "expand_group_pairs",
    "find_group_conflicts",
    "load_inputs",
    "make_pair_id",
    "make_release_decision",
    "predict_exact_pairs",
    "rank_review_pairs",
    "rebuild_after_review",
    "render_pair_review",
    "resolve_image_path",
    "score_duplicate_evidence",
    "summarize_split",
    "summarize_risks",
    "validate_image_table",
    "validate_pair_reviews",
    "validate_predictions",
    "validate_split",
    "validate_params",
    "write_review_context",
    "write_artifacts",
    "verify_visual_pairs",
]
