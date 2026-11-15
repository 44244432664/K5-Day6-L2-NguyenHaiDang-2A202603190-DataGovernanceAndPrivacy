"""Task 9 tests for deterministic pair-review ranking and offline artifacts."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.common.io import load_config
from src.common.review import REVIEW_COLUMNS, AuditLog
from src.t05 import (
    PAIR_REVIEW_COLUMNS,
    apply_pair_reviews,
    assign_group_splits,
    build_duplicate_groups,
    build_pair_evidence,
    build_split_groups,
    classify_pairs,
    extract_features,
    load_inputs,
    make_pair_id,
    rank_review_pairs,
    rebuild_after_review,
    render_pair_review,
    score_duplicate_evidence,
    validate_pair_reviews,
    write_review_context,
)
from src.t05.contracts import CLASSIFIED_PAIR_COLUMNS


def _pair(
    img_a,
    img_b,
    classification,
    *,
    duplicate_score=0.75,
    review_priority=1.0,
    exact=False,
    reason_codes="phash_within_limit;dhash_within_limit;visual_signals:2;needs_review",
):
    positive = classification == "positive"
    return {
        "pair_id": make_pair_id(img_a, img_b),
        "img_a": min(img_a, img_b),
        "img_b": max(img_a, img_b),
        "exact_match": exact,
        "phash_distance": 4.0,
        "dhash_distance": 4.0,
        "histogram_distance": 0.04,
        "thumbnail_distance": 0.05,
        "same_camera": True,
        "time_delta_seconds": 10.0,
        "duplicate_score": duplicate_score,
        "review_priority": review_priority,
        "reason_codes": reason_codes,
        "classification": classification,
        "pred_dup": int(positive),
    }


def _pairs(*rows):
    return pd.DataFrame(rows, columns=CLASSIFIED_PAIR_COLUMNS)


def _groups():
    return pd.DataFrame(
        [
            ("a", "dup:ab", "scene:ab", 2, False),
            ("b", "dup:ab", "scene:ab", 2, False),
            ("c", "dup:cd", "scene:cd", 2, False),
            ("d", "dup:cd", "scene:cd", 2, False),
            ("e", "dup:e", "scene:e", 1, False),
        ],
        columns=(
            "image_id",
            "duplicate_group_id",
            "split_group_id",
            "split_group_size",
            "exceeds_max_component_size",
        ),
    )


def _split():
    return pd.DataFrame(
        [("a", "train"), ("b", "train"), ("c", "test"), ("d", "test"), ("e", "test")],
        columns=("image_id", "split"),
    )


def test_review_ranking_is_capped_stable_and_uses_shared_schema():
    pairs = _pairs(
        _pair("a", "c", "ambiguous", review_priority=1.0),
        _pair("d", "e", "ambiguous", review_priority=0.8),
        _pair("b", "c", "positive", duplicate_score=0.95, review_priority=0.0),
        _pair("a", "b", "positive", duplicate_score=1.0, review_priority=0.0, exact=True),
    )
    first = rank_review_pairs(pairs, _groups(), _split(), budget=2)
    second = rank_review_pairs(
        pairs.sample(frac=1, random_state=42),
        _groups().sample(frac=1, random_state=7),
        _split().sample(frac=1, random_state=9),
        budget=2,
    )

    pd.testing.assert_frame_equal(first, second)
    assert tuple(first.columns) == PAIR_REVIEW_COLUMNS == tuple(REVIEW_COLUMNS)
    assert len(first) == 2
    assert first.item_id.iloc[0] == make_pair_id("a", "c")
    assert make_pair_id("a", "b") not in set(first.item_id)  # exact byte match excluded
    assert first["rank"].tolist() == [1, 2]
    assert first.score.is_monotonic_decreasing


def test_color_similarity_trap_is_reviewable_even_with_zero_uncertainty():
    color_trap = _pair(
        "a",
        "e",
        "negative",
        duplicate_score=0.2,
        review_priority=0.0,
        reason_codes="histogram_within_limit;thumbnail_within_limit;visual_signals:1;auto_negative",
    )
    queue = rank_review_pairs(_pairs(color_trap), _groups(), _split(), budget=20)
    assert len(queue) == 1
    assert "color_similarity_trap" in queue.iloc[0]["slice"]
    assert "reject or edit" in queue.iloc[0].suggested_action


def test_empty_candidate_queue_preserves_shared_headers():
    exact = _pair(
        "a", "b", "positive", duplicate_score=1.0, review_priority=0.0, exact=True
    )
    queue = rank_review_pairs(_pairs(exact), _groups(), _split(), budget=20)
    assert queue.empty
    assert tuple(queue.columns) == PAIR_REVIEW_COLUMNS


def test_pair_review_html_embeds_both_images_and_shared_export_schema(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (20, 12), "red").save(image_dir / "a.png")
    Image.new("RGB", (20, 12), "blue").save(image_dir / "c.png")
    images = pd.DataFrame(
        [("a", "images/a.png"), ("c", "images/c.png")], columns=("image_id", "path")
    )
    images.attrs["input_dir"] = str(tmp_path)
    queue = rank_review_pairs(
        _pairs(_pair("a", "c", "ambiguous")),
        _groups()[_groups().image_id.isin(["a", "c"])],
        _split()[_split().image_id.isin(["a", "c"])],
        budget=1,
    )
    output = render_pair_review(queue, images, tmp_path / "contact_sheet.html")
    html = output.read_text(encoding="utf-8")

    assert html.count("data:image/jpeg;base64,") == 2
    assert make_pair_id("a", "c") in html
    assert 'const COLS = ["rank","item_id","score","reason","suggested_action","slice","evidence_path"' in html
    assert "http://" not in html and "https://" not in html


def test_review_context_detects_queue_input_and_config_identity_changes(tmp_path):
    queue = rank_review_pairs(
        _pairs(_pair("a", "c", "ambiguous")), _groups(), _split(), budget=1
    )
    first_path = write_review_context(
        queue, "input-v1", "config-v1", tmp_path / "context-1.json"
    )
    second_path = write_review_context(
        queue, "input-v2", "config-v1", tmp_path / "context-2.json"
    )
    third_path = write_review_context(
        queue, "input-v1", "config-v2", tmp_path / "context-3.json"
    )
    first = json.loads(first_path.read_text(encoding="utf-8"))
    second = json.loads(second_path.read_text(encoding="utf-8"))
    third = json.loads(third_path.read_text(encoding="utf-8"))

    assert first["queue_fingerprint"] == second["queue_fingerprint"] == third["queue_fingerprint"]
    assert len(
        {
            first["context_fingerprint"],
            second["context_fingerprint"],
            third["context_fingerprint"],
        }
    ) == 3
    assert first["item_ids"] == queue.item_id.tolist()
    assert tuple(first["review_columns"]) == PAIR_REVIEW_COLUMNS


def test_repository_fixture_review_queue_is_capped_and_excludes_exact_pairs():
    loaded = load_inputs("fixtures/t05_near_duplicate", strict_provenance=True)
    params = load_config("configs/lab.yaml")["params"]
    features = extract_features(
        loaded["images"], "fixtures/t05_near_duplicate", params
    )
    evidence = build_pair_evidence(loaded["images"], features)
    classified = classify_pairs(score_duplicate_evidence(evidence, params), params)
    duplicate_groups = build_duplicate_groups(loaded["images"], classified)
    split_groups = build_split_groups(
        loaded["images"], duplicate_groups, classified, params=params
    )
    assignments = assign_group_splits(
        split_groups, params["test_fraction"], np.random.default_rng(42)
    )
    queue = rank_review_pairs(classified, split_groups, assignments, params["budget"])
    exact_ids = set(classified.loc[classified.exact_match, "pair_id"])

    assert len(queue) == 20
    assert tuple(queue.columns) == PAIR_REVIEW_COLUMNS
    assert queue.item_id.is_unique
    assert set(queue.item_id).isdisjoint(exact_ids)
    assert queue.score.is_monotonic_decreasing


def _reviewed_queue(queue, decisions):
    reviewed = queue.copy()
    for item_id, decision, override, reviewer, note in decisions:
        mask = reviewed.item_id == item_id
        reviewed.loc[mask, "reviewer_decision"] = decision
        reviewed.loc[mask, "override_value"] = override
        reviewed.loc[mask, "reviewer"] = reviewer
        reviewed.loc[mask, "review_note"] = note
        reviewed.loc[mask, "reviewed_at"] = "2026-11-15T10:00:00Z"
    return reviewed


def test_validate_pair_reviews_normalizes_all_actions_and_relations(tmp_path):
    pairs = _pairs(
        _pair("a", "c", "ambiguous"),
        _pair("a", "d", "ambiguous", review_priority=0.9),
        _pair("b", "c", "positive", review_priority=0.8),
        _pair("b", "e", "ambiguous", review_priority=0.7),
        _pair("d", "e", "ambiguous", review_priority=0.6),
    )
    queue = rank_review_pairs(pairs, _groups(), _split(), budget=5)
    item_ids = queue.item_id.tolist()
    reviewed = _reviewed_queue(
        queue,
        [
            (item_ids[0], "approve", "", "alice", "confirmed duplicate"),
            (item_ids[1], "reject", "same_scene_only", "bob", "same scene, different view"),
            (item_ids[2], "edit", "different_scene", "cara", "visually distinct scene"),
            (item_ids[3], "escalate", "", "dan", "requires domain owner"),
            (item_ids[4], "skip", "", "erin", ""),
        ],
    )
    context_path = write_review_context(
        queue, "input-v1", "config-v1", tmp_path / "review_context.json"
    )
    validated = validate_pair_reviews(reviewed, queue, context_path)
    relation_of = dict(zip(validated.item_id, validated.review_relation))

    assert len(validated) == 5
    assert relation_of[item_ids[0]] == "duplicate"
    assert relation_of[item_ids[1]] == "same_scene_only"
    assert relation_of[item_ids[2]] == "different_scene"
    assert relation_of[item_ids[3]] == ""
    assert relation_of[item_ids[4]] == ""


@pytest.mark.parametrize(
    ("decision", "override", "reviewer", "note", "message"),
    [
        ("approve", "", "", "confirmed", "require a reviewer"),
        ("reject", "", "alice", "not duplicate", "requires explicit"),
        ("edit", "unknown", "alice", "change relation", "requires one of"),
        ("escalate", "duplicate", "alice", "needs owner", "must not set"),
        ("approve", "", "alice", "", "require a review_note"),
    ],
)
def test_validate_pair_reviews_rejects_invalid_action_details(
    tmp_path, decision, override, reviewer, note, message
):
    queue = rank_review_pairs(
        _pairs(_pair("a", "c", "ambiguous")), _groups(), _split(), budget=1
    )
    reviewed = _reviewed_queue(
        queue, [(queue.item_id.iloc[0], decision, override, reviewer, note)]
    )
    context = write_review_context(queue, "input", "config", tmp_path / "context.json")
    with pytest.raises(ValueError, match=message):
        validate_pair_reviews(reviewed, queue, context)


def test_validate_pair_reviews_rejects_stale_or_unknown_rows(tmp_path):
    queue = rank_review_pairs(
        _pairs(_pair("a", "c", "ambiguous")), _groups(), _split(), budget=1
    )
    context = write_review_context(queue, "input", "config", tmp_path / "context.json")
    reviewed = _reviewed_queue(
        queue, [(queue.item_id.iloc[0], "approve", "", "alice", "confirmed")]
    )
    stale = reviewed.copy()
    stale.loc[0, "reason"] = "changed evidence"
    with pytest.raises(ValueError, match="immutable field"):
        validate_pair_reviews(stale, queue, context)

    unknown = reviewed.copy()
    unknown.loc[0, "item_id"] = make_pair_id("x", "y")
    with pytest.raises(ValueError, match="unknown queue item"):
        validate_pair_reviews(unknown, queue, context)

    altered_context = json.loads(context.read_text(encoding="utf-8"))
    altered_context["input_fingerprint"] = "other-input"
    with pytest.raises(ValueError, match="fingerprint is invalid"):
        validate_pair_reviews(reviewed, queue, altered_context)


def test_apply_pair_reviews_preserves_automatic_input_and_appends_audit(tmp_path):
    pairs = _pairs(
        _pair("a", "c", "ambiguous"),
        _pair("b", "c", "positive", review_priority=0.8),
        _pair("d", "e", "ambiguous", review_priority=0.7),
    )
    original = pairs.copy(deep=True)
    queue = rank_review_pairs(pairs, _groups(), _split(), budget=3)
    decisions = []
    for item_id in queue.item_id:
        if item_id == make_pair_id("a", "c"):
            decisions.append((item_id, "approve", "", "alice", "confirmed duplicate"))
        elif item_id == make_pair_id("b", "c"):
            decisions.append((item_id, "reject", "same_scene_only", "bob", "different view"))
        else:
            decisions.append((item_id, "escalate", "", "cara", "needs owner"))
    reviewed = _reviewed_queue(queue, decisions)
    audit = AuditLog(tmp_path / "audit.jsonl")
    audit.log("system", "existing_event", "seed")
    applied = apply_pair_reviews(pairs, reviewed, audit)
    by_id = applied.set_index("pair_id")

    pd.testing.assert_frame_equal(pairs, original)
    assert by_id.at[make_pair_id("a", "c"), "classification"] == "positive"
    assert by_id.at[make_pair_id("a", "c"), "pred_dup"] == 1
    assert by_id.at[make_pair_id("b", "c"), "classification"] == "negative"
    assert by_id.at[make_pair_id("b", "c"), "review_relation"] == "same_scene_only"
    assert bool(by_id.at[make_pair_id("d", "e"), "review_unresolved"])
    events = audit.read()
    assert events[0]["action"] == "existing_event"
    assert len(events) == 4
    assert events[1]["before"] and events[1]["after"]


def _rebuild_images(ids):
    return pd.DataFrame(
        {
            "image_id": ids,
            "camera_id": [f"cam-{index}" for index in range(len(ids))],
            "captured_at": pd.to_datetime(
                [f"2026-01-01T00:{index:02d}:00Z" for index in range(len(ids))], utc=True
            ),
        }
    )


def _complete_pairs(images, states):
    rows = []
    metadata = images.set_index("image_id")
    for (img_a, img_b), state in states.items():
        row = _pair(
            img_a,
            img_b,
            state,
            duplicate_score=0.9 if state == "positive" else 0.7,
            review_priority=0.8,
        )
        row["same_camera"] = metadata.at[img_a, "camera_id"] == metadata.at[img_b, "camera_id"]
        row["time_delta_seconds"] = abs(
            (metadata.at[img_a, "captured_at"] - metadata.at[img_b, "captured_at"]).total_seconds()
        )
        rows.append(row)
    return _pairs(*rows)


def test_rebuild_rejection_changes_duplicate_output_but_keeps_scene_together(tmp_path):
    images = _rebuild_images(["a", "b", "c"])
    pairs = _complete_pairs(
        images,
        {
            ("a", "b"): "positive",
            ("a", "c"): "negative",
            ("b", "c"): "negative",
        },
    )
    reviewed = pd.DataFrame(columns=PAIR_REVIEW_COLUMNS)
    queue_row = rank_review_pairs(
        pairs,
        pd.DataFrame(
            [("a", "dup:ab", "scene:ab"), ("b", "dup:ab", "scene:ab"), ("c", "dup:c", "scene:c")],
            columns=("image_id", "duplicate_group_id", "split_group_id"),
        ),
        pd.DataFrame([("a", "train"), ("b", "train"), ("c", "test")], columns=("image_id", "split")),
        budget=3,
    )
    reviewed = _reviewed_queue(
        queue_row,
        [(make_pair_id("a", "b"), "reject", "same_scene_only", "alice", "same scene only")],
    )
    audit = AuditLog(tmp_path / "audit.jsonl")
    result = rebuild_after_review(
        images, pairs, reviewed, {}, np.random.default_rng(42), audit
    )
    automatic = result["automatic_predictions"].set_index(["img_a", "img_b"])
    final = result["predictions"].set_index(["img_a", "img_b"])
    groups = result["split_groups"].set_index("image_id")
    duplicate_groups = result["duplicate_groups"].set_index("image_id")

    assert automatic.at[("a", "b"), "pred_dup"] == 1
    assert final.at[("a", "b"), "pred_dup"] == 0
    assert duplicate_groups.at["a", "duplicate_group_id"] != duplicate_groups.at["b", "duplicate_group_id"]
    assert groups.at["a", "split_group_id"] == groups.at["b", "split_group_id"]
    assert result["constraints"].iloc[0].relation == "same_scene_only"
    assert len(audit.read()) == 1


def test_rebuild_rejection_cannot_return_through_positive_graph_path():
    images = _rebuild_images(["a", "b", "c"])
    pairs = _complete_pairs(
        images,
        {
            ("a", "b"): "positive",
            ("a", "c"): "positive",
            ("b", "c"): "positive",
        },
    )
    reviewed = pd.DataFrame(
        [
            {
                **{column: "" for column in PAIR_REVIEW_COLUMNS},
                "rank": 1,
                "item_id": make_pair_id("a", "c"),
                "score": 1.0,
                "reviewer": "alice",
                "reviewer_decision": "reject",
                "override_value": "same_scene_only",
                "review_note": "not duplicates",
                "reviewed_at": "2026-11-15T10:00:00Z",
            }
        ],
        columns=PAIR_REVIEW_COLUMNS,
    )
    result = rebuild_after_review(
        images, pairs, reviewed, {}, np.random.default_rng(42)
    )
    predicted = result["predictions"].set_index(["img_a", "img_b"]).pred_dup
    assert predicted.at[("a", "c")] == 0
    assert predicted.at[("b", "c")] == 0
    assert result["duplicate_groups"].duplicate_group_id.nunique() == 2


def test_rebuild_escalation_is_unresolved_and_blocks_provisional_go():
    images = _rebuild_images(["a", "b", "c"])
    pairs = _complete_pairs(
        images,
        {
            ("a", "b"): "positive",
            ("a", "c"): "negative",
            ("b", "c"): "negative",
        },
    )
    reviewed = pd.DataFrame(
        [
            {
                **{column: "" for column in PAIR_REVIEW_COLUMNS},
                "rank": 1,
                "item_id": make_pair_id("a", "b"),
                "score": 1.0,
                "reviewer": "owner",
                "reviewer_decision": "escalate",
                "review_note": "material uncertainty",
                "reviewed_at": "2026-11-15T10:00:00Z",
            }
        ],
        columns=PAIR_REVIEW_COLUMNS,
    )
    result = rebuild_after_review(
        images, pairs, reviewed, {}, np.random.default_rng(42)
    )
    assert result["review_summary"]["escalation_count"] == 1
    assert result["review_summary"]["unresolved_count"] == 1
    assert result["review_summary"]["recommended_status"] == "HOLD"
    assert result["split_summary"]["recommended_status"] == "HOLD"
