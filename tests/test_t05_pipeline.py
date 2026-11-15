"""Task 12 integration tests for the standard T05 five-stage CLI pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src import main as runner
from src.common.manifest import build_manifest, write_manifest
from src.t05 import PAIR_REVIEW_COLUMNS, PREDICTION_COLUMNS


REQUIRED_OUTPUTS = {
    "suspect_pairs.csv",
    "split_manifest.csv",
    "pair_evidence.csv",
    "groups.csv",
    "automatic_pairs.csv",
    "review_queue.csv",
    "contact_sheet.html",
    "review_context.json",
    "audit_log.jsonl",
    "diagnostics.json",
    "decision.json",
    "decision_log.csv",
    "run_info.json",
}


def _run(input_dir: Path | str, output: Path, *, seed=42) -> int:
    return runner.main(
        [
            "--track",
            "t05",
            "--input",
            str(input_dir),
            "--output",
            str(output),
            "--offline",
            "--seed",
            str(seed),
            "--run-name",
            "poc",
        ]
    )


def _write_fixture(root: Path, colors: list[tuple[int, int, int]]) -> Path:
    image_dir = root / "images"
    image_dir.mkdir(parents=True)
    rows = []
    files = ["images.csv"]
    for index, color in enumerate(colors):
        image_id = f"im_{index:03d}"
        relative = f"images/{image_id}.png"
        array = np.zeros((24, 32, 3), dtype=np.uint8)
        array[:, :] = color
        # Add deterministic structure so perceptual hashes are not all flat.
        array[index % 24, :, :] = tuple(255 - channel for channel in color)
        Image.fromarray(array).save(root / relative)
        rows.append(
            {
                "image_id": image_id,
                "path": relative,
                "camera_id": f"cam-{index % 2}",
                "captured_at": f"2026-01-01T00:{index:02d}:00Z",
            }
        )
        files.append(relative)
    pd.DataFrame(rows).to_csv(root / "images.csv", index=False)
    manifest = build_manifest(root, files, source="generated-test", license="CC0")
    write_manifest(manifest, root / "manifest.csv")
    return root


def test_generated_clean_runs_are_reproducible_and_complete(tmp_path):
    fixture = _write_fixture(
        tmp_path / "fixture",
        [(20, 30, 40), (20, 30, 40), (180, 20, 40), (10, 190, 80)],
    )
    first = tmp_path / "first"
    second = tmp_path / "second"

    assert _run(fixture, first) == 0
    assert _run(fixture, second) == 0
    assert REQUIRED_OUTPUTS <= {path.name for path in first.iterdir()}

    for name in (
        "suspect_pairs.csv",
        "automatic_pairs.csv",
        "split_manifest.csv",
        "pair_evidence.csv",
        "groups.csv",
        "review_queue.csv",
        "review_context.json",
    ):
        assert (first / name).read_bytes() == (second / name).read_bytes(), name

    split = pd.read_csv(first / "split_manifest.csv")
    diagnostics = json.loads((first / "diagnostics.json").read_text(encoding="utf-8"))
    run_info = json.loads((first / "run_info.json").read_text(encoding="utf-8"))
    assert len(split) == 4 and split.image_id.is_unique
    assert set(split.split) == {"train", "test"}
    assert diagnostics["n_images"] == 4
    assert diagnostics["n_drop"] == 0
    assert run_info["status"] == 0
    assert run_info["stats"]["n_candidate_pairs"] == 6


def test_single_image_run_keeps_valid_empty_prediction_headers(tmp_path):
    fixture = _write_fixture(tmp_path / "single", [(30, 60, 90)])
    output = tmp_path / "output"

    assert _run(fixture, output) == 0
    predictions = pd.read_csv(output / "suspect_pairs.csv")
    automatic = pd.read_csv(output / "automatic_pairs.csv")
    decision = json.loads((output / "decision.json").read_text(encoding="utf-8"))

    assert predictions.empty and tuple(predictions.columns) == PREDICTION_COLUMNS
    assert automatic.empty and tuple(automatic.columns) == PREDICTION_COLUMNS
    assert decision["status"] == "HOLD"
    assert decision["reason_code"] == "IMPOSSIBLE_TWO_WAY_GROUP_SPLIT"


def test_corrupt_input_fails_and_cannot_leave_stale_success_artifacts(tmp_path):
    fixture = tmp_path / "corrupt"
    (fixture / "images").mkdir(parents=True)
    (fixture / "images" / "bad.png").write_bytes(b"not an image")
    pd.DataFrame(
        [
            {
                "image_id": "bad",
                "path": "images/bad.png",
                "camera_id": "cam",
                "captured_at": "2026-01-01T00:00:00Z",
            }
        ]
    ).to_csv(fixture / "images.csv", index=False)
    write_manifest(
        build_manifest(
            fixture,
            ["images.csv", "images/bad.png"],
            source="generated-test",
            license="CC0",
        ),
        fixture / "manifest.csv",
    )
    output = tmp_path / "output"
    output.mkdir()
    (output / "suspect_pairs.csv").write_text(
        "img_a,img_b,score,pred_dup\nold_a,old_b,1,1\n", encoding="utf-8"
    )
    (output / "decision.json").write_text('{"status":"GO"}', encoding="utf-8")

    assert _run(fixture, output) == 2
    assert not (output / "suspect_pairs.csv").exists()
    assert not (output / "decision.json").exists()
    info = json.loads((output / "run_info.json").read_text(encoding="utf-8"))
    assert info["status"] == 2


def test_public_fixture_reviewed_rerun_changes_final_and_audits(tmp_path):
    output = tmp_path / "poc"
    fixture = Path("fixtures/t05_near_duplicate")
    assert _run(fixture, output) == 0

    automatic_before = (output / "automatic_pairs.csv").read_bytes()
    queue = pd.read_csv(output / "review_queue.csv", dtype=str).fillna("")
    evidence = pd.read_csv(output / "pair_evidence.csv", dtype={"pair_id": str})
    groups = pd.read_csv(output / "groups.csv", dtype=str)
    assert len(queue) > 0
    duplicate_size = groups.groupby("duplicate_group_id").size().to_dict()
    group_of = dict(zip(groups.image_id, groups.duplicate_group_id))
    evidence_by_id = evidence.set_index("pair_id")

    chosen = None
    for item_id in queue.item_id:
        row = evidence_by_id.loc[item_id]
        if row.classification != "positive":
            # Prefer a forced relation between automatic singleton groups so the
            # reviewed pair necessarily changes group-derived prediction output.
            if duplicate_size[group_of[row.img_a]] == duplicate_size[group_of[row.img_b]] == 1:
                chosen = (item_id, "approve", "")
                break
    if chosen is None:
        for item_id in queue.item_id:
            row = evidence_by_id.loc[item_id]
            if not bool(row.exact_match) and row.classification == "positive":
                chosen = (item_id, "reject", "same_scene_only")
                break
    assert chosen is not None
    item_id, action, override = chosen
    reviewed = queue.copy()
    mask = reviewed.item_id == item_id
    reviewed.loc[mask, "reviewer"] = "integration-reviewer"
    reviewed.loc[mask, "reviewer_decision"] = action
    reviewed.loc[mask, "override_value"] = override
    reviewed.loc[mask, "review_note"] = "Task 12 integration review decision"
    reviewed.loc[mask, "reviewed_at"] = "2026-11-15T12:00:00Z"
    assert tuple(reviewed.columns) == PAIR_REVIEW_COLUMNS
    reviewed.to_csv(output / "review_queue_reviewed.csv", index=False)

    assert _run(fixture, output) == 0
    assert (output / "automatic_pairs.csv").read_bytes() == automatic_before
    final = pd.read_csv(output / "suspect_pairs.csv")
    selected = evidence_by_id.loc[item_id]
    final_value = final.set_index(["img_a", "img_b"]).at[
        (selected.img_a, selected.img_b), "pred_dup"
    ]
    assert final_value == (1 if action == "approve" else 0)
    audit_lines = [
        json.loads(line)
        for line in (output / "audit_log.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(event["item_id"] == item_id for event in audit_lines)
    run_info = json.loads((output / "run_info.json").read_text(encoding="utf-8"))
    assert run_info["stats"]["n_reviewed"] == 1
    assert run_info["stats"]["n_final_prediction_changes"] == int(
        (final.pred_dup != pd.read_csv(output / "automatic_pairs.csv").pred_dup).sum()
    )
    assert run_info["stats"]["n_final_prediction_changes"] > 0

    # A reviewed export is bound to the original seed/config. Failure must clear
    # the previous success prediction rather than allowing main.py to reuse it.
    assert _run(fixture, output, seed=43) == 2
    assert not (output / "suspect_pairs.csv").exists()
    assert (output / "review_context.json").exists()
