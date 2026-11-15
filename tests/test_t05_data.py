"""Task 2 tests for T05 input and provenance validation."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from src.common.manifest import build_manifest, write_manifest
from src.t05.data import load_inputs, resolve_image_path, validate_image_table


def _write_image(path: Path, color=(10, 20, 30)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 6), color).save(path, format="JPEG")


def _valid_table(path: str = "images/im_001.jpg") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_id": ["im_001"],
            "path": [path],
            "camera_id": ["cam_0"],
            "captured_at": ["2026-10-01T08:00:00.200000"],
        }
    )


def _write_fixture(root: Path) -> Path:
    _write_image(root / "images/im_001.jpg")
    _valid_table().to_csv(root / "images.csv", index=False)
    manifest = build_manifest(
        root,
        ["images.csv", "images/im_001.jpg"],
        source="unit-test",
        license="CC0-1.0",
    )
    write_manifest(manifest, root / "manifest.csv")
    return root


def test_load_inputs_validates_fixture_and_preserves_string_ids(tmp_path):
    root = _write_fixture(tmp_path)
    loaded = load_inputs(root)

    assert loaded["images"].image_id.tolist() == ["im_001"]
    assert loaded["images"].path.tolist() == ["images/im_001.jpg"]
    assert pd.api.types.is_datetime64_any_dtype(loaded["images"].captured_at)
    assert str(loaded["images"].captured_at.dt.tz) == "UTC"
    assert loaded["provenance"] == {
        "strict": True,
        "verified": True,
        "problems": [],
        "n_images": 1,
        "n_manifest_rows": 2,
    }


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda df: df.drop(columns="camera_id"), "missing required columns"),
        (lambda df: pd.concat([df, df], ignore_index=True), "duplicate image_id"),
        (lambda df: df.assign(image_id="  "), "blank values"),
        (lambda df: df.assign(captured_at="01/02/2026 08:00"), "invalid ISO-8601"),
        (lambda df: df.assign(camera_id=""), "blank values"),
    ],
)
def test_validate_image_table_rejects_bad_metadata(tmp_path, change, message):
    _write_image(tmp_path / "images/im_001.jpg")
    with pytest.raises(ValueError, match=message):
        validate_image_table(change(_valid_table()), tmp_path)


def test_validate_image_table_rejects_missing_image(tmp_path):
    with pytest.raises(ValueError, match="file not found"):
        validate_image_table(_valid_table(), tmp_path)


def test_validate_image_table_rejects_corrupt_image(tmp_path):
    path = tmp_path / "images/im_001.jpg"
    path.parent.mkdir()
    path.write_text("not an image", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot be decoded"):
        validate_image_table(_valid_table(), tmp_path)


def test_resolve_image_path_rejects_escape_and_absolute_path(tmp_path):
    outside = tmp_path.parent / "outside.jpg"
    _write_image(outside)
    with pytest.raises(ValueError, match="escapes input directory"):
        resolve_image_path(tmp_path, "../outside.jpg")
    with pytest.raises(ValueError, match="must be relative"):
        resolve_image_path(tmp_path, outside.resolve())


def test_validate_image_table_rejects_duplicate_resolved_path(tmp_path):
    _write_image(tmp_path / "images/im_001.jpg")
    table = pd.concat(
        [_valid_table(), _valid_table().assign(image_id="im_002")], ignore_index=True
    )
    with pytest.raises(ValueError, match="same path"):
        validate_image_table(table, tmp_path)


def test_checksum_mismatch_fails_in_strict_mode(tmp_path):
    root = _write_fixture(tmp_path)
    # Keep the file decodeable while changing its checksum after manifest creation.
    _write_image(root / "images/im_001.jpg", color=(200, 10, 10))
    with pytest.raises(ValueError, match="hash_mismatch"):
        load_inputs(root, strict_provenance=True)


def test_checksum_mismatch_is_diagnostic_in_non_strict_mode(tmp_path):
    root = _write_fixture(tmp_path)
    _write_image(root / "images/im_001.jpg", color=(200, 10, 10))
    loaded = load_inputs(root, strict_provenance=False)
    assert loaded["provenance"]["verified"] is False
    assert any(p["problem"] == "hash_mismatch" for p in loaded["provenance"]["problems"])


def test_manifest_must_cover_every_image(tmp_path):
    root = _write_fixture(tmp_path)
    manifest = pd.read_csv(root / "manifest.csv", dtype=str)
    manifest[manifest.file_id == "images"].to_csv(root / "manifest.csv", index=False)
    # Rebuild the metadata checksum so this test isolates image coverage.
    manifest = build_manifest(root, ["images.csv"], source="unit-test", license="CC0-1.0")
    write_manifest(manifest, root / "manifest.csv")
    with pytest.raises(ValueError, match="manifest_coverage"):
        load_inputs(root)


def test_missing_manifest_can_be_reported_non_strictly(tmp_path):
    _write_image(tmp_path / "images/im_001.jpg")
    _valid_table().to_csv(tmp_path / "images.csv", index=False)
    loaded = load_inputs(tmp_path, strict_provenance=False)
    assert loaded["manifest"].empty
    assert loaded["provenance"]["problems"][0]["problem"] == "missing_manifest"
    with pytest.raises(ValueError, match="missing_manifest"):
        load_inputs(tmp_path, strict_provenance=True)


def test_manifest_path_cannot_escape_root_even_non_strict(tmp_path):
    root = _write_fixture(tmp_path)
    manifest = pd.read_csv(root / "manifest.csv", dtype=str, keep_default_na=False)
    manifest.loc[manifest.file_id == "im_001", "path"] = "../outside.jpg"
    manifest.to_csv(root / "manifest.csv", index=False)
    loaded = load_inputs(root, strict_provenance=False)
    kinds = {problem["problem"] for problem in loaded["provenance"]["problems"]}
    assert "unsafe_manifest_path" in kinds
    assert "hash_verification_skipped" in kinds


def test_manifest_directory_path_is_a_provenance_problem(tmp_path):
    root = _write_fixture(tmp_path)
    manifest = pd.read_csv(root / "manifest.csv", dtype=str, keep_default_na=False)
    manifest.loc[manifest.file_id == "im_001", "path"] = "images"
    manifest.to_csv(root / "manifest.csv", index=False)
    loaded = load_inputs(root, strict_provenance=False)
    kinds = {problem["problem"] for problem in loaded["provenance"]["problems"]}
    assert "non_file_manifest_path" in kinds
    assert "hash_verification_skipped" in kinds


def test_repository_fixture_passes_strict_validation():
    loaded = load_inputs("fixtures/t05_near_duplicate", strict_provenance=True)
    assert len(loaded["images"]) == 120
    assert loaded["provenance"]["verified"] is True
