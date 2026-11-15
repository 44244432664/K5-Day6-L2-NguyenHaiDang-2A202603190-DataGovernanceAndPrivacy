"""Task 1 tests for T05 contracts and frozen configuration."""
from __future__ import annotations

import pytest

from src.common.io import load_config
from src.t05.contracts import DEFAULT_PARAMS, canonical_pair, make_pair_id, validate_params


def test_unordered_pair_is_canonical_and_has_stable_id():
    assert canonical_pair("im_010", "im_002") == ("im_002", "im_010")
    assert make_pair_id("im_010", "im_002") == make_pair_id("im_002", "im_010")


def test_length_framing_prevents_separator_collisions():
    # A delimiter-only implementation could confuse these component boundaries.
    assert make_pair_id("a|b", "c") != make_pair_id("a", "b|c")
    assert make_pair_id("a:1", "2:b") != make_pair_id("a", "1:2:b")


@pytest.mark.parametrize("pair", [("x", "x"), ("", "x"), ("   ", "x"), (None, "x")])
def test_invalid_pair_identifiers_fail_clearly(pair):
    with pytest.raises(ValueError, match="self-pair|nonblank"):
        canonical_pair(*pair)


def test_validate_params_supplies_normalized_defaults_without_mutation():
    supplied = {"budget": 7, "test_fraction": 0.25}
    result = validate_params(supplied)
    assert supplied == {"budget": 7, "test_fraction": 0.25}
    assert result["budget"] == 7
    assert result["test_fraction"] == 0.25
    assert set(result) == set(DEFAULT_PARAMS)
    assert isinstance(result["test_fraction"], float)


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"budget": 0}, "budget"),
        ({"test_fraction": 1.0}, "test_fraction"),
        ({"phash_size": 4, "phash_distance_max": 17}, "phash_distance_max"),
        ({"histogram_distance_max": 2.1}, "histogram_distance_max"),
        ({"duplicate_score_threshold": 0.5, "review_score_low": 0.6}, "review band"),
        ({"burst_window_seconds": 61, "burst_max_span_seconds": 60}, "burst_max_span_seconds"),
        ({"budget": True}, "budget"),
        ({"unknown_threshold": 1}, "unknown T05 params"),
    ],
)
def test_invalid_params_have_actionable_errors(params, message):
    with pytest.raises(ValueError, match=message):
        validate_params(params)


def test_track_defaults_and_group_overrides_deep_merge(tmp_path):
    tracks = tmp_path / "tracks"
    tracks.mkdir()
    (tracks / "t05.yaml").write_text(
        "track: t05\nparams:\n  budget: 20\n  test_fraction: 0.3\n  phash_size: 8\n",
        encoding="utf-8",
    )
    lab = tmp_path / "lab.yaml"
    lab.write_text(
        "track_config: tracks/t05.yaml\ntrack: t05\nowner: tester\nparams:\n  budget: 9\n",
        encoding="utf-8",
    )

    cfg = load_config(lab)
    assert cfg["owner"] == "tester"
    assert cfg["params"] == {"budget": 9, "test_fraction": 0.3, "phash_size": 8}


def test_repository_t05_config_satisfies_contract():
    cfg = load_config("configs/lab.yaml")
    normalized = validate_params(cfg["params"])
    assert cfg["track"] == "t05"
    # Keep executable defaults and the documented YAML defaults synchronized.
    assert normalized == dict(DEFAULT_PARAMS)
