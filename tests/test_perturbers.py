"""Tests for adam_assist.perturbers."""

from __future__ import annotations

import numpy as np

from adam_assist.perturbers import (
    get_perturber_designations,
    is_perturber,
    unique_perturber_ids_from_candidates,
)


def test_get_perturber_designations_returns_frozenset() -> None:
    out = get_perturber_designations()
    assert isinstance(out, frozenset)
    assert "pluto" in out
    assert "16" in out
    assert "psyche" in out
    assert "ceres" in out
    assert "sun" in out


def test_is_perturber_scalar_matches() -> None:
    assert is_perturber("pluto") == "pluto"
    assert is_perturber("Pluto") == "pluto"
    assert is_perturber("16") == "16"
    assert is_perturber("Psyche") == "psyche"
    assert is_perturber("16 Psyche") == "16"
    assert is_perturber("  4 Vesta  ") == "4"
    assert is_perturber("Ceres") == "ceres"
    assert is_perturber("1") == "1"


def test_is_perturber_scalar_non_match() -> None:
    assert is_perturber("2019 QU127") is None
    assert is_perturber("00000") is None
    assert is_perturber("random") is None


def test_is_perturber_scalar_normalize_false() -> None:
    assert is_perturber("pluto", normalize=False) == "pluto"
    assert is_perturber("Pluto", normalize=False) is None
    assert is_perturber("16", normalize=False) == "16"


def test_is_perturber_scalar_null_nan_empty() -> None:
    # None/nan are not str, so they go through the array path; result is 0-d array with None.
    r = is_perturber(None)
    assert r is None or (isinstance(r, np.ndarray) and r.ndim == 0 and r.item() is None)
    r_nan = is_perturber(np.nan)
    assert r_nan is None or (
        isinstance(r_nan, np.ndarray) and r_nan.ndim == 0 and r_nan.item() is None
    )
    assert is_perturber("") is None
    assert is_perturber("   ") is None


def test_is_perturber_array_matches() -> None:
    arr = np.array(["pluto", "16", "2019 QU127", "Psyche"], dtype=object)
    out = is_perturber(arr)
    assert out.shape == arr.shape
    assert out[0] == "pluto"
    assert out[1] == "16"
    assert out[2] is None
    assert out[3] == "psyche"


def test_is_perturber_array_first_token_semantics() -> None:
    arr = np.array(["16 Psyche", "4 Vesta"], dtype=object)
    out = is_perturber(arr)
    assert out[0] == "16"
    assert out[1] == "4"


def test_is_perturber_array_null_nan_empty() -> None:
    arr = np.array(["pluto", None, np.nan, "", "   ", "16"], dtype=object)
    out = is_perturber(arr)
    assert out[0] == "pluto"
    assert out[1] is None
    assert out[2] is None
    assert out[3] is None
    assert out[4] is None
    assert out[5] == "16"


def test_is_perturber_array_empty() -> None:
    arr = np.array([], dtype=object)
    out = is_perturber(arr)
    assert out.shape == (0,)
    assert out.dtype == object


def test_is_perturber_array_2d_shape_preserved() -> None:
    arr = np.array([["pluto", "vesta"], ["ceres", "random"]], dtype=object)
    out = is_perturber(arr)
    assert out.shape == (2, 2)
    assert out[0, 0] == "pluto"
    assert out[0, 1] == "vesta"
    assert out[1, 0] == "ceres"
    assert out[1, 1] is None


def test_unique_perturber_ids_from_candidates_empty() -> None:
    out = unique_perturber_ids_from_candidates(np.array([], dtype=object))
    assert out.size == 0
    assert out.dtype == object


def test_unique_perturber_ids_from_candidates_all_non_perturber() -> None:
    arr = np.array(["2019 QU127", "00000", "foo"], dtype=object)
    out = unique_perturber_ids_from_candidates(arr)
    assert out.size == 0


def test_unique_perturber_ids_from_candidates_mix_deduplicated() -> None:
    arr = np.array(["pluto", "16", "pluto", "Psyche", "16"], dtype=object)
    out = unique_perturber_ids_from_candidates(arr)
    assert out.size == 3
    assert set(out.tolist()) == {"pluto", "16", "psyche"}


def test_unique_perturber_ids_from_candidates_with_null_nan() -> None:
    arr = np.array([None, "pluto", np.nan, "16", ""], dtype=object)
    out = unique_perturber_ids_from_candidates(arr)
    assert out.size == 2
    assert set(out.tolist()) == {"pluto", "16"}


def test_unique_perturber_ids_from_candidates_concat_object_orbit_style() -> None:
    """Simulate propagator passing concat of object_id and orbit_id columns."""
    object_ids = np.array(["16 Psyche", "00000", "4 Vesta"], dtype=object)
    orbit_ids = np.array(["00000", "00001", "00002"], dtype=object)
    candidate_ids = np.concatenate([object_ids, orbit_ids])
    out = unique_perturber_ids_from_candidates(candidate_ids)
    assert out.size == 2
    assert set(out.tolist()) == {"16", "4"}
