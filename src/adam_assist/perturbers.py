"""
ASSIST perturber designations (DE440 + de441_n16).

ASSIST perturbs test particles using the Sun, Moon, major planets (including Pluto),
and 16 massive asteroids. Propagating one of these bodies with ASSIST is invalid
because the body would be perturbing itself. This module exposes the canonical set
of designations so callers can detect matches (e.g. by orbit_id) and attach warnings
to results without changing the propagator return type.
"""

from __future__ import annotations

from typing import overload

import numpy as np
import numpy.typing as npt

# DE440: Sun, Moon, Mercury–Neptune, Pluto. de441_n16: 16 massive asteroids.
# Canonical orbit_id-like designations mapped to NAIF body IDs for warning/reporting.
PERTURBER_NAIF_IDS: dict[str, int] = {
    "sun": 10,
    "mercury": 199,
    "venus": 299,
    "earth": 399,
    "mars": 499,
    "jupiter": 599,
    "saturn": 699,
    "uranus": 799,
    "neptune": 899,
    "moon": 301,
    "pluto": 999,
    "134340": 999,
    "1": 2000001,
    "ceres": 2000001,
    "2": 2000002,
    "pallas": 2000002,
    "3": 2000003,
    "juno": 2000003,
    "4": 2000004,
    "vesta": 2000004,
    "5": 2000005,
    "astraea": 2000005,
    "6": 2000006,
    "hebe": 2000006,
    "7": 2000007,
    "iris": 2000007,
    "8": 2000008,
    "flora": 2000008,
    "9": 2000009,
    "metis": 2000009,
    "10": 2000010,
    "hygiea": 2000010,
    "11": 2000011,
    "parthenope": 2000011,
    "12": 2000012,
    "victoria": 2000012,
    "13": 2000013,
    "egeria": 2000013,
    "14": 2000014,
    "irene": 2000014,
    "15": 2000015,
    "eunomia": 2000015,
    "16": 2000016,
    "psyche": 2000016,
}
PERTURBER_DESIGNATIONS: frozenset[str] = frozenset(PERTURBER_NAIF_IDS)


def get_perturber_designations() -> frozenset[str]:
    """Return the set of orbit_id-like strings that are ASSIST perturbers (read-only)."""
    return PERTURBER_DESIGNATIONS


def get_perturber_naif_ids() -> dict[str, int]:
    """Return a copy of canonical designation -> NAIF body ID mappings."""
    return dict(PERTURBER_NAIF_IDS)


def _normalize(orbit_id: str) -> str:
    """
    Normalize for set lookup: strip, lower, then use first token.
    Aligns with designation-style ids (e.g. SBDB fullname "16 Psyche" -> "16",
    precovery normalize_designation). When normalize=False we do not use this.
    """
    s = str(orbit_id).strip().lower()
    if not s:
        return s
    parts = s.split()
    return parts[0] if parts else s


def _match_one(orbit_id: str | None | float, *, normalize: bool) -> str | None:
    """Return the matched perturber designation or None. Treats None, nan, and empty as non-match."""
    if orbit_id is None:
        return None
    if isinstance(orbit_id, float) and np.isnan(orbit_id):
        return None
    if not str(orbit_id).strip():
        return None
    s = _normalize(str(orbit_id)) if normalize else str(orbit_id).strip()
    return s if s in PERTURBER_DESIGNATIONS else None


@overload
def is_perturber(
    orbit_id: str,
    *,
    normalize: bool = True,
) -> str | None: ...


@overload
def is_perturber(
    orbit_id: npt.NDArray[np.str_] | npt.NDArray[np.object_],
    *,
    normalize: bool = True,
) -> npt.NDArray[np.object_]: ...


def is_perturber(
    orbit_id: str | npt.NDArray[np.str_] | npt.NDArray[np.object_],
    *,
    normalize: bool = True,
) -> str | None | npt.NDArray[np.object_]:
    """Return the matched ASSIST perturber designation, or None if not a perturber.

    None, nan, and empty string are treated as non-match (return None). Accepts a single
    orbit_id (str) or an array of orbit_ids. For a single id returns str | None; for an
    array returns an array of the same shape with str | None per element.

    When normalize is True (default), each id is stripped, lowercased, and the first
    token is used for lookup so that SBDB-style fullnames (e.g. "16 Psyche") and
    precovery-style orbit_ids (e.g. "16", "Psyche") all match. When normalize is
    False, only exact set membership is checked.

    Notes
    -----
    SBDB (query_sbdb / query_sbdb_new) returns orbit_id as a zero-padded row index
    ("00000", "00001", ...), not the object name. That value has no object meaning,
    so is_perturber(orbit_id) will not identify perturbers when given raw SBDB
    orbit_id. Use object_id (fullname, e.g. "16 Psyche", "4 Vesta") or a canonical
    orbit_id derived from it (e.g. precovery's normalize_designation or
    _canonical_orbit_id(orbit_id, object_id)) when checking SBDB-sourced orbits.
    With first-token normalization, object_id forms like "16 Psyche", "Psyche", "16"
    match; "2019 QU127" -> "2019" and "3753 Cruithne (1986 TO)" -> "3753" correctly
    do not match (not in the perturber set).
    """
    if isinstance(orbit_id, str):
        return _match_one(orbit_id, normalize=normalize)
    arr = np.asarray(orbit_id, dtype=object)
    if arr.size == 0:
        return np.empty(arr.shape, dtype=object)
    # Treat None, nan, and empty string as non-match (return None for those elements).
    invalid = np.array(
        [
            x is None
            or (isinstance(x, float) and np.isnan(x))
            or (isinstance(x, str) and x.strip() == "")
            for x in arr.ravel()
        ],
        dtype=bool,
    ).reshape(arr.shape)
    arr_str = arr.astype(str)
    if normalize:
        normalized = np.array(
            [_normalize(str(x)) for x in arr_str.ravel()], dtype=object
        ).reshape(arr.shape)
    else:
        normalized = arr_str
    perturber_arr = np.fromiter(PERTURBER_DESIGNATIONS, dtype=object)
    mask = np.isin(normalized, perturber_arr) & ~invalid
    result = np.empty(arr.shape, dtype=object)
    result[mask] = normalized[mask]
    result[~mask] = None
    return result


@overload
def get_perturber_naif_id(
    orbit_id: str | None | float,
    *,
    normalize: bool = True,
) -> int | None: ...


@overload
def get_perturber_naif_id(
    orbit_id: npt.NDArray[np.str_] | npt.NDArray[np.object_],
    *,
    normalize: bool = True,
) -> npt.NDArray[np.object_]: ...


def get_perturber_naif_id(
    orbit_id: str | None | float | npt.NDArray[np.str_] | npt.NDArray[np.object_],
    *,
    normalize: bool = True,
) -> int | None | npt.NDArray[np.object_]:
    """Return the NAIF body ID for matched perturber designation(s), else None."""
    if isinstance(orbit_id, np.ndarray):
        matched = is_perturber(orbit_id, normalize=normalize)
        flat = matched.ravel()
        return np.array(
            [PERTURBER_NAIF_IDS[x] if isinstance(x, str) else None for x in flat],
            dtype=object,
        ).reshape(matched.shape)
    matched_scalar = _match_one(orbit_id, normalize=normalize)
    return PERTURBER_NAIF_IDS[matched_scalar] if matched_scalar is not None else None


def unique_perturber_ids_from_candidates(
    candidate_ids: npt.NDArray[np.object_],
) -> npt.NDArray[np.object_]:
    """Return the unique subset of candidate_ids that are ASSIST perturbers (1d, possibly empty)."""
    ids = np.asarray(candidate_ids, dtype=object).ravel()
    if ids.size == 0:
        return np.array([], dtype=object)
    matched = is_perturber(ids)
    non_none = matched[matched != None]  # noqa: E711
    return np.unique(non_none) if non_none.size > 0 else np.array([], dtype=object)
