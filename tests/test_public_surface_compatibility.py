from __future__ import annotations

import importlib
import inspect
import json
from pathlib import Path

from adam_core.propagator.propagator import Propagator

import adam_assist
from adam_assist.propagator import ASSISTPropagator
from adam_assist.version import __version__

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads(
    (ROOT / "migration/public_surface/adam_assist_0_3_10.json").read_text()
)
CURRENT_MANIFEST = json.loads(
    (ROOT / "migration/public_surface/manifest.json").read_text()
)


def test_published_0310_modules_remain_importable() -> None:
    for module in MANIFEST["modules"]:
        importlib.import_module(module)


def test_assist_propagator_preserves_the_public_base_contract() -> None:
    assert issubclass(ASSISTPropagator, Propagator)
    assert not inspect.isabstract(ASSISTPropagator)
    assert hasattr(ASSISTPropagator, "_propagate_orbits")
    for method in MANIFEST["legacy_class_methods"]["ASSISTPropagator"]:
        assert hasattr(ASSISTPropagator, method)
    for hook in MANIFEST["private_compatibility_hooks"]:
        class_name, method = hook.split(".", 1)
        assert class_name == "ASSISTPropagator"
        assert hasattr(ASSISTPropagator, method)


def test_published_0310_root_exports_are_exactly_preserved() -> None:
    assert adam_assist.__all__ == MANIFEST["root_exports"]
    for symbol in MANIFEST["root_exports"]:
        assert hasattr(adam_assist, symbol)


def test_published_0310_module_symbols_are_preserved() -> None:
    perturbers = importlib.import_module("adam_assist.perturbers")
    propagator = importlib.import_module("adam_assist.propagator")
    for symbol in MANIFEST["perturber_symbols"]:
        assert hasattr(perturbers, symbol)
    for symbol in MANIFEST["propagator_symbols"]:
        assert hasattr(propagator, symbol)


def _parameters(method: object) -> list[inspect.Parameter]:
    return list(inspect.signature(method).parameters.values())


def test_propagation_signature_preserves_legacy_prefix_and_latest_extensions() -> None:
    parameters = _parameters(ASSISTPropagator.propagate_orbits)
    assert [parameter.name for parameter in parameters] == [
        "self",
        "orbits",
        "times",
        "covariance",
        "covariance_method",
        "num_samples",
        "chunk_size",
        "max_processes",
        "seed",
        "include_nongrav",
    ]
    assert [parameter.default for parameter in parameters[3:]] == [
        False,
        "monte-carlo",
        1000,
        100,
        1,
        None,
        True,
    ]


def test_legacy_ephemeris_positional_prefix_and_defaults_are_preserved() -> None:
    parameters = _parameters(ASSISTPropagator.generate_ephemeris)
    legacy = parameters[:11]
    assert [parameter.name for parameter in legacy] == [
        "self",
        "orbits",
        "observers",
        "covariance",
        "covariance_method",
        "num_samples",
        "chunk_size",
        "max_processes",
        "seed",
        "predict_magnitudes",
        "predict_phase_angle",
    ]
    assert all(
        parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        for parameter in legacy
    )
    assert [parameter.default for parameter in legacy[3:]] == [
        False,
        "monte-carlo",
        1000,
        100,
        1,
        None,
        True,
        False,
    ]
    assert parameters[11].name == "include_nongrav"
    assert parameters[11].default is True


def test_legacy_collision_signature_is_preserved() -> None:
    parameters = _parameters(ASSISTPropagator.detect_collisions)
    assert [parameter.name for parameter in parameters] == [
        "self",
        "orbits",
        "num_days",
        "conditions",
        "max_processes",
        "chunk_size",
    ]
    assert [parameter.default for parameter in parameters[3:]] == [None, 1, 100]


def test_legacy_constructor_parameters_remain_accepted() -> None:
    parameters = _parameters(ASSISTPropagator.__init__)
    names = [parameter.name for parameter in parameters]
    assert names[0:2] == ["self", "args"]
    assert names[-1] == "kwargs"
    for name, default in [
        ("min_dt", 1e-9),
        ("initial_dt", 1e-6),
        ("adaptive_mode", 1),
        ("epsilon", 1e-6),
    ]:
        assert parameters[names.index(name)].default == default


def test_current_surface_manifest_reconciles_latest_upstream() -> None:
    summary = CURRENT_MANIFEST["summary"]
    assert CURRENT_MANIFEST["scope"]["frozen_updated_upstream_commit"] == (
        "cb5bb14b5c1c6b27f43595d327f1a7b3f819e5c2"
    )
    assert summary == {
        "symbols": 25,
        "frozen_updated_upstream_symbols": 14,
        "missing_upstream_symbols": 0,
        "latest_extension_symbols": 11,
        "all_non_plotting_symbols_classified": True,
    }
    symbols = CURRENT_MANIFEST["symbols"]
    assert len(symbols) == summary["symbols"]
    assert len({symbol["id"] for symbol in symbols}) == len(symbols)
    assert all(symbol["classification"] for symbol in symbols)
    assert set(CURRENT_MANIFEST["correctness_classification"]) == {
        "bitwise",
        "tolerance-based",
        "statistical",
    }


def test_rc_version_module_is_preserved() -> None:
    assert __version__ == "0.4.0rc5"
