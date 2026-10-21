from copy import deepcopy
from pathlib import Path

import pytest

from migration.scripts import verify_dev_environment
from migration.scripts.verify_dev_environment import FROZEN_DEV_VERSIONS
from migration.scripts.verify_source_pair_contract import (
    _load_toml,
    verify,
    verify_cargo_core_dependencies,
)


def test_source_pair_contract_accepts_current_tree() -> None:
    contract = verify()
    assert contract["assist_version"] == "0.6.0"
    assert contract["core_version"] == "0.7.0"


def test_every_cargo_dependency_table_rejects_mixed_core_versions() -> None:
    root = Path(__file__).resolve().parents[1]
    cargo = _load_toml(root / "rust" / "adam_assist_rs" / "Cargo.toml")
    mixed = deepcopy(cargo)
    mixed.setdefault("target", {}).setdefault("cfg(unix)", {})["build-dependencies"] = {
        "adam_core_rs_coords": "0.6"
    }
    with pytest.raises(ValueError, match="coherent 0.7 line"):
        verify_cargo_core_dependencies(mixed)


def test_every_cargo_dependency_table_rejects_nonregistry_core_sources() -> None:
    root = Path(__file__).resolve().parents[1]
    cargo = _load_toml(root / "rust" / "adam_assist_rs" / "Cargo.toml")
    mixed = deepcopy(cargo)
    mixed.setdefault("target", {}).setdefault("cfg(unix)", {})["dependencies"] = {
        "renamed_core": {
            "package": "adam_core_rs_spice",
            "version": "0.7",
            "git": "https://example.invalid/adam_core",
        }
    }
    with pytest.raises(ValueError, match="registry-version-only"):
        verify_cargo_core_dependencies(mixed)


def test_dev_extra_retains_all_frozen_runtime_pins() -> None:
    assert FROZEN_DEV_VERSIONS == {
        "naif-leapseconds": "2025.4.22",
        "naif-eop-predict": "2024.8.28.1",
        "naif-eop-historical": "2024.8.28.1",
        "naif-eop-high-prec": "2026.5.9",
        "naif-earth-itrf93": "2007.4.3.1",
    }


def test_dev_environment_verifier_enforces_installed_runtime_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    versions = {
        **FROZEN_DEV_VERSIONS,
        "adam-core": "0.7.0",
        "adam-assist": "0.6.0",
    }
    imported: list[str] = []
    monkeypatch.setattr(
        verify_dev_environment.importlib.metadata,
        "version",
        versions.__getitem__,
    )
    monkeypatch.setattr(
        verify_dev_environment.importlib,
        "import_module",
        lambda name: imported.append(name),
    )
    verify_dev_environment.verify()
    assert imported == ["astropy"]

    versions["naif-eop-high-prec"] = "2026.10.21"
    with pytest.raises(ValueError, match="naif-eop-high-prec"):
        verify_dev_environment.verify()
