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
    assert contract["source_pair_status"] == "provisional"


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
    root = Path(__file__).resolve().parents[1]
    pyproject = _load_toml(root / "pyproject.toml")
    dev = pyproject["project"]["optional-dependencies"]["dev"]
    assert {"astropy", "packaging", "PyYAML"} <= set(dev)
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
    assert imported == ["astropy", "packaging", "yaml"]

    versions["naif-eop-high-prec"] = "2026.10.21"
    with pytest.raises(ValueError, match="naif-eop-high-prec"):
        verify_dev_environment.verify()


def _copy_contract_tree(tmp_path: Path) -> Path:
    import shutil

    source = Path(__file__).resolve().parents[1]
    for relative in (
        Path("pyproject.toml"),
        Path("migration/source_pair_contract.json"),
        Path("rust/adam_assist_rs/Cargo.toml"),
    ):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, destination)
    return tmp_path


def _append(relative: str, text: str):
    def mutate(root: Path) -> None:
        path = root / relative
        path.write_text(path.read_text() + text)

    return mutate


def _replace(relative: str, old: str, new: str):
    def mutate(root: Path) -> None:
        path = root / relative
        source = path.read_text()
        assert old in source
        path.write_text(source.replace(old, new, 1))

    return mutate


def _cargo_config(root: Path) -> None:
    path = root / ".cargo" / "config.toml"
    path.parent.mkdir(parents=True)
    path.write_text('[patch.crates-io]\nadam_core_rs_coords = { path = "../core" }\n')


SOURCE_CONTRACT_MUTATIONS = (
    (
        "D01-build-dependency",
        _append(
            "rust/adam_assist_rs/Cargo.toml",
            '\n[build-dependencies]\nadam_core_rs_coords = "0.6"\n',
        ),
    ),
    (
        "D02-target-path",
        _append(
            "rust/adam_assist_rs/Cargo.toml",
            "\n[target.'cfg(unix)'.dependencies]\n"
            'adam_core_rs_spice = { version = "0.7", path = "../core" }\n',
        ),
    ),
    (
        "D03-dev-git",
        _replace(
            "rust/adam_assist_rs/Cargo.toml",
            'adam_core_rs_kernel_data = { version = "0.7", default-features = false }',
            'adam_core_rs_kernel_data = { version = "0.7", git = "https://example.invalid/core" }',
        ),
    ),
    (
        "D04-workspace-source",
        _append(
            "rust/adam_assist_rs/Cargo.toml",
            '\n[workspace.dependencies]\nadam_core_rs_coords = { version = "0.7", path = "../core" }\n',
        ),
    ),
    (
        "D05-renamed-old-core",
        _append(
            "rust/adam_assist_rs/Cargo.toml",
            '\n[target.\'cfg(unix)\'.dependencies]\nold = { package = "adam_core_rs_coords", version = "0.6" }\n',
        ),
    ),
    (
        "D06-patch-table",
        _append(
            "rust/adam_assist_rs/Cargo.toml",
            '\n[patch.crates-io]\nadam_core_rs_coords = { path = "../core" }\n',
        ),
    ),
    (
        "D07-replace-table",
        _append(
            "rust/adam_assist_rs/Cargo.toml",
            '\n[replace]\n"adam_core_rs_coords:0.7.0" = { path = "../core" }\n',
        ),
    ),
    (
        "D09-alternative-registry",
        _replace(
            "rust/adam_assist_rs/Cargo.toml",
            'adam_core_rs_coords = "0.7"',
            'adam_core_rs_coords = { version = "0.7", registry = "internal" }',
        ),
    ),
    (
        "D10-hyphenated-core",
        _append(
            "rust/adam_assist_rs/Cargo.toml",
            "\n[target.'cfg(windows)'.dev-dependencies]\nadam-core-rs-coords = \"0.6\"\n",
        ),
    ),
    (
        "D11-umbrella-git",
        _append(
            "rust/adam_assist_rs/Cargo.toml",
            "\n[target.'cfg(unix)'.dependencies]\nadam_core = { git = \"https://example.invalid/core\" }\n",
        ),
    ),
    (
        "D12-second-python-core",
        _replace(
            "pyproject.toml",
            '    "adam-core>=0.7.0,<0.8.0",',
            '    "adam-core>=0.7.0,<0.8.0",\n    "adam_core==0.6.1",',
        ),
    ),
    (
        "D13-dev-extra-git",
        _replace(
            "pyproject.toml",
            '    "astropy",',
            '    "astropy",\n    "adam-core @ git+https://example.invalid/core",',
        ),
    ),
    (
        "D14-optional-extra",
        _replace(
            "pyproject.toml",
            "[project.optional-dependencies]",
            '[project.optional-dependencies]\nlegacy = ["adam-core==0.6.1"]',
        ),
    ),
    ("D15-cargo-config", _cargo_config),
    (
        "D17-pdm-dev-dependency",
        _append(
            "pyproject.toml",
            '\n[tool.pdm.dev-dependencies]\ncore = ["adam-core @ git+https://example.invalid/core"]\n',
        ),
    ),
    (
        "D18-target-build",
        _append(
            "rust/adam_assist_rs/Cargo.toml",
            '\n[target.\'cfg(target_os = "linux")\'.build-dependencies]\nadam_core_rs_spice = "0.6"\n',
        ),
    ),
    (
        "python-dependency-group",
        _append(
            "pyproject.toml",
            '\n[dependency-groups]\nlegacy = ["adam-core==0.6.1"]\n',
        ),
    ),
    (
        "python-poetry-dependency",
        _append(
            "pyproject.toml",
            '\n[tool.poetry.dependencies]\nadam-core = { git = "https://example.invalid/core" }\n',
        ),
    ),
)


@pytest.mark.parametrize(
    ("identifier", "mutate"),
    SOURCE_CONTRACT_MUTATIONS,
    ids=[item[0] for item in SOURCE_CONTRACT_MUTATIONS],
)
def test_source_contract_rejects_every_hidden_or_mixed_dependency(
    tmp_path: Path, identifier: str, mutate
) -> None:
    root = _copy_contract_tree(tmp_path)
    mutate(root)
    with pytest.raises(ValueError):
        verify(root=root)
