import importlib.util
import tomllib
from pathlib import Path

from adam_assist.version import __version__

ROOT = Path(__file__).resolve().parents[1]


def _project_dependencies() -> list[str]:
    with (ROOT / "pyproject.toml").open("rb") as pyproject_file:
        pyproject = tomllib.load(pyproject_file)
    return list(pyproject["project"]["dependencies"])


def _cargo_manifest() -> dict:
    with (ROOT / "rust" / "adam_assist_rs" / "Cargo.toml").open("rb") as cargo_file:
        return tomllib.load(cargo_file)


def test_legacy_python_assist_stack_is_not_a_runtime_dependency() -> None:
    names = {
        dependency.split("=")[0].split(">")[0] for dependency in _project_dependencies()
    }
    assert {"assist", "rebound", "ray", "spiceypy"}.isdisjoint(names)


def test_canonical_sys_crates_are_pinned_without_assist_rs() -> None:
    dependencies = _cargo_manifest()["dependencies"]
    assert "assist-rs" not in dependencies
    assert dependencies["libassist-sys"] == "=1.2.1"
    assert dependencies["librebound-sys"] == "=4.6.0"


def test_preview_dependencies_are_exact_public_releases() -> None:
    assert "adam-core==0.5.6rc2" in _project_dependencies()
    manifest = _cargo_manifest()
    dependencies = manifest["dependencies"]
    assert dependencies["adam_core_rs_coords"] == "=0.1.0-rc.2"
    assert dependencies["adam_core_rs_spice"] == "=0.1.0-rc.2"
    assert manifest["dev-dependencies"]["adam_core_rs_kernel_data"] == {
        "version": "=0.1.0-rc.2",
        "default-features": False,
    }
    assert not (ROOT / "rust" / "vendor").exists()


def test_public_extension_is_packaged_inside_adam_assist() -> None:
    with (ROOT / "pyproject.toml").open("rb") as pyproject_file:
        pyproject = tomllib.load(pyproject_file)
    assert pyproject["tool"]["maturin"]["module-name"] == "adam_assist._native"


def test_python_preview_version_matches_cargo_semver() -> None:
    script = ROOT / "migration" / "scripts" / "write_maturin_version.py"
    spec = importlib.util.spec_from_file_location("assist_write_version", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cargo_version = _cargo_manifest()["package"]["version"]
    assert cargo_version == "0.4.0-rc.2"
    assert module.cargo_version_to_pep440(cargo_version) == __version__ == "0.4.0rc2"
