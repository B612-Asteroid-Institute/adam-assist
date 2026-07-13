import tomllib
from pathlib import Path

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


def test_assist_rs_is_pinned_to_reviewed_revision() -> None:
    # Lean binding-only assist-rs (PR #11 rework): the crate exposes no cargo
    # features anymore, so the pin is just the reviewed revision.
    dependency = _cargo_manifest()["dependencies"]["assist-rs"]
    assert dependency["rev"] == "33233c6efd7f9fb4a5e61538f5c332262f731170"
    assert "default-features" not in dependency
    assert "features" not in dependency


def test_public_extension_is_packaged_inside_adam_assist() -> None:
    with (ROOT / "pyproject.toml").open("rb") as pyproject_file:
        pyproject = tomllib.load(pyproject_file)
    assert pyproject["tool"]["maturin"]["module-name"] == "adam_assist._native"
