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
    dependency = _cargo_manifest()["dependencies"]["assist-rs"]
    assert dependency["rev"] == "f66517aa9b920ad250312aceb28a2635cdb999a5"
    assert dependency["default-features"] is False
    assert dependency["features"] == ["parallel"]


def test_public_extension_is_packaged_inside_adam_assist() -> None:
    with (ROOT / "pyproject.toml").open("rb") as pyproject_file:
        pyproject = tomllib.load(pyproject_file)
    assert pyproject["tool"]["maturin"]["module-name"] == "adam_assist._native"
