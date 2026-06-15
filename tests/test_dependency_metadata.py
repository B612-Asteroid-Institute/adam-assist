import tomllib
from pathlib import Path


def _project_dependencies() -> list[str]:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject_path.open("rb") as pyproject_file:
        pyproject = tomllib.load(pyproject_file)
    return list(pyproject["project"]["dependencies"])


def test_assist_dependency_uses_rebound_header_compatible_release() -> None:
    assert "assist>=1.2.3,<1.3" in _project_dependencies()


def test_rebound_dependency_stays_on_assist_supported_major_version() -> None:
    assert "rebound>=4.4.11,!=4.5.0,<5" in _project_dependencies()


def test_rebound_dependency_excludes_heap_corruption_release() -> None:
    """REBOUND 4.5.0 paired with ASSIST 1.2.3 has a destructor/lifetime
    heap-corruption bug that crashes ASSIST during propagation; it is fixed in
    4.5.1. Guard that the exclusion stays in place so a constrained resolution
    can never land on the broken release."""
    rebound_dep = next(dep for dep in _project_dependencies() if dep.startswith("rebound"))
    assert "!=4.5.0" in rebound_dep
