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
    assert "rebound>=4.4.11,<5" in _project_dependencies()
