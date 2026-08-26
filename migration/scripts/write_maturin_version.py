"""Mirror the adam-assist Cargo SemVer into Python PEP 440 version metadata."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CARGO_TOML = ROOT / "rust" / "adam_assist_rs" / "Cargo.toml"
VERSION_FILE = ROOT / "src" / "adam_assist" / "version.py"
SEMVER = re.compile(
    r"^(?P<release>(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\."
    r"(?:0|[1-9][0-9]*))(?:-(?P<phase>alpha|beta|rc)\."
    r"(?P<number>0|[1-9][0-9]*))?$"
)


def cargo_version_to_pep440(version: str) -> str:
    match = SEMVER.fullmatch(version)
    if match is None:
        raise ValueError(
            f"Cargo version {version!r} must be X.Y.Z or " "X.Y.Z-(alpha|beta|rc).N"
        )
    release = match.group("release")
    phase = match.group("phase")
    if phase is None:
        return release
    pep_phase = {"alpha": "a", "beta": "b", "rc": "rc"}[phase]
    return f"{release}{pep_phase}{match.group('number')}"


def main() -> None:
    with CARGO_TOML.open("rb") as source:
        cargo_version = tomllib.load(source)["package"]["version"]
    python_version = cargo_version_to_pep440(cargo_version)
    VERSION_FILE.write_text(f'__version__ = "{python_version}"\n', encoding="utf-8")
    print(f"Wrote {VERSION_FILE.relative_to(ROOT)} = {python_version}")


if __name__ == "__main__":
    main()
