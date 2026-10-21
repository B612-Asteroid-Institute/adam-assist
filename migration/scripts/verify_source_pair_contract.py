#!/usr/bin/env python3
"""Verify the explicit pre-release adam-core/adam-assist source-pair contract."""

from __future__ import annotations

import argparse
import json
import subprocess
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

CORE_CRATES = (
    "adam_core_rs_coords",
    "adam_core_rs_spice",
    "adam_core_rs_kernel_data",
)
CONTRACT_PATH = ROOT / "migration" / "source_pair_contract.json"
PYTHON_REQUIREMENT = "adam-core>=0.7.0,<0.8.0"
DEPENDENCY_TABLES = {"dependencies", "dev-dependencies", "build-dependencies"}
FORBIDDEN_CARGO_SOURCES = {
    "path",
    "git",
    "registry",
    "registry-index",
    "branch",
    "tag",
    "rev",
}


def _cargo_dependency_entries(table: dict[str, Any], path: tuple[str, ...] = ()):
    for key, value in table.items():
        current = (*path, key)
        if key in DEPENDENCY_TABLES and isinstance(value, dict):
            for dependency, declaration in value.items():
                yield current, dependency, declaration
        elif isinstance(value, dict):
            yield from _cargo_dependency_entries(value, current)


def verify_cargo_core_dependencies(cargo: dict[str, Any]) -> None:
    found: set[tuple[tuple[str, ...], str]] = set()
    for table_path, dependency, declaration in _cargo_dependency_entries(cargo):
        canonical = dependency
        if isinstance(declaration, dict) and isinstance(
            declaration.get("package"), str
        ):
            canonical = declaration["package"]
        if not canonical.startswith("adam_core"):
            continue
        version = _dependency_version(declaration)
        if version != "0.7":
            location = ".".join((*table_path, dependency))
            raise ValueError(f"{location} must require the coherent 0.7 line")
        if isinstance(declaration, dict):
            forbidden = sorted(FORBIDDEN_CARGO_SOURCES & declaration.keys())
            if forbidden:
                location = ".".join((*table_path, dependency))
                raise ValueError(
                    f"{location} must be registry-version-only, found {forbidden}"
                )
        found.add((table_path, canonical))

    required = {
        (("dependencies",), "adam_core_rs_coords"),
        (("dependencies",), "adam_core_rs_spice"),
        (("dependencies",), "adam_core_rs_kernel_data"),
        (("dev-dependencies",), "adam_core_rs_kernel_data"),
    }
    missing = required - found
    if missing:
        raise ValueError(
            f"missing required Core dependency declarations: {sorted(missing)!r}"
        )


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def _dependency_version(value: object) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        version = value.get("version")
        return version if isinstance(version, str) else None
    return None


def _git_head(repository: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()


def verify(core_tree: Path | None = None) -> dict[str, object]:
    contract = json.loads(CONTRACT_PATH.read_text())
    expected = {
        "schema_version": 1,
        "assist_version": "0.6.0",
        "core_version": "0.7.0",
        "core_commit": "f0f1aec8b3c059f0cc4af054b5ebe8ed7760da19",
        "capsule": "adam_core.propagator.v2",
        "registry_state": "awaiting-core-0.7.0-publication",
        "lock_state": "absent-until-core-0.7.0-is-public",
    }
    if contract != expected:
        raise ValueError(f"unexpected source-pair contract: {contract!r}")

    pyproject = _load_toml(ROOT / "pyproject.toml")
    project = pyproject["project"]
    core_requirements = [
        item for item in project["dependencies"] if item.startswith("adam-core")
    ]
    if core_requirements != [PYTHON_REQUIREMENT]:
        raise ValueError(
            f"expected exactly {PYTHON_REQUIREMENT!r}, got {core_requirements!r}"
        )

    cargo = _load_toml(ROOT / "rust" / "adam_assist_rs" / "Cargo.toml")
    if cargo["package"]["version"] != contract["assist_version"]:
        raise ValueError(
            "adam_assist Cargo version does not match source-pair contract"
        )
    verify_cargo_core_dependencies(cargo)

    forbidden_locks = (
        ROOT / "pdm.lock",
        ROOT / "rust" / "adam_assist_rs" / "Cargo.lock",
    )
    present = [str(path.relative_to(ROOT)) for path in forbidden_locks if path.exists()]
    if present:
        raise ValueError(
            "pre-release source pairing must not retain stale registry locks: "
            + ", ".join(present)
        )

    if core_tree is not None:
        core_tree = core_tree.resolve()
        if _git_head(core_tree) != contract["core_commit"]:
            raise ValueError(
                "checked-out adam-core commit does not match source-pair contract"
            )
        core_extension = _load_toml(core_tree / "rust" / "adam_core_py" / "Cargo.toml")
        if core_extension["package"]["version"] != contract["core_version"]:
            raise ValueError("checked-out adam-core extension version is not 0.7.0")
        for crate in CORE_CRATES:
            manifest = _load_toml(core_tree / "rust" / crate / "Cargo.toml")
            if manifest["package"]["version"] != contract["core_version"]:
                raise ValueError(f"checked-out {crate} version is not 0.7.0")

    return contract


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-tree", type=Path)
    args = parser.parse_args()
    contract = verify(args.core_tree)
    print(json.dumps(contract, sort_keys=True))


if __name__ == "__main__":
    main()
