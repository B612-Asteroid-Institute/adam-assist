#!/usr/bin/env python3
"""Verify the exact public adam-core/adam-assist release-pair contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tomllib
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[2]

CORE_CRATES = (
    "adam_core",
    "adam_core_rs_coords",
    "adam_core_rs_spice",
    "adam_core_rs_kernel_data",
)
PYTHON_REQUIREMENT = "adam-core>=0.8.0,<0.9.0"
DEPENDENCY_TABLES = {"dependencies", "dev-dependencies", "build-dependencies"}
PYTHON_CORE_NAME_IN_TEXT = re.compile(r"(?i)\badam[-_.]+core\b")
FORBIDDEN_CARGO_SOURCES = {
    "workspace",
    "path",
    "git",
    "registry",
    "registry-index",
    "branch",
    "tag",
    "rev",
}


def _cargo_dependency_entries(
    table: dict[str, Any], path: tuple[str, ...] = ()
) -> Iterator[tuple[tuple[str, ...], str, Any]]:
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
        canonical = dependency.replace("-", "_")
        if isinstance(declaration, dict) and isinstance(
            declaration.get("package"), str
        ):
            canonical = declaration["package"].replace("-", "_")
        if not canonical.startswith("adam_core"):
            continue
        version = _dependency_version(declaration)
        if version != "0.8":
            location = ".".join((*table_path, dependency))
            raise ValueError(f"{location} must require the coherent 0.8 line")
        if isinstance(declaration, dict):
            forbidden = sorted(FORBIDDEN_CARGO_SOURCES & declaration.keys())
            if forbidden:
                location = ".".join((*table_path, dependency))
                raise ValueError(
                    f"{location} must be registry-version-only, found {forbidden}"
                )
        found.add((table_path, canonical))

    required: set[tuple[tuple[str, ...], str]] = {
        (("dev-dependencies",), "adam_core"),
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


def _forbidden_cargo_override_paths(
    table: dict[str, Any], path: tuple[str, ...] = ()
) -> list[str]:
    found: list[str] = []
    for key, value in table.items():
        current = (*path, key)
        if key in {"patch", "replace"}:
            found.append(".".join(current))
        if isinstance(value, dict):
            found.extend(_forbidden_cargo_override_paths(value, current))
    return found


def _canonical_python_name(value: str) -> str | None:
    match = re.match(r"\s*([A-Za-z0-9]+(?:[-_.]+[A-Za-z0-9]+)*)", value)
    if match is None:
        return None
    return re.sub(r"[-_.]+", "-", match.group(1)).lower()


def _python_core_requirement_paths(
    value: Any, path: tuple[str, ...] = ()
) -> list[tuple[tuple[str, ...], str]]:
    found: list[tuple[tuple[str, ...], str]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            current = (*path, str(key))
            normalized_key = _canonical_python_name(str(key))
            if normalized_key == "adam-core":
                found.append((current, f"{key}{child}"))
            found.extend(_python_core_requirement_paths(child, current))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_python_core_requirement_paths(child, (*path, str(index))))
    elif isinstance(value, str):
        normalized_name = _canonical_python_name(value)
        if (
            normalized_name == "adam-core"
            or PYTHON_CORE_NAME_IN_TEXT.search(value)
            or (normalized_name == "adam-assist" and "[" in value)
        ):
            found.append((path, value))
    return found


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


def verify(core_tree: Path | None = None, *, root: Path = ROOT) -> dict[str, object]:
    contract = cast(
        dict[str, object],
        json.loads((root / "migration" / "source_pair_contract.json").read_text()),
    )
    expected = {
        "schema_version": 1,
        "assist_version": "0.6.0",
        "core_version": "0.8.0",
        "core_commit": "37fcd1c6f5d38d70335e56c3b095658a466d1f50",
        "capsule": "adam_core.propagator.v2",
        "registry_state": "core-0.8.0-public-and-verified",
        "lock_state": "registry-only-core-0.8.0-locked",
        "source_pair_status": "exact-public-release",
        "release_automation": "present-and-version-alignment-required",
        "workflow_sha256": "4e4b5a8100dc7ff1ba24187ec3819441d67edb3ef745f3cf5321b6725409a838",
    }
    if contract != expected:
        raise ValueError(f"unexpected source-pair contract: {contract!r}")

    workflow = root / ".github" / "workflows" / "pip-build-lint-test-coverage.yml"
    if hashlib.sha256(workflow.read_bytes()).hexdigest() != contract["workflow_sha256"]:
        raise ValueError("canonical exact-source workflow content drifted")

    pyproject = _load_toml(root / "pyproject.toml")
    project = pyproject["project"]
    core_requirements = [
        item
        for item in project["dependencies"]
        if _canonical_python_name(item) == "adam-core"
    ]
    if core_requirements != [PYTHON_REQUIREMENT]:
        raise ValueError(
            f"expected exactly {PYTHON_REQUIREMENT!r}, got {core_requirements!r}"
        )
    all_python_core = _python_core_requirement_paths(pyproject)
    expected_path = (
        "project",
        "dependencies",
        str(project["dependencies"].index(PYTHON_REQUIREMENT)),
    )
    if all_python_core != [(expected_path, PYTHON_REQUIREMENT)]:
        raise ValueError(
            "Core requirements and Assist self-references may appear only in the "
            "exact primary dependency configuration: "
            f"{all_python_core!r}"
        )

    cargo = _load_toml(root / "rust" / "adam_assist_rs" / "Cargo.toml")
    if cargo["package"]["version"] != contract["assist_version"]:
        raise ValueError(
            "adam_assist Cargo version does not match source-pair contract"
        )
    verify_cargo_core_dependencies(cargo)
    overrides = _forbidden_cargo_override_paths(cargo)
    if overrides:
        raise ValueError(f"Cargo patch/replace tables are forbidden: {overrides!r}")

    required_locks = {
        root / "pdm.lock",
        root / "rust" / "adam_assist_rs" / "Cargo.lock",
    }
    missing_locks = sorted(
        str(path.relative_to(root)) for path in required_locks if not path.is_file()
    )
    if missing_locks:
        raise ValueError(
            "public release pairing requires authentic registry locks: "
            + ", ".join(missing_locks)
        )
    forbidden_state = {root / "uv.lock", root / "Cargo.lock"}
    forbidden_state.update(root.glob("**/.cargo/config"))
    forbidden_state.update(root.glob("**/.cargo/config.toml"))
    present = sorted(
        str(path.relative_to(root)) for path in forbidden_state if path.exists()
    )
    if present:
        raise ValueError(
            "release pairing forbids alternate locks and Cargo config overrides: "
            + ", ".join(present)
        )

    if __package__:
        from .verify_registry_locks import verify_python_lock, verify_rust_lock
    else:
        import sys

        sys.path.insert(0, str(root))
        from migration.scripts.verify_registry_locks import (
            verify_python_lock,
            verify_rust_lock,
        )
    verify_python_lock(root, str(contract["core_version"]))
    verify_rust_lock(root, str(contract["core_version"]))

    if core_tree is not None:
        core_tree = core_tree.resolve()
        if _git_head(core_tree) != contract["core_commit"]:
            raise ValueError(
                "checked-out adam-core commit does not match source-pair contract"
            )
        core_extension = _load_toml(core_tree / "rust" / "adam_core_py" / "Cargo.toml")
        if core_extension["package"]["version"] != contract["core_version"]:
            raise ValueError("checked-out adam-core extension version is not 0.8.0")
        for crate in CORE_CRATES:
            manifest = _load_toml(core_tree / "rust" / crate / "Cargo.toml")
            if manifest["package"]["version"] != contract["core_version"]:
                raise ValueError(f"checked-out {crate} version is not 0.8.0")

    return contract


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-tree", type=Path)
    args = parser.parse_args()
    contract = verify(args.core_tree)
    print(json.dumps(contract, sort_keys=True))


if __name__ == "__main__":
    main()
