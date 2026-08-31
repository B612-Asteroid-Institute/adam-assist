#!/usr/bin/env python3
"""Verify one exact adam-assist source and public Rust prerequisite release."""

from __future__ import annotations

import argparse
import ast
import tomllib
from pathlib import Path

try:
    from .publish_crate_archive import (
        CRATE_NAME,
        cargo_package,
        published_crate_entry,
        validate_existing_archive,
        validate_package,
    )
    from .write_maturin_version import cargo_version_to_pep440
except ImportError:
    from publish_crate_archive import (
        CRATE_NAME,
        cargo_package,
        published_crate_entry,
        validate_existing_archive,
        validate_package,
    )
    from write_maturin_version import cargo_version_to_pep440


def is_python_prerelease(version: str) -> bool:
    return any(marker in version for marker in ("a", "b", "rc"))


def verify(
    repo: Path,
    *,
    python_version: str,
    rust_version: str,
    core_python_version: str,
    core_rust_version: str,
    channel: str,
    rust_checksum: str,
    registry_index: str = "https://index.crates.io",
) -> None:
    python_prerelease = is_python_prerelease(python_version)
    core_python_prerelease = is_python_prerelease(core_python_version)
    if channel == "preview" and not (python_prerelease and core_python_prerelease):
        raise ValueError("preview Python package and Core versions must be prereleases")
    if channel == "stable" and (python_prerelease or core_python_prerelease):
        raise ValueError(
            "stable Python package and Core versions must not be prereleases"
        )

    manifest_path = repo / "rust" / "adam_assist_rs" / "Cargo.toml"
    package = cargo_package(manifest_path)
    validate_package(package, rust_version, core_rust_version, channel)
    if cargo_version_to_pep440(rust_version) != python_version:
        raise ValueError(
            f"Python/Rust source version mismatch: {python_version} != {rust_version}"
        )

    with (repo / "pyproject.toml").open("rb") as source:
        project = tomllib.load(source)["project"]
    expected_dependency = f"adam-core=={core_python_version}"
    if expected_dependency not in project["dependencies"]:
        raise ValueError(f"missing exact Core dependency {expected_dependency}")
    version_source = (repo / "src" / "adam_assist" / "version.py").read_text()
    prefix = "__version__ = "
    if not version_source.startswith(prefix):
        raise ValueError("Python runtime version file has an unexpected format")
    runtime_version = ast.literal_eval(version_source.removeprefix(prefix).strip())
    if runtime_version != python_version:
        raise ValueError("Python runtime version does not match the release version")
    if (repo / "rust" / "vendor").exists():
        raise ValueError("temporary vendored Core crates must be absent")

    entry = published_crate_entry(registry_index, CRATE_NAME, rust_version)
    if entry is None:
        raise ValueError(f"public {CRATE_NAME} {rust_version} does not exist")
    validate_existing_archive(entry, CRATE_NAME, rust_version, rust_checksum)
    if entry.get("rust_version") != "1.87":
        raise ValueError(
            f"unexpected public {CRATE_NAME} MSRV: {entry.get('rust_version')}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--python-version", required=True)
    parser.add_argument("--rust-version", required=True)
    parser.add_argument("--core-python-version", required=True)
    parser.add_argument("--core-rust-version", required=True)
    parser.add_argument("--channel", choices=("preview", "stable"), required=True)
    parser.add_argument("--rust-checksum", required=True)
    parser.add_argument("--registry-index", default="https://index.crates.io")
    args = parser.parse_args()
    verify(
        args.repo.resolve(),
        python_version=args.python_version,
        rust_version=args.rust_version,
        core_python_version=args.core_python_version,
        core_rust_version=args.core_rust_version,
        channel=args.channel,
        rust_checksum=args.rust_checksum,
        registry_index=args.registry_index,
    )
    print(
        f"verified {args.channel} adam-assist {args.python_version}/"
        f"{args.rust_version} with Core {args.core_python_version}/"
        f"{args.core_rust_version} and exact public Rust checksum"
    )


if __name__ == "__main__":
    main()
