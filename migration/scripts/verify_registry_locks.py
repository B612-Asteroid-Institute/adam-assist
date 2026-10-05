#!/usr/bin/env python3
"""Reject provisional source-pair locks at registry publication boundaries."""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

CORE_RUST_PACKAGES = {
    "adam_core_rs_autodiff",
    "adam_core_rs_coords",
    "adam_core_rs_kernel_data",
    "adam_core_rs_orbit_determination",
    "adam_core_rs_spice",
}
CRATES_IO_SOURCE = "registry+https://github.com/rust-lang/crates.io-index"


def _packages(path: Path) -> dict[str, dict[str, object]]:
    with path.open("rb") as stream:
        return {package["name"]: package for package in tomllib.load(stream)["package"]}


def verify_python_lock(repo: Path, core_version: str) -> None:
    core = _packages(repo / "pdm.lock")["adam-core"]
    if core.get("version") != core_version:
        raise ValueError(f"pdm.lock adam-core {core.get('version')} != {core_version}")
    provisional = {key: core[key] for key in ("git", "path", "url") if key in core}
    if provisional:
        raise ValueError(
            f"pdm.lock adam-core must be registry-only, found {provisional}"
        )
    files = core.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("pdm.lock adam-core registry files are missing")
    prefix = f"adam_core-{core_version}-"
    if any(not str(item.get("file", "")).startswith(prefix) for item in files):
        raise ValueError("pdm.lock adam-core files do not match the release version")


def verify_rust_lock(repo: Path, core_version: str) -> None:
    lock_path = repo / "rust" / "adam_assist_rs" / "Cargo.lock"
    with lock_path.open("rb") as stream:
        lock = tomllib.load(stream)
    packages = {package["name"]: package for package in lock["package"]}
    for name in sorted(CORE_RUST_PACKAGES):
        package = packages[name]
        if package.get("version") != core_version:
            raise ValueError(
                f"Cargo.lock {name} {package.get('version')} != {core_version}"
            )
        if package.get("source") != CRATES_IO_SOURCE or not package.get("checksum"):
            raise ValueError(f"Cargo.lock {name} must be registry-only")
    unused = lock.get("patch", {}).get("unused", [])
    provisional = sorted(
        package.get("name", "")
        for package in unused
        if str(package.get("name", "")).startswith("adam_core")
    )
    if provisional:
        raise ValueError(f"Cargo.lock contains provisional Core patches: {provisional}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--core-version", required=True)
    parser.add_argument("--python-lock", action="store_true")
    parser.add_argument("--rust-lock", action="store_true")
    args = parser.parse_args()
    if not args.python_lock and not args.rust_lock:
        parser.error("select --python-lock and/or --rust-lock")
    if args.python_lock:
        verify_python_lock(args.repo, args.core_version)
    if args.rust_lock:
        verify_rust_lock(args.repo, args.core_version)
    print(f"verified registry-only Core {args.core_version} locks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
