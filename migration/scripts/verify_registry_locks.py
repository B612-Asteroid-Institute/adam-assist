#!/usr/bin/env python3
"""Verify that release locks select checksum-pinned public Core artifacts."""

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
    if any(not str(item.get("hash", "")).startswith("sha256:") for item in files):
        raise ValueError("pdm.lock adam-core registry hashes are missing")


def verify_rust_lock(repo: Path, core_version: str) -> None:
    lock_path = repo / "rust" / "adam_assist_rs" / "Cargo.lock"
    with lock_path.open("rb") as stream:
        lock = tomllib.load(stream)
    packages: dict[str, list[dict[str, object]]] = {}
    for package in lock["package"]:
        packages.setdefault(str(package["name"]), []).append(package)
    for name in sorted(CORE_RUST_PACKAGES):
        matches = packages.get(name, [])
        if len(matches) != 1:
            raise ValueError(
                f"Cargo.lock must contain one {name}, found {len(matches)}"
            )
        package = matches[0]
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
