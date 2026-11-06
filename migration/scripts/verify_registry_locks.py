#!/usr/bin/env python3
"""Verify the authentic public Core registry locks used by Assist 0.6."""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

CORE_RUST_PACKAGES = {
    "adam_core",
    "adam_core_rs_autodiff",
    "adam_core_rs_coords",
    "adam_core_rs_kernel_data",
    "adam_core_rs_orbit_determination",
    "adam_core_rs_spice",
}
CRATES_IO_SOURCE = "registry+https://github.com/rust-lang/crates.io-index"
PDM_NON_REGISTRY_SOURCE_KEYS = frozenset(
    {
        "git",
        "hg",
        "svn",
        "bzr",
        "vcs",
        "ref",
        "revision",
        "subdirectory",
        "path",
        "file",
        "editable",
        "url",
    }
)
RUST_ROOT_PACKAGE = "adam_assist"


def verify_python_lock(repo: Path, core_version: str) -> None:
    lock_path = repo / "pdm.lock"
    with lock_path.open("rb") as stream:
        lock = tomllib.load(stream)
    lock_packages = lock.get("package")
    if not isinstance(lock_packages, list):
        raise ValueError("pdm.lock package inventory is missing")  # noqa: TRY004
    non_registry_sources = sorted(
        (
            str(package.get("name", "<unnamed>")),
            sorted(PDM_NON_REGISTRY_SOURCE_KEYS.intersection(package)),
        )
        for package in lock_packages
        if PDM_NON_REGISTRY_SOURCE_KEYS.intersection(package)
    )
    if non_registry_sources:
        raise ValueError(
            "pdm.lock must be registry-only; non-registry sources: "
            f"{non_registry_sources}"
        )
    core_matches = [
        package for package in lock_packages if package.get("name") == "adam-core"
    ]
    if len(core_matches) != 1:
        raise ValueError(
            f"pdm.lock must contain one adam-core, found {len(core_matches)}"
        )
    core = core_matches[0]
    if core.get("version") != core_version:
        raise ValueError(f"pdm.lock adam-core {core.get('version')} != {core_version}")
    files = core.get("files")
    if not isinstance(files, list) or len(files) != 12:
        raise ValueError("pdm.lock adam-core must contain the 12 release wheel hashes")
    prefix = f"adam_core-{core_version}-"
    file_names = [str(item.get("file", "")) for item in files]
    if len(set(file_names)) != 12 or any(
        not file_name.startswith(prefix) for file_name in file_names
    ):
        raise ValueError("pdm.lock adam-core files do not match the release version")
    if any(not str(item.get("hash", "")).startswith("sha256:") for item in files):
        raise ValueError("pdm.lock adam-core registry hashes are missing")


def verify_rust_lock(repo: Path, core_version: str) -> None:
    lock_path = repo / "rust" / "adam_assist_rs" / "Cargo.lock"
    with lock_path.open("rb") as stream:
        lock = tomllib.load(stream)
    lock_packages = lock.get("package")
    if not isinstance(lock_packages, list):
        raise ValueError("Cargo.lock package inventory is missing")  # noqa: TRY004
    root_packages = [
        package for package in lock_packages if package.get("name") == RUST_ROOT_PACKAGE
    ]
    if len(root_packages) != 1 or "source" in root_packages[0]:
        raise ValueError(
            f"Cargo.lock must contain exactly one source-less {RUST_ROOT_PACKAGE} root"
        )
    root_package = root_packages[0]
    non_registry_packages = sorted(
        str(package.get("name", "<unnamed>"))
        for package in lock_packages
        if package is not root_package
        and (package.get("source") != CRATES_IO_SOURCE or not package.get("checksum"))
    )
    if non_registry_packages:
        raise ValueError(
            "Cargo.lock must be registry-only with checksums for every non-root "
            f"package; invalid packages: {non_registry_packages}"
        )
    packages: dict[str, list[dict[str, object]]] = {}
    for package in lock_packages:
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
    non_registry_core_patches = sorted(
        package.get("name", "")
        for package in unused
        if str(package.get("name", "")).startswith("adam_core")
    )
    if non_registry_core_patches:
        raise ValueError(
            "Cargo.lock contains non-registry Core patches: "
            f"{non_registry_core_patches}"
        )
    for forbidden in ("zip", "zopfli"):
        if packages.get(forbidden):
            raise ValueError(f"Cargo.lock contains forbidden dependency {forbidden}")
    rawzip = packages.get("rawzip", [])
    if len(rawzip) != 1:
        raise ValueError(f"Cargo.lock must contain one rawzip, found {len(rawzip)}")
    rawzip_package = rawzip[0]
    if (
        rawzip_package.get("version") != "0.5.2"
        or rawzip_package.get("source") != CRATES_IO_SOURCE
        or not rawzip_package.get("checksum")
    ):
        raise ValueError("Cargo.lock rawzip must be registry-only version 0.5.2")


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
