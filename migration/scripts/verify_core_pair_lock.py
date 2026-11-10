#!/usr/bin/env python3
"""Verify the dedicated merged-Core direct-Rust pair lock."""

from __future__ import annotations

import argparse
import hashlib
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAIR_LOCK = ROOT / "migration" / "locks" / "adam-assist-core-8dc01202.Cargo.lock"
EXPECTED_SHA256 = "4e48473a76dc6f28bd80efafcbf8a17cad427b2fc90c2b3cdf7959bd30a900fc"
CORE_PACKAGES = {
    "adam_core",
    "adam_core_rs_autodiff",
    "adam_core_rs_coords",
    "adam_core_rs_kernel_data",
    "adam_core_rs_orbit_determination",
    "adam_core_rs_spice",
}
CRATES_IO_SOURCE = "registry+https://github.com/rust-lang/crates.io-index"


def verify(lock_path: Path = PAIR_LOCK, *, check_digest: bool = True) -> None:
    raw = lock_path.read_bytes()
    if check_digest and hashlib.sha256(raw).hexdigest() != EXPECTED_SHA256:
        raise ValueError("merged-Core direct-Rust pair lock digest drifted")
    lock = tomllib.loads(raw.decode())
    packages = lock.get("package")
    if not isinstance(packages, list):
        raise ValueError("pair lock package inventory is missing")

    by_name: dict[str, list[dict[str, object]]] = {}
    for package in packages:
        by_name.setdefault(str(package["name"]), []).append(package)

    roots = by_name.get("adam_assist", [])
    if len(roots) != 1 or "source" in roots[0] or roots[0].get("version") != "0.6.0":
        raise ValueError("pair lock must contain one source-less adam_assist 0.6.0 root")
    for name in sorted(CORE_PACKAGES):
        matches = by_name.get(name, [])
        if len(matches) != 1 or matches[0].get("version") != "0.7.0":
            raise ValueError(f"pair lock must contain one source Core {name} 0.7.0")
        if "source" in matches[0] or "checksum" in matches[0]:
            raise ValueError(f"pair lock Core source {name} must be path-patched")

    local_names = CORE_PACKAGES | {"adam_assist"}
    invalid_registry = sorted(
        str(package.get("name", "<unnamed>"))
        for package in packages
        if str(package.get("name")) not in local_names
        and (
            package.get("source") != CRATES_IO_SOURCE
            or not package.get("checksum")
        )
    )
    if invalid_registry:
        raise ValueError(
            "pair lock non-Core dependencies must be checksummed crates.io entries: "
            f"{invalid_registry}"
        )

    for name, version in {
        "arrow-array": "58.4.1",
        "generator": "0.8.11",
        "parquet": "58.4.1",
        "numpy": "0.28.0",
        "pyo3": "0.28.3",
        "uuid": "1.26.1",
    }.items():
        matches = by_name.get(name, [])
        if len(matches) != 1 or matches[0].get("version") != version:
            raise ValueError(f"pair lock requires exactly {name} {version}")
    forbidden_high_msrv = sorted(
        (str(package["name"]), str(package["version"]))
        for package in packages
        if str(package["name"]).startswith("windows-")
        and str(package["version"]) == "0.100.0"
    )
    if forbidden_high_msrv:
        raise ValueError(
            f"pair lock contains Rust-1.95 Windows crates: {forbidden_high_msrv}"
        )
    stale_arrow = sorted(
        (str(package["name"]), str(package["version"]))
        for package in packages
        if (
            str(package["name"]) == "parquet"
            or str(package["name"]) == "arrow"
            or str(package["name"]).startswith("arrow-")
        )
        and not str(package["version"]).startswith("58.")
    )
    if stale_arrow:
        raise ValueError(f"pair lock contains a non-58 Arrow boundary: {stale_arrow}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, default=PAIR_LOCK)
    args = parser.parse_args()
    verify(args.lock.resolve())
    print("verified merged Core 8dc01202 direct-Rust pair lock")


if __name__ == "__main__":
    main()
