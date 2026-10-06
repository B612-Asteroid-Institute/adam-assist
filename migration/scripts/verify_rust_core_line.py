#!/usr/bin/env python3
"""Assert that a Cargo graph contains one exact, coherent adam-core line."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REQUIRED_CORE_PACKAGES = {
    "adam_core",
    "adam_core_rs_autodiff",
    "adam_core_rs_coords",
    "adam_core_rs_kernel_data",
    "adam_core_rs_orbit_determination",
    "adam_core_rs_spice",
}


def verify_core_line(metadata: dict[str, Any], expected_version: str) -> dict[str, str]:
    core_packages: dict[str, list[dict[str, Any]]] = {}
    for package in metadata["packages"]:
        name = package["name"]
        if name == "adam_core" or name.startswith("adam_core_rs_"):
            core_packages.setdefault(name, []).append(package)
    missing = REQUIRED_CORE_PACKAGES - core_packages.keys()
    if missing:
        raise ValueError(f"Cargo graph is missing Core packages: {sorted(missing)}")
    unexpected = {
        name: [package["version"] for package in packages]
        for name, packages in core_packages.items()
        if len(packages) != 1 or packages[0]["version"] != expected_version
    }
    if unexpected:
        raise ValueError(
            f"Cargo graph mixes Core lines; expected {expected_version}, got {unexpected}"
        )
    return {name: packages[0]["version"] for name, packages in core_packages.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--core-version", required=True)
    args = parser.parse_args()
    selected = verify_core_line(
        json.loads(args.metadata.read_text()), args.core_version
    )
    print(json.dumps(selected, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
