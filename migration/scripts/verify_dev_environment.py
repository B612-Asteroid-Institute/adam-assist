#!/usr/bin/env python3
"""Verify the declared and installed source-pair test environment."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import tomllib
from pathlib import Path

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[2]
FROZEN_DEV_VERSIONS = {
    "naif-leapseconds": "2025.4.22",
    "naif-eop-predict": "2024.8.28.1",
    "naif-eop-historical": "2024.8.28.1",
    "naif-eop-high-prec": "2026.5.9",
    "naif-earth-itrf93": "2007.4.3.1",
}


def verify(*, require_wheel: bool = False) -> None:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        pyproject = tomllib.load(stream)
    declared = {
        requirement.name: requirement
        for value in pyproject["project"]["optional-dependencies"]["dev"]
        for requirement in [Requirement(value)]
    }
    for required in ("astropy", "packaging", "PyYAML"):
        if required not in declared:
            raise ValueError(f"the dev extra must declare {required}")
    for name, expected in FROZEN_DEV_VERSIONS.items():
        requirement = declared.get(name)
        if requirement is None or str(requirement.specifier) != f"=={expected}":
            raise ValueError(f"dev extra must pin {name}=={expected}")

    for module_name in ("astropy", "packaging", "yaml"):
        importlib.import_module(module_name)
    for name, expected in FROZEN_DEV_VERSIONS.items():
        actual = importlib.metadata.version(name)
        if actual != expected:
            raise ValueError(f"installed {name} {actual} != {expected}")
    if importlib.metadata.version("adam-core") != "0.7.0":
        raise ValueError("installed adam-core must be exact source-paired 0.7.0")
    if importlib.metadata.version("adam-assist") != "0.6.0":
        raise ValueError("installed adam-assist must remain 0.6.0")

    if require_wheel:
        for distribution_name, module_name in (
            ("adam-core", "adam_core"),
            ("adam-assist", "adam_assist"),
        ):
            module = importlib.import_module(module_name)
            module_path = Path(module.__file__).resolve()
            installed_root = Path(
                importlib.metadata.distribution(distribution_name).locate_file("")
            ).resolve()
            if not module_path.is_relative_to(installed_root):
                raise ValueError(
                    f"{module_name} did not resolve from its installed wheel: "
                    f"{module_path}"
                )
        source_root = (ROOT / "src" / "adam_assist").resolve()
        if (
            Path(importlib.import_module("adam_assist").__file__)
            .resolve()
            .is_relative_to(source_root)
        ):
            raise ValueError("adam_assist resolved to repository source")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-wheel", action="store_true")
    args = parser.parse_args()
    verify(require_wheel=args.require_wheel)
    print("verified declared and installed adam-assist dev environment")


if __name__ == "__main__":
    main()
