#!/usr/bin/env python3
"""Verify the authentic public Core registry locks used by this Assist source."""

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
CORE_080_RUST_CHECKSUMS = {
    "adam_core": "221eca15523aa6b14d4cbbbe73208ccd55d673a26b50851b128ff374ca03a247",
    "adam_core_rs_autodiff": "b2080cf3184b9f04082ae233679607eeb6a21011b6fbff878fde8d742fae4d02",
    "adam_core_rs_coords": "12ba44ef82996715024dea4419ed4312aed5be9e9c8c043d8d238e818c893c90",
    "adam_core_rs_kernel_data": "4957f4580885c92ede9c9a2ddc15fa0657914d9cfdd0ba4b2c5d012196638346",
    "adam_core_rs_orbit_determination": "adc136457a9c93dcc3ae92cf7706935a865d3e19c25fbac96a62e2b666f17f39",
    "adam_core_rs_spice": "2fcbd37df82f538c5cbfc939faec2b08bebc50dd9ff35d9f2f8dd6870dcac354",
}
CORE_080_WHEEL_HASHES = {
    "adam_core-0.8.0-cp311-cp311-macosx_10_12_x86_64.whl": "sha256:676f0551250c6f152bd6b3851d8bd088524f3e9f44aac2ed021e58fc3313fe99",
    "adam_core-0.8.0-cp311-cp311-macosx_11_0_arm64.whl": "sha256:ab30f3d9a29891f0003d28fc6cf71055559526312a53deb9ba1c8cae2ed575d0",
    "adam_core-0.8.0-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl": "sha256:e746c1f43bdd6644988cb2589e87c54c40161ea9783b9a998832dfff930962b8",
    "adam_core-0.8.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl": "sha256:9a21cabb8e403ea711354d93c8634a531a8ad2fe1348546672d33da8e8e70c30",
    "adam_core-0.8.0-cp312-cp312-macosx_10_12_x86_64.whl": "sha256:0e4233b569b0c913ee2a90451f1827ec23fdf33400e05e342ef732619da64c0f",
    "adam_core-0.8.0-cp312-cp312-macosx_11_0_arm64.whl": "sha256:f0dab6a0ca15b6f20f8db1c10fb5b8568015d13994931c6f1302de148885099c",
    "adam_core-0.8.0-cp312-cp312-manylinux_2_17_aarch64.manylinux2014_aarch64.whl": "sha256:ec317f014a367692b14ea1738144e43e9ffb90e5533a56566a229b9e99481042",
    "adam_core-0.8.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl": "sha256:31dc803b441775388babe3737e03a5d272e82c322dea1c130b7019d70a0533bf",
    "adam_core-0.8.0-cp313-cp313-macosx_10_12_x86_64.whl": "sha256:dd56d4f3ee1e8a43a0a5c26a57d0a291f25b93bacd66c4d531cabf4c53e4e104",
    "adam_core-0.8.0-cp313-cp313-macosx_11_0_arm64.whl": "sha256:5a9899666dc2aa263bb11586b2a83cf06f945cd616ab7ed2bbc825a1a49f0b53",
    "adam_core-0.8.0-cp313-cp313-manylinux_2_17_aarch64.manylinux2014_aarch64.whl": "sha256:259241ff65635076549cdeef486adb41657286b45e3a813ae86caf2eb289d4ba",
    "adam_core-0.8.0-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl": "sha256:929258794f2ac8b74cef645da69459cf247bd868c7258098be28480ed5c3c9e2",
    "adam_core-0.8.0-cp314-cp314-macosx_10_12_x86_64.whl": "sha256:e7b55bcc9accd35be0b505ecb16cc923ff200e4776efc26d4ff4c8eec9e24ea3",
    "adam_core-0.8.0-cp314-cp314-macosx_11_0_arm64.whl": "sha256:77be4b984d82a35f27a69a56685010c6cb386bf28f0571baacce62aef6a1ba36",
    "adam_core-0.8.0-cp314-cp314-manylinux_2_17_aarch64.manylinux2014_aarch64.whl": "sha256:506dccf917ec491a5b38f0f655fe3bf97b84166ee35759294d84158834772f83",
    "adam_core-0.8.0-cp314-cp314-manylinux_2_17_x86_64.manylinux2014_x86_64.whl": "sha256:a3e9f1888b31b35133bf8d5e56e391fffadfaecfeaa35e575d25fe07555d1b59",
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
    if not isinstance(files, list) or len(files) != 16:
        raise ValueError("pdm.lock adam-core must contain the 16 release wheel hashes")
    prefix = f"adam_core-{core_version}-"
    file_names = [str(item.get("file", "")) for item in files]
    if len(set(file_names)) != 16 or any(
        not file_name.startswith(prefix) for file_name in file_names
    ):
        raise ValueError("pdm.lock adam-core files do not match the release version")
    locked_files = {
        str(item.get("file", "")): str(item.get("hash", "")) for item in files
    }
    if core_version == "0.8.0" and locked_files != CORE_080_WHEEL_HASHES:
        raise ValueError("pdm.lock adam-core wheel hashes do not match public 0.8.0")
    if any(not value.startswith("sha256:") for value in locked_files.values()):
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
        if (
            core_version == "0.8.0"
            and package.get("checksum") != CORE_080_RUST_CHECKSUMS[name]
        ):
            raise ValueError(
                f"Cargo.lock {name} checksum does not match public Core 0.8.0"
            )
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
