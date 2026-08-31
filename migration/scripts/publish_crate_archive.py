#!/usr/bin/env python3
"""Validate or publish the exact prebuilt ``adam_assist`` crate archive.

Candidate CI packages and tests the archive once. The publication workflow
then downloads those exact bytes and sends them through Cargo's registry
protocol without repackaging.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
import tarfile
import time
import tomllib
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

CRATE_NAME = "adam_assist"
CORE_DEPENDENCIES = (
    "adam_core_rs_coords",
    "adam_core_rs_kernel_data",
    "adam_core_rs_spice",
)


def core_requirements(version: str) -> dict[str, str]:
    return {name: f"={version}" for name in CORE_DEPENDENCIES}


def cargo_package(manifest_path: Path) -> dict[str, Any]:
    output = subprocess.check_output(
        [
            "cargo",
            "metadata",
            "--format-version",
            "1",
            "--locked",
            "--no-deps",
            "--manifest-path",
            str(manifest_path),
        ],
        text=True,
    )
    packages = json.loads(output)["packages"]
    matches = [package for package in packages if package["name"] == CRATE_NAME]
    if len(matches) != 1:
        raise ValueError(f"expected one {CRATE_NAME} package, found {len(matches)}")
    return matches[0]


def publish_metadata(package: dict[str, Any]) -> dict[str, Any]:
    manifest_dir = Path(package["manifest_path"]).parent
    readme_path = package.get("readme")
    readme = None
    readme_file = None
    if readme_path:
        path = Path(readme_path)
        readme = path.read_text()
        readme_file = os.path.relpath(path, manifest_dir)

    dependencies = [
        {
            "name": dependency["name"],
            "version_req": dependency["req"],
            "features": dependency["features"],
            "optional": dependency["optional"],
            "default_features": dependency["uses_default_features"],
            "target": dependency["target"],
            "kind": dependency["kind"] or "normal",
            "registry": dependency["registry"],
            "explicit_name_in_toml": dependency["rename"],
        }
        for dependency in package["dependencies"]
    ]
    dependencies.sort(
        key=lambda item: (
            item["kind"],
            item["target"] or "",
            item["explicit_name_in_toml"] or item["name"],
        )
    )

    return {
        "name": package["name"],
        "vers": package["version"],
        "deps": dependencies,
        "features": package["features"],
        "authors": package["authors"],
        "description": package["description"],
        "documentation": package["documentation"],
        "homepage": package["homepage"],
        "readme": readme,
        "readme_file": readme_file,
        "keywords": package["keywords"],
        "categories": package["categories"],
        "license": package["license"],
        "license_file": package["license_file"],
        "repository": package["repository"],
        "badges": {},
        "links": package["links"],
        "rust_version": package["rust_version"],
    }


def publish_body(metadata: dict[str, Any], archive: bytes) -> bytes:
    encoded = json.dumps(metadata, separators=(",", ":")).encode()
    return (
        struct.pack("<I", len(encoded))
        + encoded
        + struct.pack("<I", len(archive))
        + archive
    )


def archive_identity(path: Path) -> tuple[str, str]:
    with tarfile.open(path, mode="r:gz") as archive:
        manifests = [
            member
            for member in archive.getmembers()
            if member.name.endswith("/Cargo.toml")
        ]
        if len(manifests) != 1:
            raise ValueError(f"{path.name} has {len(manifests)} root manifests")
        source = archive.extractfile(manifests[0])
        if source is None:
            raise ValueError(f"could not read Cargo.toml from {path.name}")
        manifest = tomllib.loads(source.read().decode())
    package = manifest.get("package", {})
    name = package.get("name")
    version = package.get("version")
    if not isinstance(name, str) or not isinstance(version, str):
        raise TypeError(f"{path.name} archive has no package identity")
    return name, version


def validate_package(
    package: dict[str, Any],
    expected_version: str,
    expected_core_version: str,
    channel: str,
) -> None:
    if package["version"] != expected_version:
        raise ValueError(
            f"{CRATE_NAME} version {package['version']} != {expected_version}"
        )
    if channel not in {"preview", "stable"}:
        raise ValueError(f"unsupported release channel: {channel}")
    is_prerelease = "-" in expected_version
    core_is_prerelease = "-" in expected_core_version
    if channel == "preview" and (not is_prerelease or not core_is_prerelease):
        raise ValueError("preview package and Core versions must be prereleases")
    if channel == "stable" and (is_prerelease or core_is_prerelease):
        raise ValueError("stable package and Core versions must not be prereleases")
    if package["rust_version"] != "1.87":
        raise ValueError(f"unexpected MSRV {package['rust_version']}")
    if package.get("publish") == []:
        raise ValueError(f"{CRATE_NAME} is marked publish=false")

    requirements = {
        dependency["name"]: dependency["req"] for dependency in package["dependencies"]
    }
    for name, expected in core_requirements(expected_core_version).items():
        actual = requirements.get(name)
        if actual != expected:
            raise ValueError(f"{CRATE_NAME}->{name} must use {expected}, got {actual}")


def request_json(request: urllib.request.Request, timeout: float = 60.0) -> Any:
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise RuntimeError(
            f"registry request failed ({error.code}): {detail}"
        ) from error


def crates_io_index_path(name: str) -> str:
    normalized = name.lower()
    if len(normalized) == 1:
        return f"1/{normalized}"
    if len(normalized) == 2:
        return f"2/{normalized}"
    if len(normalized) == 3:
        return f"3/{normalized[0]}/{normalized}"
    return f"{normalized[:2]}/{normalized[2:4]}/{normalized}"


def published_crate_entry(index: str, name: str, version: str) -> dict[str, Any] | None:
    request = urllib.request.Request(
        f"{index.rstrip('/')}/{crates_io_index_path(name)}",
        headers={
            "Accept": "text/plain",
            "User-Agent": "adam-assist-release-automation/1",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            entries = [json.loads(line) for line in response if line.strip()]
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        detail = error.read().decode(errors="replace")
        raise RuntimeError(
            f"registry index request failed ({error.code}): {detail}"
        ) from error
    matches = [entry for entry in entries if entry.get("vers") == version]
    if len(matches) > 1:
        raise ValueError(f"multiple registry entries for {name} {version}")
    return matches[0] if matches else None


def validate_existing_archive(
    entry: dict[str, Any], name: str, version: str, digest: str
) -> None:
    if entry.get("cksum") != digest:
        raise ValueError(
            f"published {name} {version} checksum {entry.get('cksum')} != {digest}"
        )
    if entry.get("yanked", False):
        raise ValueError(f"published {name} {version} is yanked")


def wait_for_version(api: str, name: str, version: str) -> None:
    url = f"{api.rstrip('/')}/api/v1/crates/{name}/{version}"
    for attempt in range(60):
        try:
            request_json(
                urllib.request.Request(url, headers={"Accept": "application/json"})
            )
            return
        except RuntimeError as error:
            if "(404)" not in str(error) or attempt == 59:
                raise
        time.sleep(2.0)
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("rust/adam_assist_rs/Cargo.toml"),
    )
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--expected-version", required=True)
    parser.add_argument("--expected-core-version", required=True)
    parser.add_argument("--channel", choices=("preview", "stable"), default="preview")
    parser.add_argument("--registry-api", default="https://crates.io")
    parser.add_argument("--registry-index", default="https://index.crates.io")
    parser.add_argument("--token-env", default="CARGO_REGISTRY_TOKEN")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    manifest_path = args.manifest_path.resolve()
    package = cargo_package(manifest_path)
    validate_package(
        package,
        args.expected_version,
        args.expected_core_version,
        args.channel,
    )
    archive_path = args.archives / f"{CRATE_NAME}-{args.expected_version}.crate"
    archive_name, archive_version = archive_identity(archive_path)
    if (archive_name, archive_version) != (CRATE_NAME, args.expected_version):
        raise ValueError(
            f"archive identity mismatch for {archive_path.name}: "
            f"{archive_name} {archive_version}"
        )

    archive = archive_path.read_bytes()
    digest = hashlib.sha256(archive).hexdigest()
    body = publish_body(publish_metadata(package), archive)
    print(f"prepared {archive_path.name}: archive={len(archive)} body={len(body)}")
    if not args.execute:
        print("dry run only; pass --execute to publish")
        return

    entry = published_crate_entry(
        args.registry_index, CRATE_NAME, args.expected_version
    )
    if entry is not None:
        validate_existing_archive(entry, CRATE_NAME, args.expected_version, digest)
        print(f"already published exact archive {archive_path.name}; skipping")
        return

    token = os.environ.get(args.token_env)
    if not token:
        raise SystemExit(f"{args.token_env} is required with --execute")
    request = urllib.request.Request(
        f"{args.registry_api.rstrip('/')}/api/v1/crates/new",
        data=body,
        method="PUT",
        headers={
            "Accept": "application/json",
            "Authorization": token,
            "Content-Type": "application/octet-stream",
            "User-Agent": "adam-assist-release-automation/1",
        },
    )
    request_json(request)
    print(f"published exact archive {archive_path.name}")
    wait_for_version(args.registry_api, CRATE_NAME, args.expected_version)


if __name__ == "__main__":
    main()
