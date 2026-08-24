#!/usr/bin/env python3
"""Validate or publish the exact prebuilt ``adam_assist`` crate archive.

Candidate CI packages and tests the archive once. The publication workflow
then downloads those exact bytes and sends them through Cargo's registry
protocol without repackaging.
"""

from __future__ import annotations

import argparse
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
CORE_REQUIREMENTS = {
    "adam_core_rs_coords": "=0.1.0-rc.4",
    "adam_core_rs_kernel_data": "=0.1.0-rc.4",
    "adam_core_rs_spice": "=0.1.0-rc.4",
}
MSRV_REQUIREMENTS = {
    "icu_locale_core": "=2.2.0",
    "icu_normalizer": "=2.2.0",
    "icu_properties": "=2.2.0",
    "icu_provider": "=2.2.0",
}


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


def validate_package(package: dict[str, Any], expected_version: str) -> None:
    if package["version"] != expected_version:
        raise ValueError(
            f"{CRATE_NAME} version {package['version']} != {expected_version}"
        )
    if "-" not in expected_version:
        raise ValueError(f"only prerelease versions may be published: {expected_version}")
    if package["rust_version"] != "1.87":
        raise ValueError(f"unexpected MSRV {package['rust_version']}")
    if package.get("publish") == []:
        raise ValueError(f"{CRATE_NAME} is marked publish=false")

    requirements = {
        dependency["name"]: dependency["req"]
        for dependency in package["dependencies"]
    }
    for name, expected in (CORE_REQUIREMENTS | MSRV_REQUIREMENTS).items():
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
    parser.add_argument("--registry-api", default="https://crates.io")
    parser.add_argument("--token-env", default="CARGO_REGISTRY_TOKEN")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    manifest_path = args.manifest_path.resolve()
    package = cargo_package(manifest_path)
    validate_package(package, args.expected_version)
    archive_path = args.archives / f"{CRATE_NAME}-{args.expected_version}.crate"
    archive_name, archive_version = archive_identity(archive_path)
    if (archive_name, archive_version) != (CRATE_NAME, args.expected_version):
        raise ValueError(
            f"archive identity mismatch for {archive_path.name}: "
            f"{archive_name} {archive_version}"
        )

    archive = archive_path.read_bytes()
    body = publish_body(publish_metadata(package), archive)
    print(f"prepared {archive_path.name}: archive={len(archive)} body={len(body)}")
    if not args.execute:
        print("dry run only; pass --execute to publish")
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
