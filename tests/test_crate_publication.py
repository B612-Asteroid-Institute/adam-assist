import io
import json
import struct
import tarfile
from pathlib import Path

import pytest

from migration.scripts import publish_crate_archive

VERSION = "0.4.0-rc.7"
CORE_VERSION = "0.1.0-rc.5"


def _package(
    version: str = VERSION, core_version: str = CORE_VERSION
) -> dict[str, object]:
    return {
        "name": "adam_assist",
        "version": version,
        "rust_version": "1.87",
        "publish": None,
        "dependencies": [
            {"name": name, "req": requirement}
            for name, requirement in publish_crate_archive.core_requirements(
                core_version
            ).items()
        ],
    }


def _write_crate(path: Path, name: str = "adam_assist", version: str = VERSION) -> None:
    manifest = f'[package]\nname = "{name}"\nversion = "{version}"\n'.encode()
    info = tarfile.TarInfo(f"{name}-{version}/Cargo.toml")
    info.size = len(manifest)
    with tarfile.open(path, mode="w:gz") as archive:
        archive.addfile(info, io.BytesIO(manifest))


def test_archive_identity_reads_the_normalized_manifest(tmp_path: Path) -> None:
    crate = tmp_path / f"adam_assist-{VERSION}.crate"
    _write_crate(crate)
    assert publish_crate_archive.archive_identity(crate) == ("adam_assist", VERSION)


def test_publish_body_preserves_metadata_and_exact_archive_bytes() -> None:
    metadata = {"name": "adam_assist", "vers": VERSION}
    archive = b"exact tested crate bytes"
    body = publish_crate_archive.publish_body(metadata, archive)
    metadata_size = struct.unpack("<I", body[:4])[0]
    metadata_end = 4 + metadata_size
    assert json.loads(body[4:metadata_end]) == metadata
    archive_size = struct.unpack("<I", body[metadata_end : metadata_end + 4])[0]
    assert archive_size == len(archive)
    assert body[metadata_end + 4 :] == archive


def test_package_validation_requires_msrv_channel_and_core_pins() -> None:
    publish_crate_archive.validate_package(_package(), VERSION, CORE_VERSION, "preview")
    publish_crate_archive.validate_package(
        _package("0.4.0", "0.5.7"), "0.4.0", "0.5.7", "stable"
    )

    with pytest.raises(ValueError, match="stable package"):
        publish_crate_archive.validate_package(
            _package(), VERSION, CORE_VERSION, "stable"
        )

    unpinned = _package()
    dependencies = unpinned["dependencies"]
    assert isinstance(dependencies, list)
    dependencies[0]["req"] = "0.1"
    with pytest.raises(ValueError, match="must use"):
        publish_crate_archive.validate_package(
            unpinned, VERSION, CORE_VERSION, "preview"
        )

    unpublished = _package()
    unpublished["publish"] = []
    with pytest.raises(ValueError, match="publish=false"):
        publish_crate_archive.validate_package(
            unpublished, VERSION, CORE_VERSION, "preview"
        )


def test_existing_archive_must_be_exact_and_unyanked() -> None:
    entry = {"cksum": "abc", "yanked": False}
    publish_crate_archive.validate_existing_archive(
        entry, "adam_assist", VERSION, "abc"
    )
    with pytest.raises(ValueError, match="checksum"):
        publish_crate_archive.validate_existing_archive(
            entry, "adam_assist", VERSION, "def"
        )
    with pytest.raises(ValueError, match="yanked"):
        publish_crate_archive.validate_existing_archive(
            {"cksum": "abc", "yanked": True},
            "adam_assist",
            VERSION,
            "abc",
        )
