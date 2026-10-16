from __future__ import annotations

from pathlib import Path

import pytest

from migration.scripts.verify_registry_locks import (
    CORE_RUST_PACKAGES,
    CRATES_IO_SOURCE,
    verify_python_lock,
    verify_rust_lock,
)


def _write_locks(repo: Path) -> None:
    (repo / "rust" / "adam_assist_rs").mkdir(parents=True, exist_ok=True)
    (repo / "pdm.lock").write_text("""[[package]]
name = "adam-core"
version = "0.6.1"
files = [{file = "adam_core-0.6.1-cp312.whl", hash = "sha256:abc"}]
""")
    packages = "\n".join(f"""[[package]]
name = "{name}"
version = "0.6.1"
source = "{CRATES_IO_SOURCE}"
checksum = "abc"
""" for name in sorted(CORE_RUST_PACKAGES))
    rawzip = f"""[[package]]
name = "rawzip"
version = "0.5.2"
source = "{CRATES_IO_SOURCE}"
checksum = "abc"
"""
    (repo / "rust" / "adam_assist_rs" / "Cargo.lock").write_text(
        packages + "\n" + rawzip
    )


def test_registry_locks_accept_exact_public_core(tmp_path: Path) -> None:
    _write_locks(tmp_path)
    verify_python_lock(tmp_path, "0.6.1")
    verify_rust_lock(tmp_path, "0.6.1")


def test_registry_locks_reject_provisional_sources(tmp_path: Path) -> None:
    _write_locks(tmp_path)
    pdm_lock = tmp_path / "pdm.lock"
    pdm_lock.write_text(pdm_lock.read_text() + 'git = "https://example.invalid/core"\n')
    with pytest.raises(ValueError, match="registry-only"):
        verify_python_lock(tmp_path, "0.6.1")

    cargo_lock = tmp_path / "rust" / "adam_assist_rs" / "Cargo.lock"
    cargo_lock.write_text(
        cargo_lock.read_text().replace(f'source = "{CRATES_IO_SOURCE}"\n', "", 1)
    )
    with pytest.raises(ValueError, match="registry-only"):
        verify_rust_lock(tmp_path, "0.6.1")


def test_registry_locks_reject_missing_hashes_and_duplicate_core(
    tmp_path: Path,
) -> None:
    _write_locks(tmp_path)
    pdm_lock = tmp_path / "pdm.lock"
    pdm_lock.write_text(pdm_lock.read_text().replace("sha256:abc", ""))
    with pytest.raises(ValueError, match="hashes are missing"):
        verify_python_lock(tmp_path, "0.6.1")

    _write_locks(tmp_path)
    cargo_lock = tmp_path / "rust" / "adam_assist_rs" / "Cargo.lock"
    duplicate = f"""[[package]]
name = "adam_core_rs_coords"
version = "0.6.1"
source = "{CRATES_IO_SOURCE}"
checksum = "def"
"""
    cargo_lock.write_text(cargo_lock.read_text() + duplicate)
    with pytest.raises(ValueError, match="must contain one adam_core_rs_coords"):
        verify_rust_lock(tmp_path, "0.6.1")


def test_registry_locks_reject_zip_zopfli_and_wrong_rawzip(tmp_path: Path) -> None:
    for forbidden in ("zip", "zopfli"):
        _write_locks(tmp_path)
        cargo_lock = tmp_path / "rust" / "adam_assist_rs" / "Cargo.lock"
        cargo_lock.write_text(
            cargo_lock.read_text()
            + f'[[package]]\nname = "{forbidden}"\nversion = "2.4.2"\n'
        )
        with pytest.raises(ValueError, match="forbidden dependency"):
            verify_rust_lock(tmp_path, "0.6.1")

    _write_locks(tmp_path)
    cargo_lock = tmp_path / "rust" / "adam_assist_rs" / "Cargo.lock"
    cargo_lock.write_text(
        cargo_lock.read_text().replace('version = "0.5.2"', 'version = "0.6.0"')
    )
    with pytest.raises(
        ValueError, match="rawzip must be registry-only version 0.5.2"
    ):
        verify_rust_lock(tmp_path, "0.6.1")
