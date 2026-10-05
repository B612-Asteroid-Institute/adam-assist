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
    (repo / "rust" / "adam_assist_rs").mkdir(parents=True)
    (repo / "pdm.lock").write_text("""[[package]]
name = "adam-core"
version = "0.5.8"
files = [{file = "adam_core-0.5.8-cp312.whl", hash = "sha256:abc"}]
""")
    packages = "\n".join(f"""[[package]]
name = "{name}"
version = "0.5.8"
source = "{CRATES_IO_SOURCE}"
checksum = "abc"
""" for name in sorted(CORE_RUST_PACKAGES))
    (repo / "rust" / "adam_assist_rs" / "Cargo.lock").write_text(packages)


def test_registry_locks_accept_exact_public_core(tmp_path: Path) -> None:
    _write_locks(tmp_path)
    verify_python_lock(tmp_path, "0.5.8")
    verify_rust_lock(tmp_path, "0.5.8")


def test_registry_locks_reject_prepublication_sources(tmp_path: Path) -> None:
    _write_locks(tmp_path)
    pdm_lock = tmp_path / "pdm.lock"
    pdm_lock.write_text(pdm_lock.read_text() + 'git = "https://example.invalid/core"\n')
    with pytest.raises(ValueError, match="registry-only"):
        verify_python_lock(tmp_path, "0.5.8")

    cargo_lock = tmp_path / "rust" / "adam_assist_rs" / "Cargo.lock"
    cargo_lock.write_text(
        cargo_lock.read_text().replace(f'source = "{CRATES_IO_SOURCE}"\n', "", 1)
    )
    with pytest.raises(ValueError, match="registry-only"):
        verify_rust_lock(tmp_path, "0.5.8")
