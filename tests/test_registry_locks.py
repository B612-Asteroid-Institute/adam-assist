from __future__ import annotations

from pathlib import Path

import pytest

from migration.scripts.verify_registry_locks import (
    CORE_RUST_PACKAGES,
    CRATES_IO_SOURCE,
    PDM_NON_REGISTRY_SOURCE_KEYS,
    verify_python_lock,
    verify_rust_lock,
)

PDM_SOURCE_MUTATIONS = {
    "git": '"https://example.invalid/repository"',
    "hg": '"https://example.invalid/repository"',
    "svn": '"https://example.invalid/repository"',
    "bzr": '"https://example.invalid/repository"',
    "vcs": '"git"',
    "ref": '"main"',
    "revision": '"0123456789abcdef"',
    "subdirectory": '"python"',
    "path": '"../local-package"',
    "file": '"../local-package.whl"',
    "editable": "true",
    "url": '"https://example.invalid/package.whl"',
}


def _write_locks(repo: Path) -> None:
    (repo / "rust" / "adam_assist_rs").mkdir(parents=True, exist_ok=True)
    (repo / "pdm.lock").write_text("""[[package]]
name = "adam-core"
version = "0.8.1"
files = [
  {file = "adam_core-0.8.1-cp311-cp311-manylinux_2_17_x86_64.whl", hash = "sha256:001"},
  {file = "adam_core-0.8.1-cp312-cp312-manylinux_2_17_x86_64.whl", hash = "sha256:002"},
  {file = "adam_core-0.8.1-cp313-cp313-manylinux_2_17_x86_64.whl", hash = "sha256:003"},
  {file = "adam_core-0.8.1-cp314-cp314-manylinux_2_17_x86_64.whl", hash = "sha256:013"},
  {file = "adam_core-0.8.1-cp311-cp311-manylinux_2_17_aarch64.whl", hash = "sha256:004"},
  {file = "adam_core-0.8.1-cp312-cp312-manylinux_2_17_aarch64.whl", hash = "sha256:005"},
  {file = "adam_core-0.8.1-cp313-cp313-manylinux_2_17_aarch64.whl", hash = "sha256:006"},
  {file = "adam_core-0.8.1-cp314-cp314-manylinux_2_17_aarch64.whl", hash = "sha256:014"},
  {file = "adam_core-0.8.1-cp311-cp311-macosx_11_0_arm64.whl", hash = "sha256:007"},
  {file = "adam_core-0.8.1-cp312-cp312-macosx_11_0_arm64.whl", hash = "sha256:008"},
  {file = "adam_core-0.8.1-cp313-cp313-macosx_11_0_arm64.whl", hash = "sha256:009"},
  {file = "adam_core-0.8.1-cp314-cp314-macosx_11_0_arm64.whl", hash = "sha256:015"},
  {file = "adam_core-0.8.1-cp311-cp311-macosx_10_12_x86_64.whl", hash = "sha256:010"},
  {file = "adam_core-0.8.1-cp312-cp312-macosx_10_12_x86_64.whl", hash = "sha256:011"},
  {file = "adam_core-0.8.1-cp313-cp313-macosx_10_12_x86_64.whl", hash = "sha256:012"},
  {file = "adam_core-0.8.1-cp314-cp314-macosx_10_12_x86_64.whl", hash = "sha256:016"},
]
""")
    packages = "\n".join(f"""[[package]]
name = "{name}"
version = "0.8.1"
source = "{CRATES_IO_SOURCE}"
checksum = "abc"
""" for name in sorted(CORE_RUST_PACKAGES))
    root_and_non_core = f"""[[package]]
name = "adam_assist"
version = "0.6.0"

[[package]]
name = "rawzip"
version = "0.5.2"
source = "{CRATES_IO_SOURCE}"
checksum = "abc"

[[package]]
name = "serde"
version = "1.0.0"
source = "{CRATES_IO_SOURCE}"
checksum = "abc"
"""
    (repo / "rust" / "adam_assist_rs" / "Cargo.lock").write_text(
        packages + "\n" + root_and_non_core
    )


def test_registry_locks_accept_exact_public_core(tmp_path: Path) -> None:
    _write_locks(tmp_path)
    verify_python_lock(tmp_path, "0.8.1")
    verify_rust_lock(tmp_path, "0.8.1")


def test_current_registry_locks_match_exact_public_core() -> None:
    repo = Path(__file__).resolve().parents[1]
    verify_python_lock(repo, "0.8.0")
    verify_rust_lock(repo, "0.8.0")


def test_exact_public_core_hashes_reject_drift(tmp_path: Path) -> None:
    import shutil

    repo = Path(__file__).resolve().parents[1]
    shutil.copy2(repo / "pdm.lock", tmp_path / "pdm.lock")
    rust = tmp_path / "rust" / "adam_assist_rs"
    rust.mkdir(parents=True)
    shutil.copy2(repo / "rust" / "adam_assist_rs" / "Cargo.lock", rust / "Cargo.lock")

    python_lock = tmp_path / "pdm.lock"
    python_lock.write_text(
        python_lock.read_text().replace(
            "sha256:676f0551250c6f152bd6b3851d8bd088524f3e9f44aac2ed021e58fc3313fe99",
            "sha256:" + "0" * 64,
            1,
        )
    )
    with pytest.raises(ValueError, match="wheel hashes"):
        verify_python_lock(tmp_path, "0.8.0")

    rust_lock = rust / "Cargo.lock"
    rust_lock.write_text(
        rust_lock.read_text().replace(
            "221eca15523aa6b14d4cbbbe73208ccd55d673a26b50851b128ff374ca03a247",
            "0" * 64,
            1,
        )
    )
    with pytest.raises(ValueError, match="checksum"):
        verify_rust_lock(tmp_path, "0.8.0")


@pytest.mark.parametrize(("source_key", "source_value"), PDM_SOURCE_MUTATIONS.items())
def test_python_lock_rejects_non_core_source_forms(
    tmp_path: Path, source_key: str, source_value: str
) -> None:
    assert set(PDM_SOURCE_MUTATIONS) == PDM_NON_REGISTRY_SOURCE_KEYS
    _write_locks(tmp_path)
    pdm_lock = tmp_path / "pdm.lock"
    pdm_lock.write_text(pdm_lock.read_text() + f"""\n[[package]]
name = "hidden-source"
version = "1.0.0"
{source_key} = {source_value}
""")
    with pytest.raises(
        ValueError, match=rf"non-registry sources.*hidden-source.*{source_key}"
    ):
        verify_python_lock(tmp_path, "0.8.1")


def test_rust_lock_rejects_non_registry_core_source(tmp_path: Path) -> None:
    _write_locks(tmp_path)
    cargo_lock = tmp_path / "rust" / "adam_assist_rs" / "Cargo.lock"
    cargo_lock.write_text(
        cargo_lock.read_text().replace(
            f'source = "{CRATES_IO_SOURCE}"',
            'source = "git+https://example.invalid/core"',
            1,
        )
    )
    with pytest.raises(ValueError, match="registry-only"):
        verify_rust_lock(tmp_path, "0.8.1")


@pytest.mark.parametrize(
    ("package_name", "source_lines"),
    [
        (
            "git-dependency",
            'source = "git+https://example.invalid/dependency"\nchecksum = "abc"',
        ),
        (
            "path-dependency",
            'source = "path+file:///example/dependency"\nchecksum = "abc"',
        ),
        ("missing-source", ""),
        ("missing-checksum", f'source = "{CRATES_IO_SOURCE}"'),
    ],
)
def test_rust_lock_rejects_non_registry_non_core_packages(
    tmp_path: Path, package_name: str, source_lines: str
) -> None:
    _write_locks(tmp_path)
    cargo_lock = tmp_path / "rust" / "adam_assist_rs" / "Cargo.lock"
    cargo_lock.write_text(cargo_lock.read_text() + f"""\n[[package]]
name = "{package_name}"
version = "1.0.0"
{source_lines}
""")
    with pytest.raises(ValueError, match=rf"registry-only.*{package_name}"):
        verify_rust_lock(tmp_path, "0.8.1")


def test_registry_locks_reject_missing_hashes_and_duplicate_core(
    tmp_path: Path,
) -> None:
    _write_locks(tmp_path)
    pdm_lock = tmp_path / "pdm.lock"
    pdm_lock.write_text(pdm_lock.read_text().replace("sha256:001", ""))
    with pytest.raises(ValueError, match="hashes are missing"):
        verify_python_lock(tmp_path, "0.8.1")

    _write_locks(tmp_path)
    cargo_lock = tmp_path / "rust" / "adam_assist_rs" / "Cargo.lock"
    duplicate = f"""[[package]]
name = "adam_core_rs_coords"
version = "0.8.1"
source = "{CRATES_IO_SOURCE}"
checksum = "def"
"""
    cargo_lock.write_text(cargo_lock.read_text() + duplicate)
    with pytest.raises(ValueError, match="must contain one adam_core_rs_coords"):
        verify_rust_lock(tmp_path, "0.8.1")


def test_registry_locks_reject_zip_zopfli_and_wrong_rawzip(tmp_path: Path) -> None:
    for forbidden in ("zip", "zopfli"):
        _write_locks(tmp_path)
        cargo_lock = tmp_path / "rust" / "adam_assist_rs" / "Cargo.lock"
        cargo_lock.write_text(cargo_lock.read_text() + f"""[[package]]
name = "{forbidden}"
version = "2.4.2"
source = "{CRATES_IO_SOURCE}"
checksum = "abc"
""")
        with pytest.raises(ValueError, match="forbidden dependency"):
            verify_rust_lock(tmp_path, "0.8.1")

    _write_locks(tmp_path)
    cargo_lock = tmp_path / "rust" / "adam_assist_rs" / "Cargo.lock"
    cargo_lock.write_text(
        cargo_lock.read_text().replace('version = "0.5.2"', 'version = "0.6.0"')
    )
    with pytest.raises(ValueError, match="rawzip must be registry-only version 0.5.2"):
        verify_rust_lock(tmp_path, "0.8.1")
