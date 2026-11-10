import shutil
from pathlib import Path

import pytest

from migration.scripts.verify_core_pair_lock import PAIR_LOCK, verify


def test_exact_merged_core_pair_lock_is_reproducible() -> None:
    verify()


@pytest.mark.parametrize(
    ("old", "new", "message"),
    (
        (
            'name = "arrow-array"\nversion = "58.4.1"',
            'name = "arrow-array"\nversion = "54.3.1"',
            "arrow-array",
        ),
        (
            'name = "generator"\nversion = "0.8.11"',
            'name = "generator"\nversion = "0.8.12"',
            "generator",
        ),
        (
            'name = "windows-core"\nversion = "0.62.2"',
            'name = "windows-core"\nversion = "0.100.0"',
            "Rust-1.95",
        ),
        (
            'name = "adam_core"\nversion = "0.7.0"',
            'name = "adam_core"\nversion = "0.7.1"',
            "adam_core",
        ),
        (
            'name = "uuid"\nversion = "1.26.1"',
            'name = "uuid"\nversion = "1.29.2"',
            "uuid",
        ),
    ),
)
def test_pair_lock_rejects_boundary_drift(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    candidate = tmp_path / "Cargo.lock"
    shutil.copy2(PAIR_LOCK, candidate)
    source = candidate.read_text()
    assert old in source
    candidate.write_text(source.replace(old, new, 1))
    with pytest.raises(ValueError, match=message):
        verify(candidate, check_digest=False)
