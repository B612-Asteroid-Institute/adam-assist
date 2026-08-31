from __future__ import annotations

from pathlib import Path

import pytest

from migration.scripts import publish_crate_archive, verify_release


def _source(repo: Path, python_version: str, core_version: str) -> None:
    (repo / "src" / "adam_assist").mkdir(parents=True)
    (repo / "rust" / "adam_assist_rs").mkdir(parents=True)
    (repo / "src" / "adam_assist" / "version.py").write_text(
        f'__version__ = "{python_version}"\n'
    )
    (repo / "pyproject.toml").write_text(
        "[project]\n" "dependencies = [\n" f'  "adam-core=={core_version}",\n' "]\n"
    )


def _package(version: str, core_version: str) -> dict[str, object]:
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


def test_verify_release_accepts_preview_and_stable_channels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = {"cksum": "abc", "yanked": False, "rust_version": "1.87"}
    monkeypatch.setattr(
        verify_release, "published_crate_entry", lambda *args, **kwargs: entry
    )

    preview = tmp_path / "preview"
    _source(preview, "0.4.0rc7", "0.5.6rc6")
    monkeypatch.setattr(
        verify_release,
        "cargo_package",
        lambda _path: _package("0.4.0-rc.7", "0.1.0-rc.5"),
    )
    verify_release.verify(
        preview,
        python_version="0.4.0rc7",
        rust_version="0.4.0-rc.7",
        core_python_version="0.5.6rc6",
        core_rust_version="0.1.0-rc.5",
        channel="preview",
        rust_checksum="abc",
    )

    stable = tmp_path / "stable"
    _source(stable, "0.4.0", "0.5.7")
    monkeypatch.setattr(
        verify_release,
        "cargo_package",
        lambda _path: _package("0.4.0", "0.5.7"),
    )
    verify_release.verify(
        stable,
        python_version="0.4.0",
        rust_version="0.4.0",
        core_python_version="0.5.7",
        core_rust_version="0.5.7",
        channel="stable",
        rust_checksum="abc",
    )


def test_verify_release_rejects_mixed_stable_and_preview_python_versions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _source(tmp_path, "0.4.0", "0.5.6rc6")
    monkeypatch.setattr(
        verify_release,
        "cargo_package",
        lambda _path: _package("0.4.0", "0.5.7"),
    )

    with pytest.raises(ValueError, match="stable Python"):
        verify_release.verify(
            tmp_path,
            python_version="0.4.0",
            rust_version="0.4.0",
            core_python_version="0.5.6rc6",
            core_rust_version="0.5.7",
            channel="stable",
            rust_checksum="abc",
        )
