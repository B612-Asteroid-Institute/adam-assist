from pathlib import Path

from migration.scripts.verify_workflow_policy import NORMAL, load_workflows, verify

ROOT = Path(__file__).resolve().parents[1]


def test_only_exact_source_normal_ci_exists() -> None:
    workflows = load_workflows()
    assert set(workflows) == {NORMAL}
    assert set(workflows[NORMAL]["jobs"]) == {
        "source-pair",
        "rust-quality",
        "benchmark",
    }
    verify()


def test_no_workflow_contains_release_or_publication_credentials() -> None:
    source = (ROOT / ".github" / "workflows" / NORMAL).read_text().lower()
    for forbidden in (
        "id-token",
        "contents: write",
        "pypa/gh-action-pypi-publish",
        "crates-io-auth-action",
        "cargo publish",
        "twine upload",
        "gh release",
    ):
        assert forbidden not in source


def test_post_publication_release_tools_are_not_ci_entry_points() -> None:
    source = (ROOT / ".github" / "workflows" / NORMAL).read_text()
    for script in (
        "publish_crate_archive.py",
        "prepare_pypi_upload.py",
        "verify_release.py",
        "verify_registry_locks.py",
    ):
        assert script not in source
