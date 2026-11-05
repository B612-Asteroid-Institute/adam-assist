from pathlib import Path

from migration.scripts.verify_workflow_policy import (
    DISABLED_JOB_CONDITION,
    NORMAL,
    RELEASE_WORKFLOWS,
    WORKFLOW_INVENTORY,
    load_workflows,
    verify,
)

ROOT = Path(__file__).resolve().parents[1]


def test_exact_workflow_inventory_and_disabled_release_jobs() -> None:
    workflows = load_workflows()
    assert set(workflows) == set(WORKFLOW_INVENTORY)
    assert set(workflows[NORMAL]["jobs"]) == {
        "source-pair",
        "rust-quality",
        "benchmark",
    }
    for name in RELEASE_WORKFLOWS:
        assert workflows[name]["jobs"]
        assert {job["if"] for job in workflows[name]["jobs"].values()} == {
            DISABLED_JOB_CONDITION
        }
    verify()


def test_active_ci_contains_no_release_or_publication_credentials() -> None:
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


def test_release_tools_exist_only_behind_disabled_workflow_jobs() -> None:
    active = (ROOT / ".github" / "workflows" / NORMAL).read_text()
    disabled = "\n".join(
        (ROOT / ".github" / "workflows" / name).read_text()
        for name in RELEASE_WORKFLOWS
    )
    for script in (
        "publish_crate_archive",
        "prepare_pypi_upload",
        "verify_release",
        "verify_registry_locks",
    ):
        assert script not in active
        assert script in disabled


def test_critical_ci_commands_and_benchmark_evidence_are_pinned() -> None:
    workflow = load_workflows()[NORMAL]
    jobs = workflow["jobs"]
    source_runs = "\n".join(
        step.get("run", "") for step in jobs["source-pair"]["steps"]
    )
    rust_run = next(
        step["run"]
        for step in jobs["rust-quality"]["steps"]
        if step.get("name") == "Verify source pair and run Rust gates"
    )
    benchmark_runs = "\n".join(
        step.get("run", "") for step in jobs["benchmark"]["steps"]
    )
    for command in (
        "python -m pytest --benchmark-skip -q tests",
        "mypy --strict src/adam_assist",
        "test ! -e pdm.lock",
        "verify_source_pair_contract.py",
        "--core-tree .source-pair/adam-core",
    ):
        assert command in source_runs
    for command in ("cargo fmt", "cargo check", "cargo clippy", "cargo test"):
        assert command in rust_run
    assert "python -m pytest --benchmark-only" in benchmark_runs
    for name in (
        "Upload current-only benchmark artifact",
        "Upload benchmark artifact",
    ):
        step = next(
            item for item in jobs["benchmark"]["steps"] if item.get("name") == name
        )
        assert step["if"] == "always()"


def test_workflow_inventory_is_case_insensitive(tmp_path: Path) -> None:
    import shutil

    root = tmp_path
    workflow_dir = root / ".github" / "workflows"
    workflow_dir.mkdir(parents=True)
    for name in WORKFLOW_INVENTORY:
        source = ROOT / ".github" / "workflows" / name
        shutil.copy2(source, workflow_dir / name)
    (workflow_dir / "hidden.YML").write_text("name: hidden\n")
    try:
        load_workflows(root)
    except ValueError as error:
        assert "inventory" in str(error)
    else:
        raise AssertionError("uppercase second workflow was accepted")
