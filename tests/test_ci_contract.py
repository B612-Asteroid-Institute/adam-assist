from pathlib import Path

from migration.scripts.verify_workflow_policy import (
    EXPECTED_RELEASE_JOB_CONDITIONS,
    NORMAL,
    RELEASE_WORKFLOWS,
    WORKFLOW_INVENTORY,
    load_workflows,
    verify,
)

ROOT = Path(__file__).resolve().parents[1]


def test_exact_workflow_inventory_and_gated_release_jobs() -> None:
    workflows = load_workflows()
    assert set(workflows) == set(WORKFLOW_INVENTORY)
    assert set(workflows[NORMAL]["jobs"]) == {
        "source-pair",
        "rust-quality",
        "benchmark",
    }
    for name in RELEASE_WORKFLOWS:
        assert workflows[name]["jobs"]
        assert {
            job_id: job["if"] for job_id, job in workflows[name]["jobs"].items()
        } == EXPECTED_RELEASE_JOB_CONDITIONS[name]
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


def test_publication_tools_stay_behind_gated_release_jobs() -> None:
    active = (ROOT / ".github" / "workflows" / NORMAL).read_text()
    disabled = "\n".join(
        (ROOT / ".github" / "workflows" / name).read_text()
        for name in RELEASE_WORKFLOWS
    )
    for script in (
        "publish_crate_archive",
        "prepare_pypi_upload",
        "verify_release",
    ):
        assert script not in active
        assert script in disabled


def test_critical_ci_commands_and_benchmark_evidence_are_pinned() -> None:
    workflow = load_workflows()[NORMAL]
    jobs = workflow["jobs"]
    source = jobs["source-pair"]
    source_runs = "\n".join(step.get("run", "") for step in source["steps"])
    assert source["strategy"]["matrix"]["include"] == [
        {
            "python-version": version,
            "core-ref": "a6478f2f280c266d7a5d1442997eb520fc20888f",
            "core-kind": "public-release",
        }
        for version in ("3.11", "3.12", "3.13")
    ] + [
        {
            "python-version": "3.14",
            "core-ref": "8dc01202545c34b152249e7e4674915cd07f57fa",
            "core-kind": "merged-source",
        }
    ]
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
        "pdm lock --check",
        "verify_registry_locks.py",
        "verify_source_pair_contract.py",
        '--core-tree "$core"',
        "ADAM_CORE_MERGED_REF",
        "verify_core_pair_lock.py",
        "adam-assist-core-8dc01202.Cargo.lock",
        "cargo test --locked --all-targets",
        "core_owned_od",
        "-- --ignored",
        "ADAM_CORE_RS_ASSIST_PLANETS_PATH",
        "ADAM_CORE_RS_ASSIST_ASTEROIDS_PATH",
    ):
        assert command in source_runs
    setup_python = next(
        step
        for step in source["steps"]
        if step.get("uses") == "actions/setup-python@v5"
    )
    assert setup_python["with"] == {"python-version": "${{ matrix.python-version }}"}
    assert all("314" not in job_id for job_id in jobs)
    for command in ("cargo fmt", "cargo check", "cargo clippy", "cargo test"):
        assert command in rust_run
    assert "--locked" in rust_run
    assert "patch.crates-io" not in rust_run
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
