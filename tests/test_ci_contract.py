import shlex
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
NORMAL = WORKFLOWS / "pip-build-lint-test-coverage.yml"
WHEEL_CANDIDATE = WORKFLOWS / "release-candidate-wheel-matrix.yml"
RUST_CANDIDATE = WORKFLOWS / "rust-crate-release-candidate.yml"
PYTHON_PUBLISH = WORKFLOWS / "publish.yml"
RUST_PUBLISH = WORKFLOWS / "publish-rust-crate.yml"
CORE_REF = "f0f1aec8b3c059f0cc4af054b5ebe8ed7760da19"
CORE_VERSION = "0.7.0"


def test_normal_ci_builds_and_tests_the_exact_source_pair() -> None:
    workflow = NORMAL.read_text()
    assert "compatibility-floor:" not in workflow
    assert "exit 1" not in workflow
    assert f'ADAM_CORE_SOURCE_REF: "{CORE_REF}"' in workflow
    assert 'ADAM_CORE_VERSION: "0.7.0"' in workflow
    assert 'python-version: ["3.11", "3.12", "3.13"]' in workflow
    assert "repository: B612-Asteroid-Institute/adam_core" in workflow
    assert "ref: ${{ env.ADAM_CORE_SOURCE_REF }}" in workflow
    assert "verify_source_pair_contract.py" in workflow
    assert workflow.count("maturin build --release") >= 2
    source_install = next(
        step["run"]
        for step in _workflow(NORMAL)["jobs"]["source-pair"]["steps"]
        if step.get("name") == "Build Core and ASSIST wheels from the exact pair"
    )
    benchmark_install = next(
        step["run"]
        for step in _workflow(NORMAL)["jobs"]["benchmark"]["steps"]
        if step.get("name") == "Install the already-tested source-pair wheels"
    )
    for command in (source_install, benchmark_install):
        commands = _shell_commands([{"run": command}])
        assert [
            "python",
            "-m",
            "pip",
            "install",
            "$core_wheel",
            "${assist_wheel}[dev]",
        ] in commands
        assert _has_python_script(
            commands,
            "migration/scripts/verify_dev_environment.py",
            {"--require-wheel"},
        )
    assert "[patch.crates-io]" in workflow
    for crate in (
        "adam_core_rs_coords",
        "adam_core_rs_spice",
        "adam_core_rs_kernel_data",
    ):
        assert f'{crate} = {{ path = "$core/{crate}" }}' in workflow
        assert f"patch.crates-io.{crate}.path" in workflow
    assert 'python -m pip install "$core_wheel" "${assist_wheel}[dev]"' in workflow
    assert 'metadata.version("adam-core") == "0.7.0"' in workflow
    assert 'str(core.specifier) == "<0.8.0,>=0.7.0"' in workflow
    assert "python -m pytest --benchmark-skip" in workflow
    assert "mypy --strict src/adam_assist" in workflow
    assert "cargo clippy" in workflow and "-- -D warnings" in workflow
    assert "cargo test" in workflow and "cargo check" in workflow
    assert "test ! -e pdm.lock" in workflow
    assert "test ! -e rust/adam_assist_rs/Cargo.lock" in workflow
    assert "source-pair-wheels-3.12" in workflow
    assert "benchmark_current" in workflow


def test_rust_workflows_use_the_msrv_toolchain_and_components() -> None:
    for workflow in WORKFLOWS.glob("*.yml"):
        source = workflow.read_text()
        if "cargo " not in source and "maturin" not in source:
            continue
        if workflow.name == "rust-crate-release-candidate.yml":
            assert "dtolnay/rust-toolchain@stable" in source
        else:
            assert "dtolnay/rust-toolchain@stable" not in source, workflow.name
        assert "dtolnay/rust-toolchain@1.87.0" in source, workflow.name
        assert "components: rustfmt, clippy" in source, workflow.name


def test_every_release_and_publish_workflow_is_fail_closed_on_registry_state() -> None:
    for path in (WHEEL_CANDIDATE, RUST_CANDIDATE, PYTHON_PUBLISH, RUST_PUBLISH):
        source = path.read_text()
        assert "compatibility-floor:" in source, path.name
        assert "needs: compatibility-floor" in source, path.name
        assert "verify_source_pair_contract.py" in source, path.name
        assert "Core 0.7.0 is not yet public" in source, path.name
        assert "registry locks/checksums are unavailable" in source, path.name
        assert "exit 1" in source, path.name
        assert "UNASSIGNED_PROPAGATOR_V2_RELEASE" not in source, path.name


def test_rust_candidate_retains_build_once_release_contract_at_070() -> None:
    source = RUST_CANDIDATE.read_text()
    assert 'RUST_RELEASE_VERSION: "0.6.0"' in source
    assert 'CORE_RUST_VERSION: "0.7.0"' in source
    assert 'core-version: ["0.7.0"]' in source
    assert "cargo package --manifest-path" in source
    assert 'RUSTDOCFLAGS="-D warnings -D missing-docs"' in source
    assert "publish_crate_archive.py" in source
    assert 'adam_core = "=$CORE_RUST_VERSION"' in source
    assert "verify_rust_core_line.py" in source
    assert "Download the one tested adam-assist crate" in source
    assert "cargo publish" not in source


def test_publish_workflows_retain_irreversible_action_guards_at_070() -> None:
    rust = RUST_PUBLISH.read_text()
    assert "candidate_run_id:" in rust
    assert "expected_core_version:" in rust
    assert "default: 0.7.0" in rust
    assert "trusted-publishing" in rust
    assert "rust-lang/crates-io-auth-action@v1" in rust
    assert "publish_crate_archive.py" in rust
    assert "--execute" in rust
    assert "cargo publish" not in rust

    python = PYTHON_PUBLISH.read_text()
    assert "acceptance_run_id:" in python
    assert python.count("default: 0.7.0") == 2
    assert "release_sha:" in python
    assert "dry_run:" in python
    assert "recovery_from_main:" in python
    assert "refs/heads/release/*" in python
    assert "if: inputs.dry_run == false" in python
    assert 'Requirement("adam-core>=0.7.0,<0.8.0")' in python
    assert '"adam-core": "<0.8.0,>=0.7.0"' in python
    assert "prepare_pypi_upload.py" in python
    assert "packages-dir: upload-dist/" in python
    assert "skip-existing" not in python


def test_wheel_candidate_targets_public_070_but_never_builds_unpublished_core() -> None:
    source = WHEEL_CANDIDATE.read_text()
    assert 'ADAM_CORE_RELEASE_VERSION: "0.7.0"' in source
    assert f'ADAM_CORE_ACCEPTANCE_REF: "{CORE_REF}"' in source
    assert "repository: B612-Asteroid-Institute/adam_core" in source
    assert "ref: ${{ env.ADAM_CORE_ACCEPTANCE_REF }}" in source
    assert "rev-parse 'v0.7.0^{commit}'" in source
    assert '"adam-core>=0.7.0,<0.8.0"' in source
    assert "python -m pip download --no-cache-dir" in source
    assert '"adam-core==$ADAM_CORE_RELEASE_VERSION"' in source
    assert "Build adam-core native wheel" not in source
    assert "python -m pip install ./adam-core" not in source
    assert "[patch.crates-io]" not in source
    assert "run_clean_room_artifact_acceptance.py" in source
    assert "run_core_compatibility_matrix.py" not in source
    assert "Full 30-workload current benchmark" in source


def test_no_workflow_claims_the_incompatible_061_floor() -> None:
    combined = "\n".join(path.read_text() for path in WORKFLOWS.glob("*.yml"))
    assert "UNASSIGNED_PROPAGATOR_V2_RELEASE" not in combined
    assert "e1dc662ea24b5c73b212629bfa818bb602527086" not in combined
    assert "adam-core>=0.6,<0.7" not in combined
    assert "adam-core==0.6.1" not in combined


def _workflow(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text())


def _needs(job: dict) -> set[str]:
    value = job.get("needs", [])
    if isinstance(value, str):
        return {value}
    return set(value)


def _ancestors(jobs: dict[str, dict], job_id: str) -> set[str]:
    result: set[str] = set()
    pending = list(_needs(jobs[job_id]))
    while pending:
        dependency = pending.pop()
        if dependency in result:
            continue
        if dependency not in jobs:
            raise ValueError(f"unknown job dependency {dependency!r}")
        result.add(dependency)
        pending.extend(_needs(jobs[dependency]))
    return result


def _step_run(step: dict) -> str:
    value = step.get("run", "")
    return value if isinstance(value, str) else ""


def _shell_commands(steps: list[dict]) -> list[list[str]]:
    commands: list[list[str]] = []
    for step in steps:
        logical = _step_run(step).replace("\\\n", " ")
        for line in logical.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                commands.append(shlex.split(line))
            except ValueError:
                continue
    return commands


def _has_python_script(
    commands: list[list[str]], script: str, required: set[str] | None = None
) -> bool:
    required = required or set()
    return any(
        len(command) >= 2
        and command[0] in {"python", "python3"}
        and command[1] == script
        and required <= set(command[2:])
        for command in commands
    )


def _command_start(command: list[str]) -> int:
    start = 0
    while start < len(command) and "=" in command[start]:
        start += 1
    return start


def _has_registry_verification(steps: list[dict]) -> bool:
    return _has_python_script(
        _shell_commands(steps), "migration/scripts/verify_registry_locks.py"
    )


def _assert_registry_release_governance(workflow: dict, kind: str) -> None:
    jobs = workflow["jobs"]
    gate = "compatibility-floor"
    if gate not in jobs:
        raise ValueError("missing compatibility/registry/lock gate")
    gate_commands = _shell_commands(jobs[gate].get("steps", []))
    if (
        not _has_python_script(
            gate_commands, "migration/scripts/verify_source_pair_contract.py"
        )
        or ["exit", "1"] not in gate_commands
    ):
        raise ValueError("release gate must verify transition state and fail closed")
    for job_id in jobs:
        if job_id != gate and gate not in _ancestors(jobs, job_id):
            raise ValueError(f"release job {job_id!r} bypasses compatibility gate")

    if kind == "wheel-candidate":
        steps = jobs["artifact-acceptance"]["steps"]
        commands = _shell_commands(steps)
        lock_verified = _has_python_script(
            commands,
            "migration/scripts/verify_registry_locks.py",
            {"--python-lock", "--rust-lock"},
        )
        if ["pdm", "lock", "--check"] not in commands or not lock_verified:
            raise ValueError("wheel release must verify both registry locks")
        builds = [
            step
            for step in steps
            if str(step.get("uses", "")).startswith("PyO3/maturin-action@")
        ]
        if len(builds) != 2 or any(
            "--locked" not in str(step.get("with", {}).get("args", ""))
            for step in builds
        ):
            raise ValueError("every release wheel build must use --locked")
        full_commands = _shell_commands(jobs["full-current-benchmark"]["steps"])
        repository_cargo = [
            command
            for command in full_commands
            if command[_command_start(command) : _command_start(command) + 2]
            == ["cargo", "test"]
        ]
        if not repository_cargo or any(
            "--locked" not in command for command in repository_cargo
        ):
            raise ValueError("release benchmark Rust tests must use --locked")
        benchmark_step = next(
            (
                step
                for step in jobs["full-current-benchmark"]["steps"]
                if step.get("name")
                == "Install the exact accepted Core and ASSIST wheels with dev dependencies"
            ),
            {},
        )
        benchmark_commands = _shell_commands([benchmark_step])
        wheel_install = [
            "python",
            "-m",
            "pip",
            "install",
            "$core_wheel",
            "${assist_wheel}[dev]",
        ]
        if (
            wheel_install not in benchmark_commands
            or not _has_python_script(
                benchmark_commands,
                "migration/scripts/verify_dev_environment.py",
                {"--require-wheel"},
            )
            or any("-e" in command for command in benchmark_commands)
        ):
            raise ValueError("release benchmark must use the accepted wheel dev extra")
    elif kind == "rust-candidate":
        steps = jobs["package-and-test"]["steps"]
        commands = _shell_commands(steps)
        if not _has_python_script(
            commands,
            "migration/scripts/verify_registry_locks.py",
            {"--rust-lock"},
        ):
            raise ValueError("Rust release must verify the registry Cargo lock")
        for subcommand in ("clippy", "test", "doc", "package"):
            matching = [
                command
                for command in commands
                if command[_command_start(command) : _command_start(command) + 2]
                == ["cargo", subcommand]
            ]
            if not matching or any("--locked" not in command for command in matching):
                raise ValueError(
                    f"Rust release command lost --locked: cargo {subcommand}"
                )
        if _needs(jobs["latest-stable-compatibility"]) != {gate}:
            raise ValueError(
                "latest-stable compatibility must be unavailable behind gate"
            )
        stable_step = next(
            (
                step
                for step in jobs["latest-stable-compatibility"]["steps"]
                if step.get("name") == "Check and test supported feature configurations"
            ),
            {},
        )
        stable_commands = [
            command
            for command in _shell_commands([stable_step])
            if command[_command_start(command) : _command_start(command) + 1]
            == ["cargo"]
        ]
        if not stable_commands or any(
            "--locked" not in command for command in stable_commands
        ):
            raise ValueError("latest-stable repository builds must use --locked")
    elif kind == "python-publish":
        commands = _shell_commands(jobs["collect-tested-artifacts"].get("steps", []))
        if not _has_python_script(
            commands,
            "migration/scripts/verify_registry_locks.py",
            {"--python-lock", "--rust-lock"},
        ):
            raise ValueError("Python publication must verify both lock graphs")
    elif kind == "rust-publish":
        commands = _shell_commands(jobs["publish"].get("steps", []))
        if not _has_python_script(
            commands,
            "migration/scripts/verify_registry_locks.py",
            {"--rust-lock"},
        ):
            raise ValueError("Rust publication must verify the registry Cargo lock")
    else:
        raise AssertionError(kind)

    for job_id, job in jobs.items():
        for step in job.get("steps", []):
            uses = str(step.get("uses", "")).lower()
            dangerous = "pypi-publish" in uses or "upload-artifact" in uses
            for command in _shell_commands([step]):
                start = _command_start(command)
                executable = command[start:]
                dangerous |= (
                    executable[:2] in (["cargo", "publish"], ["gh", "release"])
                    or executable[:3] == ["python", "-m", "twine"]
                    and "upload" in executable[3:]
                    or executable[:2]
                    == [
                        "python",
                        "migration/scripts/publish_crate_archive.py",
                    ]
                    and "--execute" in executable[2:]
                )
            if dangerous and gate not in _ancestors(jobs, job_id):
                raise ValueError(
                    f"publication step in {job_id!r} bypasses release gate"
                )
            if dangerous:
                step_index = job.get("steps", []).index(step)
                protected = _has_registry_verification(
                    job.get("steps", [])[:step_index]
                ) or any(
                    _has_registry_verification(jobs[ancestor].get("steps", []))
                    for ancestor in _ancestors(jobs, job_id)
                )
                if not protected:
                    raise ValueError(
                        f"publication step in {job_id!r} bypasses registry lock verification"
                    )


def test_release_governance_is_semantically_gated_and_locked() -> None:
    cases = (
        (WHEEL_CANDIDATE, "wheel-candidate"),
        (RUST_CANDIDATE, "rust-candidate"),
        (PYTHON_PUBLISH, "python-publish"),
        (RUST_PUBLISH, "rust-publish"),
    )
    for path, kind in cases:
        _assert_registry_release_governance(_workflow(path), kind)


def test_release_governance_rejects_lock_and_locked_mutations() -> None:
    from copy import deepcopy

    wheel = _workflow(WHEEL_CANDIDATE)
    missing_lock = deepcopy(wheel)
    steps = missing_lock["jobs"]["artifact-acceptance"]["steps"]
    steps[:] = [
        step
        for step in steps
        if "Verify registry-only release locks" != step.get("name")
    ]
    with pytest.raises(ValueError, match="verify both registry locks"):
        _assert_registry_release_governance(missing_lock, "wheel-candidate")

    fake_lock = deepcopy(wheel)
    lock_step = next(
        step
        for step in fake_lock["jobs"]["artifact-acceptance"]["steps"]
        if step.get("name") == "Verify registry-only release locks"
    )
    lock_step["run"] = (
        'echo "python migration/scripts/verify_registry_locks.py '
        '--python-lock --rust-lock"'
    )
    with pytest.raises(ValueError, match="verify both registry locks"):
        _assert_registry_release_governance(fake_lock, "wheel-candidate")

    unlocked = deepcopy(wheel)
    build = next(
        step
        for step in unlocked["jobs"]["artifact-acceptance"]["steps"]
        if str(step.get("uses", "")).startswith("PyO3/maturin-action@")
    )
    build["with"]["args"] = build["with"]["args"].replace("--locked", "")
    with pytest.raises(ValueError, match="must use --locked"):
        _assert_registry_release_governance(unlocked, "wheel-candidate")

    rust = _workflow(RUST_CANDIDATE)
    missing_rust_lock = deepcopy(rust)
    steps = missing_rust_lock["jobs"]["package-and-test"]["steps"]
    steps[:] = [
        step for step in steps if step.get("name") != "Verify registry-only Core lock"
    ]
    with pytest.raises(ValueError, match="registry Cargo lock"):
        _assert_registry_release_governance(missing_rust_lock, "rust-candidate")

    unlocked_rust = deepcopy(rust)
    package_step = next(
        step
        for step in unlocked_rust["jobs"]["package-and-test"]["steps"]
        if step.get("name") == "Package and verify the crate once"
    )
    package_step["run"] = package_step["run"].replace(
        'cargo package --manifest-path "$manifest" --locked',
        'cargo package --manifest-path "$manifest"',
    )
    with pytest.raises(ValueError, match="cargo package"):
        _assert_registry_release_governance(unlocked_rust, "rust-candidate")

    unlocked_stable = deepcopy(rust)
    stable_step = next(
        step
        for step in unlocked_stable["jobs"]["latest-stable-compatibility"]["steps"]
        if step.get("name") == "Check and test supported feature configurations"
    )
    stable_step["run"] = stable_step["run"].replace(
        'cargo check --manifest-path "$manifest" --locked',
        'cargo check --manifest-path "$manifest"',
        1,
    )
    with pytest.raises(ValueError, match="latest-stable repository builds"):
        _assert_registry_release_governance(unlocked_stable, "rust-candidate")


def test_release_governance_rejects_ungated_publication_mutations() -> None:
    from copy import deepcopy

    workflow = deepcopy(_workflow(PYTHON_PUBLISH))
    workflow["jobs"]["rogue-upload"] = {
        "runs-on": "ubuntu-latest",
        "steps": [{"uses": "actions/upload-artifact@v4"}],
    }
    with pytest.raises(ValueError, match="bypasses compatibility gate"):
        _assert_registry_release_governance(workflow, "python-publish")

    workflow["jobs"]["rogue-upload"]["needs"] = "compatibility-floor"
    with pytest.raises(ValueError, match="bypasses registry lock verification"):
        _assert_registry_release_governance(workflow, "python-publish")

    workflow = deepcopy(_workflow(RUST_PUBLISH))
    workflow["jobs"]["publish"].pop("needs")
    with pytest.raises(ValueError, match="bypasses compatibility gate"):
        _assert_registry_release_governance(workflow, "rust-publish")
