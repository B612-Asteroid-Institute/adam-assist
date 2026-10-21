from pathlib import Path

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
    assert (
        workflow.count('python -m pip install "$core_wheel" "${assist_wheel}[dev]"')
        == 2
    )
    assert workflow.count("verify_dev_environment.py --require-wheel") == 2
    assert "verify_workflow_policy.py" in workflow
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


def test_structural_workflow_policy_accepts_repository() -> None:
    from migration.scripts.verify_workflow_policy import verify

    verify()
