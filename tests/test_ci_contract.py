from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "pip-build-lint-test-coverage.yml"
RELEASE_WORKFLOW = ROOT / ".github" / "workflows" / "release-candidate-wheel-matrix.yml"
RUST_CANDIDATE_WORKFLOW = (
    ROOT / ".github" / "workflows" / "rust-crate-release-candidate.yml"
)
RUST_PUBLISH_WORKFLOW = ROOT / ".github" / "workflows" / "publish-rust-crate.yml"


def _scripts() -> dict[str, object]:
    with (ROOT / "pyproject.toml").open("rb") as pyproject_file:
        return tomllib.load(pyproject_file)["tool"]["pdm"]["scripts"]


def test_normal_ci_runs_direct_rust_quality_and_current_only_benchmark() -> None:
    scripts = _scripts()
    workflow = WORKFLOW.read_text()

    assert "rust-quality" in scripts
    assert "pdm run rust-quality" in workflow
    assert 'MATURIN_PEP517_ARGS: "--locked"' in workflow
    assert "ADAM_CORE_REF" not in workflow
    assert "cba63f" not in workflow
    assert "Checkout exact prepublication adam-core source pair" not in workflow
    assert "[patch.crates-io]" not in workflow
    assert workflow.count("Verify registry-only Core locks") == 4
    assert workflow.count("--python-lock --rust-lock") == 4
    assert "pdm run benchmark-current-ci" in workflow
    assert "pdm install --frozen-lockfile -G dev" in workflow
    assert "fail-fast: false" in workflow
    assert "run: pdm install -G dev" not in workflow
    assert "assist-current-benchmark" in workflow
    assert "migration/artifacts/benchmark_current_assist_ci.json" in workflow


def test_every_rust_workflow_uses_the_msrv_toolchain_and_components() -> None:
    for workflow in (ROOT / ".github" / "workflows").glob("*.yml"):
        source = workflow.read_text()
        if "dtolnay/rust-toolchain@" not in source:
            continue
        if workflow.name == "rust-crate-release-candidate.yml":
            assert "dtolnay/rust-toolchain@stable" in source
        else:
            assert "dtolnay/rust-toolchain@stable" not in source, workflow.name
        assert "dtolnay/rust-toolchain@1.87.0" in source, workflow.name
        assert "components: rustfmt, clippy" in source, workflow.name


def test_rust_crate_workflows_package_once_and_publish_tested_bytes() -> None:
    candidate = RUST_CANDIDATE_WORKFLOW.read_text()
    assert 'RUST_RELEASE_VERSION: "0.4.1"' in candidate
    assert 'CORE_RUST_VERSION: "0.5.8"' in candidate
    assert "RELEASE_CHANNEL: stable" in candidate
    assert "ADAM_CORE_REF" not in candidate
    assert "cba63f" not in candidate
    assert ".ci/adam-core" not in candidate
    assert "[patch.crates-io]" not in candidate
    assert '"release-candidate/adam-assist-0.4.0rc7"' in candidate
    assert '"release/adam-assist-*"' in candidate
    assert "Verify registry-only Core lock" in candidate
    assert "verify_registry_locks.py" in candidate
    assert '--core-version "$CORE_RUST_VERSION" --rust-lock' in candidate
    assert "cargo package --manifest-path" in candidate
    assert "--locked" in candidate
    assert 'RUSTDOCFLAGS="-D warnings -D missing-docs"' in candidate
    assert "adam-assist-rust-crate-publication-set" in candidate
    assert "publish_crate_archive.py" in candidate
    assert '--expected-core-version "$CORE_RUST_VERSION"' in candidate
    assert '--channel "$RELEASE_CHANNEL"' in candidate
    assert 'adam_core = "=$CORE_RUST_VERSION"' in candidate
    assert 'adam_core = "=$CORE_VERSION"' in candidate
    assert "core-range-compatibility:" in candidate
    assert 'core-version: ["0.5.7", "0.5.8"]' in candidate
    assert "Download the one tested adam-assist crate" in candidate
    assert "verify_rust_core_line.py" in candidate
    assert "Latest-stable Rust compatibility (non-authoritative)" in candidate
    assert "dtolnay/rust-toolchain@stable" in candidate
    assert (
        'cargo check --manifest-path "$manifest" --locked --all-features' in candidate
    )
    assert (
        'cargo test --manifest-path "$manifest" --locked --features python' in candidate
    )
    assert (
        'cargo test --manifest-path "$manifest" --locked --lib --no-default-features'
        in candidate
    )
    assert 'test ! -e "$consumer/Cargo.lock"' in candidate
    assert "AssistPropagator::from_default_kernels" in candidate
    assert "cargo publish" not in candidate

    publisher = RUST_PUBLISH_WORKFLOW.read_text()
    assert "candidate_run_id:" in publisher
    assert "expected_core_version:" in publisher
    assert "release_channel:" in publisher
    assert "expected_python_version:" in publisher
    assert (
        'test "$GITHUB_REF" = "refs/tags/v${{ inputs.expected_python_version }}"'
        in publisher
    )
    assert "cargo_version_to_pep440" in publisher
    assert "inputs.release_channel == 'stable' && 'crates-io'" in publisher
    assert "adam-assist-rust-crate-publication-set" in publisher
    assert "trusted-publishing" in publisher
    assert "bootstrap-token" not in publisher
    assert "CRATES_IO_BOOTSTRAP_TOKEN" not in publisher
    assert "rust-lang/crates-io-auth-action@v1" in publisher
    assert "publish_crate_archive.py" in publisher
    assert "Verify registry-only Core Cargo lock" in publisher
    assert "verify_registry_locks.py" in publisher
    assert "--execute" in publisher
    assert "cargo publish" not in publisher

    python_publisher = (ROOT / ".github" / "workflows" / "publish.yml").read_text()
    assert "public Rust prerequisite" in python_publisher
    assert "migration/scripts/verify_release.py" in python_publisher
    assert "api/v1/crates/adam_assist" not in python_publisher
    assert "expected_rust_version:" in python_publisher
    assert "expected_adam_core_rust_version:" in python_publisher
    assert "release_channel:" in python_publisher
    assert "release_sha:" in python_publisher
    assert "recovery_from_main:" in python_publisher
    assert "dry_run:" in python_publisher
    assert "refs/heads/main" in python_publisher
    assert "refs/heads/release/*" in python_publisher
    assert "if: inputs.dry_run == false" in python_publisher
    assert "ref: v${{ inputs.expected_version }}" in python_publisher
    assert "EXPECTED_SHA: ${{ inputs.release_sha }}" in python_publisher
    assert "unconditional_requirements != expected_requirements" in python_publisher
    assert '"adam-core": "<0.6,>=0.5.7"' in python_publisher
    assert "selected Core" in python_publisher
    assert "prepare_pypi_upload.py" in python_publisher
    assert "Verify registry-only Core locks" in python_publisher
    assert "--python-lock --rust-lock" in python_publisher
    assert "packages-dir: upload-dist/" in python_publisher
    assert "skip-existing" not in python_publisher
    assert "inputs.release_channel == 'stable' && 'pypi'" in python_publisher
    assert "testpypi" not in python_publisher.lower()
    assert "to pypi" in python_publisher


def test_release_matrix_builds_once_and_clean_room_accepts_registry_wheels() -> None:
    workflow = RELEASE_WORKFLOW.read_text()

    # The final candidate remains registry-only. The Core checkout is pinned
    # acceptance tooling, never an unpublished Python/Rust source pair.
    assert "adam_core_ref" not in workflow
    assert "ADAM_CORE_REF" not in workflow
    assert "cba63f" not in workflow
    assert "Checkout adam-core candidate" not in workflow
    assert "Checkout exact adam-core candidate" not in workflow
    assert "Patch unpublished Core crates" not in workflow
    assert "[patch.crates-io]" not in workflow
    assert ".cargo/config.toml" not in workflow
    assert ".ci/adam-core" not in workflow
    assert "Write adam-core runtime version" not in workflow
    assert "python -m build" not in workflow
    assert "Build adam-core" not in workflow
    assert "Build adam-core manylinux wheel" not in workflow
    assert "Build adam-core native wheel" not in workflow
    assert "python -m pip install ./adam-core" not in workflow

    # All twelve platform/Python lanes use immutable tooling from the public
    # v0.5.8 Core release commit, including the exact reviewed smoke script.
    assert 'python-version: ["3.11", "3.12", "3.13"]' in workflow
    for platform in (
        "manylinux-x86_64",
        "manylinux-aarch64",
        "macos-arm64",
        "macos-x86_64",
    ):
        assert f"name: {platform}" in workflow
    assert (
        'ADAM_CORE_ACCEPTANCE_REF: "fff90ff458e383beacad1db79cda0484b05baffc"'
        in workflow
    )
    assert "Checkout immutable public adam-core acceptance tooling" in workflow
    assert "repository: B612-Asteroid-Institute/adam_core" in workflow
    assert "ref: ${{ env.ADAM_CORE_ACCEPTANCE_REF }}" in workflow
    assert "path: adam-core-acceptance" in workflow
    assert "rev-parse 'v0.5.8^{commit}'" in workflow
    assert (
        "464859e82bd9aea975f1209b040a97299230dd37bf370ab0132beefcbc29435f" in workflow
    )
    assert (
        "bb2df3751929b4e0391e8300d7e51eb5815f1b0b3c012b8a070052b362b24b77" in workflow
    )
    assert workflow.count("run_clean_room_artifact_acceptance.py") == 2
    assert workflow.count("clean_room_artifact_smoke.py") == 1

    assert 'ADAM_CORE_RELEASE_VERSION: "0.5.8"' in workflow
    assert 'ADAM_ASSIST_RELEASE_VERSION: "0.4.1"' in workflow
    assert "Verify registry-only release locks" in workflow
    assert "pdm lock --check" in workflow
    assert "--python-lock --rust-lock" in workflow
    assert "Build adam-assist manylinux wheel once" in workflow
    assert "Build adam-assist native wheel once" in workflow
    assert workflow.count("args: --release --locked") == 2

    # The only runtime inputs are the once-built ASSIST wheel and the exact
    # public PyPI Core wheel. The Core driver must consume, not rebuild, them.
    assert "Prepare exact registry wheelhouse for clean-room acceptance" in workflow
    assert "acceptance-prebuilt-wheelhouse" in workflow
    assert "cp wheelhouse/adam_assist-*.whl" in workflow
    assert "--index-url https://pypi.org/simple" in workflow
    assert "--only-binary=:all: --no-deps" in workflow
    assert '"adam-core==$ADAM_CORE_RELEASE_VERSION"' in workflow
    assert "Run exact registry-wheel clean-room acceptance" in workflow
    assert (
        "python adam-core-acceptance/migration/scripts/"
        "run_clean_room_artifact_acceptance.py" in workflow
    )
    assert (
        '--prebuilt-wheelhouse "$RUNNER_TEMP/acceptance-prebuilt-wheelhouse"'
        in workflow
    )
    assert '--adam-core-repo "$GITHUB_WORKSPACE/adam-core-acceptance"' in workflow
    assert '--adam-core-ref "$ADAM_CORE_ACCEPTANCE_REF"' in workflow
    assert '--adam-assist-repo "$GITHUB_WORKSPACE"' in workflow
    assert "--adam-assist-ref HEAD" in workflow
    assert "forward/backward/same-epoch propagation" in workflow
    assert "observer ephemeris through embedded Core SPICE" in workflow
    assert "offline kernel-cache" in workflow
    assert "installed-wheel provenance" in workflow
    assert "Inspect public adam-core wheel contents" in workflow
    assert "check_wheel_artifacts.py" in workflow
    assert "Verify clean-room artifact identities and platform tags" in workflow
    assert "acceptance-report.json" in workflow
    assert "acceptance/invocation/smoke-report.json" in workflow
    assert "acceptance/wheelhouse/*.whl" in workflow
    assert "acceptance/logs/*.log" in workflow

    # Backward compatibility reuses those exact accepted ASSIST/Core 0.5.8
    # bytes, then swaps only the independently downloaded public Core 0.5.7.
    assert (
        "Test the same adam-assist wheel with public Core 0.5.7 and 0.5.8" in workflow
    )
    assert "run_core_compatibility_matrix.py" in workflow
    assert '"adam-core==0.5.7"' in workflow
    assert '"adam-core>=0.5.7,<0.6"' in workflow
    assert 'find "$RUNNER_TEMP/acceptance/wheelhouse"' in workflow
    assert "core-compatibility-summary.json" in workflow

    assert "full-current-benchmark:" in workflow
    full_job_header = workflow.split("  full-current-benchmark:", maxsplit=1)[1].split(
        "    steps:", maxsplit=1
    )[0]
    assert "needs: artifact-acceptance" in full_job_header
    assert "runs-on: macos-14" in full_job_header
    assert "Full 35-workload current benchmark" in full_job_header
    assert "Install public Core 0.5.8 and adam-assist candidate" in workflow
    assert 'python -m pip install "adam-core==$ADAM_CORE_RELEASE_VERSION"' in workflow
    assert 'importlib.metadata.version("adam-core") == "0.5.8"' in workflow
    assert "Pin frozen-fixture kernel bytes" in workflow
    assert "assist_public_semantics_fixture_2026-05-20.json" in workflow
    assert "ADAM_CORE_RS_ASSIST_PLANETS_PATH" in workflow
    assert "ADAM_CORE_RS_ASSIST_ASTEROIDS_PATH" in workflow
    assert "assist_kernel_identity_ci.json" in workflow
    assert (
        "${{ github.workspace }}/migration/artifacts/"
        "assist_public_semantics_residuals_ci.json" in workflow
    )
    assert (
        "cargo test --locked --manifest-path rust/adam_assist_rs/Cargo.toml -- --ignored"
        in workflow
    )
    assert (
        "python -m pytest -m live tests/test_propagate.py tests/test_ephemeris.py"
        in workflow
    )
    assert "--lanes tiny small large" in workflow
    assert "--repeats 5" in workflow
    assert "--require-native" in workflow
    assert "assist-current-benchmark-full" in workflow
