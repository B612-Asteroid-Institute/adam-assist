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
    assert "[patch.crates-io]" not in workflow
    assert "pdm run benchmark-current-ci" in workflow
    assert "assist-current-benchmark" in workflow
    assert "migration/artifacts/benchmark_current_assist_ci.json" in workflow


def test_every_rust_workflow_uses_the_msrv_toolchain_and_components() -> None:
    for workflow in (ROOT / ".github" / "workflows").glob("*.yml"):
        source = workflow.read_text()
        if "dtolnay/rust-toolchain@" not in source:
            continue
        assert "dtolnay/rust-toolchain@stable" not in source, workflow.name
        assert "dtolnay/rust-toolchain@1.87.0" in source, workflow.name
        assert "components: rustfmt, clippy" in source, workflow.name


def test_rust_crate_workflows_package_once_and_publish_tested_bytes() -> None:
    candidate = RUST_CANDIDATE_WORKFLOW.read_text()
    assert 'RUST_PREVIEW_VERSION: "0.4.0-rc.6"' in candidate
    assert '"release-candidate/adam-assist-0.4.0rc6"' in candidate
    assert "cargo package --manifest-path" in candidate
    assert "--locked" in candidate
    assert 'RUSTDOCFLAGS="-D warnings -D missing-docs"' in candidate
    assert "adam-assist-rust-crate-publication-set" in candidate
    assert "publish_crate_archive.py" in candidate
    assert 'adam_core = "=0.1.0-rc.4"' in candidate
    assert "AssistPropagator::from_default_kernels" in candidate
    assert "cargo publish" not in candidate

    publisher = RUST_PUBLISH_WORKFLOW.read_text()
    assert "candidate_run_id:" in publisher
    assert 'test "$GITHUB_REF" = "refs/tags/v0.4.0rc6"' in publisher
    assert "adam-assist-rust-crate-publication-set" in publisher
    assert "bootstrap-token" in publisher
    assert "trusted-publishing" in publisher
    assert "CRATES_IO_BOOTSTRAP_TOKEN" in publisher
    assert "rust-lang/crates-io-auth-action@v1" in publisher
    assert "publish_crate_archive.py" in publisher
    assert "--execute" in publisher
    assert "cargo publish" not in publisher

    python_publisher = (ROOT / ".github" / "workflows" / "publish.yml").read_text()
    assert "Require the matching public Rust crate" in python_publisher
    assert "api/v1/crates/adam_assist/0.4.0-rc.6" in python_publisher
    assert 'test "$(jq -r .version.yanked' in python_publisher
    assert "testpypi" not in python_publisher.lower()
    assert "to pypi" in python_publisher


def test_release_matrix_generates_and_inspects_core_runtime_version() -> None:
    workflow = RELEASE_WORKFLOW.read_text()
    writer = "python adam-core/migration/scripts/write_maturin_version.py"
    builder = "uses: PyO3/maturin-action@v1"
    inspector = "python adam-core/migration/scripts/check_wheel_artifacts.py"

    assert workflow.index(writer) < workflow.index(builder) < workflow.index(inspector)
    assert "61cb2779b49ab2a641975942e3ce82d99b461ece" in workflow
    assert "release-candidate/adam-core-0.5.6rc5" not in workflow
    assert 'ADAM_CORE_PREVIEW_VERSION: "0.5.6rc5"' in workflow
    assert 'ADAM_ASSIST_PREVIEW_VERSION: "0.4.0rc6"' in workflow
    assert "full-current-benchmark:" in workflow
    full_job_header = workflow.split("  full-current-benchmark:", maxsplit=1)[1].split(
        "    steps:", maxsplit=1
    )[0]
    assert "needs: artifact-acceptance" in full_job_header
    assert "runs-on: macos-14" in full_job_header
    assert "Full 35-workload current benchmark" in full_job_header
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
        "cargo test --manifest-path rust/adam_assist_rs/Cargo.toml -- --ignored"
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
