from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "pip-build-lint-test-coverage.yml"
RELEASE_WORKFLOW = ROOT / ".github" / "workflows" / "release-candidate-wheel-matrix.yml"


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


def test_release_matrix_generates_and_inspects_core_runtime_version() -> None:
    workflow = RELEASE_WORKFLOW.read_text()
    writer = "python adam-core/migration/scripts/write_maturin_version.py"
    builder = "uses: PyO3/maturin-action@v1"
    inspector = "python adam-core/migration/scripts/check_wheel_artifacts.py"

    assert workflow.index(writer) < workflow.index(builder) < workflow.index(inspector)
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
    assert "assist_public_semantics_residuals_ci.json" in workflow
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
