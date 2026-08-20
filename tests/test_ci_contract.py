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
    assert "--quick --lanes tiny --require-native" in str(
        scripts["benchmark-current-ci"]
    )
    assert "assist-current-benchmark" in workflow
    assert "migration/artifacts/benchmark_current_assist_ci.json" in workflow


def test_release_matrix_generates_and_inspects_core_runtime_version() -> None:
    workflow = RELEASE_WORKFLOW.read_text()
    writer = "python adam-core/migration/scripts/write_maturin_version.py"
    builder = "uses: PyO3/maturin-action@v1"
    inspector = "python adam-core/migration/scripts/check_wheel_artifacts.py"

    assert workflow.index(writer) < workflow.index(builder) < workflow.index(inspector)
    assert 'ADAM_CORE_PREVIEW_VERSION: "0.5.6rc5"' in workflow
    assert 'ADAM_ASSIST_PREVIEW_VERSION: "0.4.0rc6"' in workflow
