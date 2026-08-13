from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "pip-build-lint-test-coverage.yml"


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
