#!/usr/bin/env python3
"""Verify public-release CI and manually gated release workflows.

Core 0.8 and authentic registry locks are public. Release infrastructure stays
workflow-dispatch-only, and every release job requires one bounded-window
repository variable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
NORMAL = "pip-build-lint-test-coverage.yml"
RELEASE_WORKFLOWS = (
    "publish-rust-crate.yml",
    "publish.yml",
    "release-candidate-wheel-matrix.yml",
    "rust-crate-release-candidate.yml",
)
WORKFLOW_INVENTORY = (NORMAL, *RELEASE_WORKFLOWS)
READINESS_VARIABLE = "ADAM_ASSIST_RELEASES_ENABLED"
READINESS_CONDITION = "${{ vars.ADAM_ASSIST_RELEASES_ENABLED == 'true' }}"
PUBLISH_PYPI_CONDITION = (
    "${{ vars.ADAM_ASSIST_RELEASES_ENABLED == 'true' " "&& inputs.dry_run == false }}"
)
PYPI_REF_CONFIRMATION_SCRIPT = """set -euo pipefail
case "$GITHUB_REF" in
  "refs/tags/v$EXPECTED_VERSION")
    test "$DRY_RUN" = "false"
    ;;
  refs/heads/release/*)
    test "$DRY_RUN" = "true"
    ;;
  *) echo "publication must run from the exact release tag or a dry-run release branch" >&2; exit 1 ;;
esac
test "$(git rev-parse HEAD)" = "$EXPECTED_SHA"
if [ "$DRY_RUN" = "true" ]; then
  test "$CONFIRMATION" = "verify adam-assist $EXPECTED_VERSION from $EXPECTED_SHA for pypi"
else
  test "$CONFIRMATION" = "publish adam-assist $EXPECTED_VERSION from $EXPECTED_SHA to pypi"
fi"""
EXPECTED_RELEASE_JOB_CONDITIONS = {
    "publish-rust-crate.yml": {
        "compatibility-floor": READINESS_CONDITION,
        "publish": READINESS_CONDITION,
    },
    "publish.yml": {
        "compatibility-floor": READINESS_CONDITION,
        "collect-tested-artifacts": READINESS_CONDITION,
        "publish-pypi": PUBLISH_PYPI_CONDITION,
    },
    "release-candidate-wheel-matrix.yml": {
        "compatibility-floor": READINESS_CONDITION,
        "artifact-acceptance": READINESS_CONDITION,
        "full-current-benchmark": READINESS_CONDITION,
    },
    "rust-crate-release-candidate.yml": {
        "compatibility-floor": READINESS_CONDITION,
        "package-and-test": READINESS_CONDITION,
        "core-range-compatibility": READINESS_CONDITION,
        "latest-stable-compatibility": READINESS_CONDITION,
    },
}
RELEASE_AUTOMATION_STATE = "present-and-version-aligned"
CORE_RELEASE_COMMIT = "37fcd1c6f5d38d70335e56c3b095658a466d1f50"
CORE_PUBLIC_REF = CORE_RELEASE_COMMIT
ALLOWED_ACTIONS = {
    "actions/checkout@v4",
    "actions/setup-python@v5",
    "actions/download-artifact@v4",
    "actions/upload-artifact@v4",
    "coverallsapp/github-action@v2.0.0",
    "dtolnay/rust-toolchain@1.87.0",
}
PUBLISH_ACTION_MARKERS = (
    "pypa/gh-action-pypi-publish",
    "pyo3/maturin-action",
    "action-gh-release",
    "upload-release-asset",
    "release-action",
    "crates-io-auth-action",
)
REGISTRY_CREDENTIAL = re.compile(
    r"(?i)(?:pypi|twine|crates(?:_io)?|cargo_registry|maturin|uv.{0,12}publish)"
    r".{0,24}(?:token|password|secret)"
)
SECRET_IDENTIFIER = re.compile(r"(?i)\bsecrets\b")
INERT_RELEASE_HELPERS = (
    "prepare_pypi_upload",
    "publish_crate_archive",
    "verify_release",
)
WORKFLOW_SHA256 = "d43bf3f3480044830ef7f583f736000359cc5f8938ff247f0232af60606baf13"


class WorkflowPolicyError(ValueError):
    """Raised when CI could publish without authorization or skip validation."""


def _fail(location: str, message: str) -> None:
    raise WorkflowPolicyError(f"{location}: {message}")


def _on(workflow: Mapping[Any, Any]) -> Any:
    return workflow.get("on", workflow.get(True))


def load_workflows(root: Path = ROOT) -> dict[str, dict[str, Any]]:
    directory = root / ".github" / "workflows"
    paths = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".yml", ".yaml"}
    )
    names = tuple(path.name for path in paths)
    expected = tuple(sorted(WORKFLOW_INVENTORY))
    if names != expected:
        _fail("workflows", f"workflow inventory {names!r} != {expected!r}")
    workflows: dict[str, dict[str, Any]] = {}
    for path in paths:
        document = yaml.safe_load(path.read_text())
        if not isinstance(document, dict) or not isinstance(document.get("jobs"), dict):
            _fail(path.name, "workflow must contain a jobs mapping")
        workflows[path.name] = document
    return workflows


def _steps(job: Mapping[str, Any]) -> list[dict[str, Any]]:
    steps = job.get("steps", [])
    if not isinstance(steps, list) or not all(isinstance(step, dict) for step in steps):
        raise WorkflowPolicyError("job steps must be mappings")
    return steps


def _run(step: Mapping[str, Any]) -> str:
    value = step.get("run", "")
    return value if isinstance(value, str) else ""


def _run_mentions_publication(run: str) -> bool:
    patterns = (
        r"\b(?:python3?|uvx)\s+-m\s+twine\s+upload\b",
        r"\btwine\s+upload\b",
        r"\b(?:cargo|uv|pdm)\s+publish\b",
        r"\bmaturin\s+(?:publish|upload)\b",
        r"\bgh\s+release\s+(?:create|upload)\b",
        r"\bgh\s+api\b[^\n]*\breleases?\b",
        r"\bgit\s+(?:push|tag)\b",
        r"\b(?:curl|wget)\b[^\n]*(?:--upload-file|-T|--post-file|--data-binary)",
    )
    return any(re.search(pattern, run, re.IGNORECASE) for pattern in patterns)


def _walk(value: Any) -> list[Any]:
    found = [value]
    if isinstance(value, dict):
        for key, child in value.items():
            found.extend(_walk(key))
            found.extend(_walk(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_walk(child))
    return found


def _step_by_name(job: Mapping[str, Any], name: str) -> dict[str, Any]:
    matches = [step for step in _steps(job) if step.get("name") == name]
    if len(matches) != 1:
        raise WorkflowPolicyError(f"expected exactly one step named {name!r}")
    return matches[0]


def _require_condition(step: Mapping[str, Any], expected: str | None) -> None:
    actual = step.get("if")
    if actual != expected:
        _fail(str(step.get("name", "<step>")), f"condition {actual!r} != {expected!r}")


def _verify_exact_structure(workflow: Mapping[str, Any]) -> None:
    if set(workflow) != {"name", True, "permissions", "concurrency", "env", "jobs"}:
        _fail(NORMAL, "workflow-level keys drifted from the exact CI policy")
    if workflow.get("concurrency") != {
        "group": "ci-${{ github.ref }}",
        "cancel-in-progress": True,
    }:
        _fail(NORMAL, "concurrency policy drifted")

    jobs = workflow["jobs"]
    allowed_job_keys = {
        "released-core": {"name", "runs-on", "strategy", "steps", "timeout-minutes"},
        "rust-quality": {"name", "runs-on", "steps", "timeout-minutes"},
        "benchmark": {"runs-on", "needs", "steps", "timeout-minutes"},
    }
    shell_steps = {
        ("released-core", "Verify exact released-Core pair contract"),
        ("released-core", "Build Assist and install released Core wheel pair"),
        ("rust-quality", "Verify released registry pair and run Rust gates"),
        ("rust-quality", "Run direct Rust runtime and provenance contracts"),
    }
    for job_id, job in jobs.items():
        unexpected = set(job) - allowed_job_keys[job_id]
        if unexpected:
            _fail(f"{NORMAL}:{job_id}", f"forbidden job keys: {sorted(unexpected)!r}")
        for step in _steps(job):
            unexpected = set(step) - {
                "name",
                "uses",
                "with",
                "run",
                "if",
                "shell",
                "env",
            }
            if unexpected:
                _fail(
                    f"{NORMAL}:{job_id}:{step.get('name', '<step>')}",
                    f"forbidden step keys: {sorted(unexpected)!r}",
                )
            expected_shell = (
                "bash" if (job_id, step.get("name")) in shell_steps else None
            )
            if step.get("shell") != expected_shell:
                _fail(
                    f"{NORMAL}:{job_id}:{step.get('name', '<step>')}",
                    "custom or missing shell override",
                )
            if "env" in step:
                _fail(
                    f"{NORMAL}:{job_id}:{step.get('name', '<step>')}",
                    "step environment is not allowlisted",
                )


def _verify_normal(workflow: Mapping[str, Any]) -> None:
    if workflow.get("name") != "CI":
        _fail(NORMAL, "workflow name must remain CI")
    if _on(workflow) != {
        "push": {"branches": ["main", "v*"]},
        "pull_request": {"branches": ["main", "v*"]},
        "workflow_dispatch": None,
    }:
        _fail(
            NORMAL,
            "allowed triggers are exactly push, pull_request, and workflow_dispatch",
        )
    if workflow.get("permissions") != {"contents": "read"}:
        _fail(NORMAL, "top-level permissions must be exactly contents: read")
    if workflow.get("env") != {
        "ADAM_CORE_VERSION": "0.8.0",
        "ADAM_CORE_RELEASE_COMMIT": CORE_RELEASE_COMMIT,
    }:
        _fail(NORMAL, "Core release identity drifted")

    jobs = workflow["jobs"]
    if set(jobs) != {"released-core", "rust-quality", "benchmark"}:
        _fail(
            NORMAL, "required job inventory is released-core, rust-quality, benchmark"
        )
    for job_id, job in jobs.items():
        location = f"{NORMAL}:{job_id}"
        if any(key in job for key in ("if", "continue-on-error", "permissions")):
            _fail(
                location, "jobs must be unconditional and inherit read-only permissions"
            )
        for index, step in enumerate(_steps(job)):
            step_location = f"{location}:step[{index}]"
            if "continue-on-error" in step:
                _fail(step_location, "continue-on-error is forbidden")
            uses = step.get("uses")
            if uses is not None and uses not in ALLOWED_ACTIONS:
                _fail(step_location, f"action is not allowlisted: {uses!r}")
            if isinstance(uses, str) and any(
                marker in uses.lower() for marker in PUBLISH_ACTION_MARKERS
            ):
                _fail(step_location, "publisher/release action is forbidden")
            if _run_mentions_publication(_run(step)):
                _fail(
                    step_location,
                    "package or GitHub-release publication command is forbidden",
                )
            if isinstance(step.get("with"), Mapping) and "repository" in step["with"]:
                _fail(
                    step_location,
                    "active CI must consume released Core artifacts, not source checkouts",
                )

    _verify_exact_structure(workflow)
    released = jobs["released-core"]
    rust = jobs["rust-quality"]
    benchmark = jobs["benchmark"]
    if released.get("strategy") != {
        "fail-fast": False,
        "matrix": {
            "python-version": ["3.11", "3.12", "3.13", "3.14"],
            "core-kind": ["released-registry"],
        },
    }:
        _fail(
            NORMAL,
            "released-Core matrix must be exactly Python 3.11-3.14 on released-registry",
        )
    if benchmark.get("needs") != ["released-core", "rust-quality"]:
        _fail(NORMAL, "benchmark must require released Core and Rust quality")
    setup_python = next(
        step
        for step in _steps(released)
        if step.get("uses") == "actions/setup-python@v5"
    )
    if setup_python.get("with") != {"python-version": "${{ matrix.python-version }}"}:
        _fail("released-core", "Python setup must use the exact matrix row")

    required_conditions = {
        "Run non-live wheel tests": "matrix.python-version != '3.12'",
        "Run non-live wheel tests with coverage": "matrix.python-version == '3.12'",
        "Lint and typecheck the source": "matrix.python-version == '3.12'",
        "Upload coverage artifact": "matrix.python-version == '3.12' && github.event_name == 'pull_request'",
        "Coverage report to coveralls": "matrix.python-version == '3.12'",
    }
    for name, condition in required_conditions.items():
        _require_condition(_step_by_name(released, name), condition)
    for name in (
        "Verify exact released-Core pair contract",
        "Install build and test tooling",
        "Build Assist and install released Core wheel pair",
        "Verify released Core identity and wheel dependency range",
        "Run released-pair capsule contracts",
        "Verify release-ready workflow policy",
        "Verify authentic registry locks and clean tracked state",
        "Upload released-Core Assist wheel",
    ):
        _require_condition(_step_by_name(released, name), None)
    for name in (
        "Install released Core runtime and kernels",
        "Verify released registry pair and run Rust gates",
        "Run direct Rust runtime and provenance contracts",
    ):
        _require_condition(_step_by_name(rust, name), None)
    for name in (
        "Install the already-tested released-Core wheel pair",
        "Run benchmarks",
        "Run current public/native benchmark smoke",
    ):
        _require_condition(_step_by_name(benchmark, name), None)
    _require_condition(
        _step_by_name(benchmark, "Upload current-only benchmark artifact"), "always()"
    )
    _require_condition(
        _step_by_name(benchmark, "Upload benchmark artifact"), "always()"
    )

    identity = _run(_step_by_name(released, "Verify exact released-Core pair contract"))
    for marker in (
        "${{ matrix.core-kind }}",
        "released-registry",
        "verify_source_pair_contract.py",
    ):
        if marker not in identity:
            _fail(NORMAL, f"released pair identity check is missing {marker!r}")
    build = _run(
        _step_by_name(released, "Build Assist and install released Core wheel pair")
    )
    for marker in (
        "maturin build --release --locked",
        "--only-binary=:all:",
        '"adam-core==$ADAM_CORE_VERSION"',
        '"${assist_wheel}[dev]"',
        "python -m pip check",
        "verify_dev_environment.py --require-wheel",
    ):
        if marker not in build:
            _fail(NORMAL, f"released wheel build/install is missing {marker!r}")
    if "adam_core-*.whl" in build or ".source-pair" in build:
        _fail(NORMAL, "active matrix must not build or source-checkout Core")
    capsule = _run(_step_by_name(released, "Run released-pair capsule contracts"))
    for marker in (
        "native_backend_is_capsule_only",
        "capsule_propagation_preserves_order_nongrav_and_row_failures",
        "capsule_stm_honors_custom_and_inactive_nongrav",
        "capsule_owns_provider_lifetime",
    ):
        if marker not in capsule:
            _fail(NORMAL, f"capsule evidence is missing {marker!r}")
    ordinary_test = _run(_step_by_name(released, "Run non-live wheel tests"))
    if (
        "python -m pytest --benchmark-skip -q tests" not in ordinary_test
        or "|| true" in ordinary_test
    ):
        _fail(NORMAL, "ordinary installed-wheel test command drifted")
    coverage_test = _run(
        _step_by_name(released, "Run non-live wheel tests with coverage")
    )
    if (
        "python -m pytest --benchmark-skip --cov=adam_assist --cov-report=xml -q tests"
        not in coverage_test
    ):
        _fail(NORMAL, "coverage installed-wheel test command drifted")
    lint = _run(_step_by_name(released, "Lint and typecheck the source"))
    for marker in (
        "ruff check",
        "black --check",
        "isort --check-only",
        "mypy --strict",
    ):
        if marker not in lint:
            _fail(NORMAL, f"lint/type gate is missing {marker!r}")
    rust_gate = _run(
        _step_by_name(rust, "Verify released registry pair and run Rust gates")
    )
    for marker in (
        "cargo fmt",
        "cargo check",
        "cargo clippy",
        "cargo test",
        "cargo doc",
    ):
        if marker not in rust_gate:
            _fail(NORMAL, f"Rust quality gate is missing {marker!r}")
    integration = "\n".join(
        _run(_step_by_name(rust, name))
        for name in (
            "Verify released registry pair and run Rust gates",
            "Run direct Rust runtime and provenance contracts",
        )
    )
    for marker in (
        "verify_registry_locks.py",
        "cargo test",
        "cargo run",
        "verify_rust_core_line.py",
        "-- --ignored",
        "ADAM_CORE_RS_ASSIST_PLANETS_PATH",
        "ADAM_CORE_RS_ASSIST_ASTEROIDS_PATH",
    ):
        if marker not in integration:
            _fail(NORMAL, f"released Core Rust integration is missing {marker!r}")
    benchmark_install = _run(
        _step_by_name(benchmark, "Install the already-tested released-Core wheel pair")
    )
    for marker in (
        '"adam-core==$ADAM_CORE_VERSION"',
        '"${assist_wheel}[dev]"',
        "python -m pip check",
    ):
        if marker not in benchmark_install:
            _fail(NORMAL, f"benchmark released-pair install is missing {marker!r}")
    policy = _run(_step_by_name(released, "Verify release-ready workflow policy"))
    if policy.strip() != "python migration/scripts/verify_workflow_policy.py":
        _fail(NORMAL, "normal CI must execute this policy exactly")


def _dispatch_inputs(workflow: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    trigger = _on(workflow)
    if not isinstance(trigger, Mapping) or set(trigger) != {"workflow_dispatch"}:
        _fail(name, "release workflows must be workflow_dispatch-only")
    dispatch = trigger["workflow_dispatch"]
    if not isinstance(dispatch, Mapping):
        return {}
    inputs = dispatch.get("inputs", {})
    if not isinstance(inputs, Mapping):
        _fail(name, "workflow_dispatch inputs must be a mapping")
    return inputs


def _input_default(inputs: Mapping[str, Any], name: str, workflow: str) -> Any:
    entry = inputs.get(name)
    if not isinstance(entry, Mapping):
        _fail(workflow, f"missing workflow input {name!r}")
    return entry.get("default")


def _verify_gated_release_workflow(name: str, workflow: Mapping[str, Any]) -> None:
    inputs = _dispatch_inputs(workflow, name)
    jobs = workflow.get("jobs")
    if not isinstance(jobs, Mapping) or not jobs:
        _fail(name, "release workflow must retain its reviewed job inventory")
    expected_conditions = EXPECTED_RELEASE_JOB_CONDITIONS[name]
    if set(jobs) != set(expected_conditions):
        _fail(name, "release job inventory drifted")
    for job_id, expected_condition in expected_conditions.items():
        job = jobs[job_id]
        if not isinstance(job, Mapping):
            _fail(f"{name}:{job_id}", "job must be a mapping")
        if job.get("if") != expected_condition:
            _fail(f"{name}:{job_id}", "release readiness condition drifted")
        is_diagnostic = (
            name == "release-candidate-wheel-matrix.yml"
            and job_id == "full-current-benchmark"
        )
        if is_diagnostic:
            if job.get("continue-on-error") is not True:
                _fail(f"{name}:{job_id}", "benchmark diagnostic must be non-gating")
        elif "continue-on-error" in job:
            _fail(f"{name}:{job_id}", "continue-on-error is forbidden")

    for value in _walk(workflow):
        if isinstance(value, str) and SECRET_IDENTIFIER.search(value):
            _fail(name, "release workflows may not use repository secrets")

    floor = jobs["compatibility-floor"]
    source_pair = _step_by_name(floor, "Verify exact public-release source-pair state")
    if (
        _run(source_pair).strip()
        != "python migration/scripts/verify_source_pair_contract.py"
    ):
        _fail(name, "compatibility floor must verify the exact public source pair")
    locks = _run(_step_by_name(floor, "Verify authentic Core registry locks"))
    if (
        "verify_registry_locks.py" not in locks
        or "--python-lock --rust-lock" not in locks
    ):
        _fail(name, "compatibility floor must verify both authentic registry locks")
    if any(re.search(r"(?m)^\s*exit\s+1\s*$", _run(step)) for step in _steps(floor)):
        _fail(name, "compatibility floor may not retain the intentional failure")

    if name == "publish-rust-crate.yml":
        expected = {
            "expected_version": "0.7.0",
            "expected_core_version": "0.8.0",
            "expected_python_version": "0.7.0",
        }
        for input_name, default in expected.items():
            if _input_default(inputs, input_name, name) != default:
                _fail(name, f"{input_name} default must be {default}")
    elif name == "publish.yml":
        expected = {
            "expected_version": "0.7.0",
            "expected_rust_version": "0.7.0",
            "expected_adam_core_version": "0.8.0",
            "expected_adam_core_rust_version": "0.8.0",
        }
        for input_name, default in expected.items():
            if _input_default(inputs, input_name, name) != default:
                _fail(name, f"{input_name} default must be {default}")
        confirmation = _run(
            _step_by_name(
                jobs["collect-tested-artifacts"],
                "Verify immutable release tag and irreversible-action confirmation",
            )
        ).strip()
        if confirmation != PYPI_REF_CONFIRMATION_SCRIPT:
            _fail(
                name,
                "non-dry-run PyPI publication must be tag-only and release branches dry-run-only",
            )
        commands = "\n".join(
            _run(step) for job in jobs.values() for step in _steps(job)
        )
        for marker in ("adam-core>=0.8.0,<0.9.0", '"adam-core": "<0.9.0,>=0.8.0"'):
            if marker not in commands:
                _fail(name, f"publication Core range is missing {marker!r}")
    elif name == "release-candidate-wheel-matrix.yml":
        if workflow.get("env") != {
            "PIP_ONLY_BINARY": ":all:",
            "MATURIN_PEP517_ARGS": "--locked",
            "ADAM_CORE_RELEASE_VERSION": "0.8.0",
            "ADAM_CORE_ACCEPTANCE_REF": CORE_PUBLIC_REF,
            "ADAM_ASSIST_RELEASE_VERSION": "0.7.0",
        }:
            _fail(name, "wheel release identities drifted")
        commands = "\n".join(
            _run(step) for job in jobs.values() for step in _steps(job)
        )
        if "adam-core>=0.8.0,<0.9.0" not in commands:
            _fail(name, "wheel acceptance must require the Core 0.8 line")
        matrix = jobs["artifact-acceptance"].get("strategy", {}).get("matrix", {})
        platforms = matrix.get("platform", [])
        macos_arm64 = [
            platform
            for platform in platforms
            if isinstance(platform, Mapping) and platform.get("name") == "macos-arm64"
        ]
        if matrix.get("python-version") != ["3.11", "3.12", "3.13", "3.14"]:
            _fail(name, "wheel release matrix must cover CPython 3.11-3.14")
        if len(macos_arm64) != 1 or macos_arm64[0].get("runner") != "macos-15":
            _fail(name, "macOS arm64 wheels require the supported macos-15 runner")
        benchmark = jobs["full-current-benchmark"]
        if benchmark.get("needs") != "artifact-acceptance":
            _fail(name, "benchmark diagnostic must depend on artifact acceptance")
        if benchmark.get("runs-on") != "macos-15":
            _fail(name, "benchmark diagnostic requires the supported macos-15 runner")
    elif name == "rust-crate-release-candidate.yml":
        if workflow.get("env") != {
            "RUST_RELEASE_VERSION": "0.7.0",
            "CORE_RUST_VERSION": "0.8.0",
            "RELEASE_CHANNEL": "stable",
        }:
            _fail(name, "Rust release identities drifted")
        for job_id in ("package-and-test", "latest-stable-compatibility"):
            setup = _step_by_name(jobs[job_id], "Set up supported Python")
            if setup.get("uses") != "actions/setup-python@v5" or setup.get("with") != {
                "python-version": "3.14"
            }:
                _fail(
                    f"{name}:{job_id}",
                    "Python-feature Rust checks require exact CPython 3.14",
                )


def _verify_publication_wheel_inventory(
    workflows: Mapping[str, Mapping[str, Any]],
) -> None:
    release = workflows["release-candidate-wheel-matrix.yml"]
    matrix = (
        release["jobs"]["artifact-acceptance"].get("strategy", {}).get("matrix", {})
    )
    python_versions = matrix.get("python-version", [])
    platforms = matrix.get("platform", [])
    if not isinstance(python_versions, list) or not isinstance(platforms, list):
        _fail("release-candidate-wheel-matrix.yml", "wheel matrix must use lists")
    expected_count = len(python_versions) * len(platforms)
    expected_python = {
        f"cp{str(version).replace('.', '')}" for version in python_versions
    }

    publish = workflows["publish.yml"]
    collect = _run(
        _step_by_name(
            publish["jobs"]["collect-tested-artifacts"],
            "Aggregate and verify adam-assist wheels without rebuilding",
        )
    )
    aggregate = re.search(r"test \"\$\(find dist [^\n]+\)\" = (\d+)", collect)
    if aggregate is None or int(aggregate.group(1)) != expected_count:
        _fail(
            "publish.yml:collect-tested-artifacts",
            f"aggregate wheel count must equal acceptance matrix size {expected_count}",
        )
    python_literal = re.search(r"expected_python\s*=\s*\{([^}]*)\}", collect)
    actual_python = (
        set(re.findall(r'"(cp\d+)"', python_literal.group(1)))
        if python_literal is not None
        else set()
    )
    if actual_python != expected_python:
        _fail(
            "publish.yml:collect-tested-artifacts",
            f"publication Python tags {actual_python} != {expected_python}",
        )

    publisher = publish["jobs"]["publish-pypi"]
    publication_counts = [
        int(match)
        for step in _steps(publisher)
        for match in re.findall(r"--expected-count\s+(\d+)", _run(step))
    ]
    if publication_counts != [expected_count, expected_count]:
        _fail(
            "publish.yml:publish-pypi",
            "every publication expected count must equal the acceptance matrix "
            f"size {expected_count}; got {publication_counts}",
        )


def verify_workflow_documents(workflows: Mapping[str, Mapping[str, Any]]) -> None:
    if set(workflows) != set(WORKFLOW_INVENTORY):
        _fail("workflows", f"workflow inventory must be {WORKFLOW_INVENTORY!r}")
    workflow = workflows[NORMAL]
    for value in _walk(workflow):
        if isinstance(value, str):
            lowered = value.lower()
            if SECRET_IDENTIFIER.search(value):
                _fail(NORMAL, "any active workflow use of secrets is forbidden")
            if REGISTRY_CREDENTIAL.search(value):
                _fail(NORMAL, "registry credential/token references are forbidden")
            if any(helper in lowered for helper in INERT_RELEASE_HELPERS):
                _fail(NORMAL, "release helpers are forbidden in active CI")
        if isinstance(value, dict) and (
            "id-token" in value or any(child == "write" for child in value.values())
        ):
            _fail(NORMAL, "write/id-token permissions are forbidden in active CI")
    _verify_normal(workflow)
    for name in RELEASE_WORKFLOWS:
        _verify_gated_release_workflow(name, workflows[name])
    _verify_publication_wheel_inventory(workflows)


def verify(root: Path = ROOT) -> None:
    contract = json.loads(
        (root / "migration" / "source_pair_contract.json").read_text()
    )
    if contract.get("registry_state") != "core-0.8.0-public-and-verified":
        _fail(
            "source_pair_contract",
            "workflow policy requires the verified public Core release",
        )
    if contract.get("release_automation") != RELEASE_AUTOMATION_STATE:
        _fail(
            "source_pair_contract",
            "release automation must be version-aligned and gated by release readiness",
        )
    if contract.get("workflow_sha256") != WORKFLOW_SHA256:
        _fail("source_pair_contract", "canonical workflow digest drifted")
    workflow_path = root / ".github" / "workflows" / NORMAL
    actual_digest = hashlib.sha256(workflow_path.read_bytes()).hexdigest()
    if actual_digest != WORKFLOW_SHA256:
        _fail(NORMAL, "workflow content changed; update policy deliberately")
    verify_workflow_documents(load_workflows(root))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    verify(args.root.resolve())
    print("verified active CI and gated release-ready workflow policy")


if __name__ == "__main__":
    main()
