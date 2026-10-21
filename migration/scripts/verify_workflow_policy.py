#!/usr/bin/env python3
"""Verify the no-release-automation transition policy.

Until Core 0.7 is public and authentic registry locks exist, the repository has
one read-only workflow: exact-source CI. Release automation must be introduced
later in a separately reviewed commit.
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
CORE_REF = "f0f1aec8b3c059f0cc4af054b5ebe8ed7760da19"
CORE_REPOSITORY = "B612-Asteroid-Institute/adam_core"
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
    "verify_registry_locks",
    "verify_release",
)
WORKFLOW_SHA256 = "0c37bce5c4c4a1067ba4098a6311ca3fc5f347096c2cc3425727ee9930d9d95c"
REQUIRED_RUN_SHA256 = {
    (
        "source-pair",
        "Verify source-pair contract",
    ): "e32dc74240bec3c4973eba9ad56592e2ebae1a6b9adaa5e29dc051f753f40b3f",
    (
        "source-pair",
        "Install build and test tooling",
    ): "f086e46169b1d26b49ea59ef0b7865f5467accd5ab79d2e6450151e9f139d01c",
    (
        "source-pair",
        "Build Core and ASSIST wheels from the exact pair",
    ): "7eed6d4891ec0d61287c8d6902d93ee0da75395185a7572cd322cf7affc1224d",
    (
        "source-pair",
        "Verify transitional workflow policy",
    ): "50a51d5358b2c995ee1ad421cfe663df89c6201557ca3775a85c1aedc41638d7",
    (
        "source-pair",
        "Verify installed versions and wheel dependency floor",
    ): "d356139c8cab6fb05be911e92bca7eba3b2e6c83a52b221d1823499a2c76e43b",
    (
        "source-pair",
        "Run non-live wheel tests",
    ): "6155eec545b37540661aa11f070ae5122d8a9ab39ba6c22db1f961c29dc8c5bd",
    (
        "source-pair",
        "Run non-live wheel tests with coverage",
    ): "8d1caa5efe72cf121fb1c9b426bf5f67ba844dc3d6c923f8009687562f497a8a",
    (
        "source-pair",
        "Lint and typecheck the source",
    ): "d88ab56c0c5025ae3cc64775839be5eec21b5c0a153ddacb1be125786dd0e23c",
    (
        "source-pair",
        "Verify no generated lock or tracked mutation survived",
    ): "3d7fc20fb73ddab51a7d4b84729d3c0e9452d380edc4c3776120d9a304567c57",
    (
        "rust-quality",
        "Verify source pair and run Rust gates",
    ): "13e6b3ffc68306b6a97bf38ceb5675e8e8fc15fb5d4199ed5da92be53e3fd9cb",
    (
        "benchmark",
        "Install the already-tested source-pair wheels",
    ): "0fc96830e3b073a75a52a3266ead70725367914f69492736d4dd4d75a2510375",
    (
        "benchmark",
        "Run benchmarks",
    ): "9a477acc4abccc5bd7f3f8f470f3baa2de3a54f6d9adb34dcf736b29cf3a9193",
    (
        "benchmark",
        "Run current public/native benchmark smoke",
    ): "5e6bf6568657875cb383aee5668bd42f59d99d5bb35dfed9b62885282697ab0b",
}


class WorkflowPolicyError(ValueError):
    """Raised when transitional CI could publish or skip required validation."""


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
    if [path.name for path in paths] != [NORMAL]:
        _fail("workflows", f"transitional inventory must be exactly [{NORMAL!r}]")
    document = yaml.safe_load(paths[0].read_text())
    if not isinstance(document, dict) or not isinstance(document.get("jobs"), dict):
        _fail(NORMAL, "workflow must contain a jobs mapping")
    return {NORMAL: document}


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


def _verify_checkout(job: Mapping[str, Any], location: str) -> None:
    step = _step_by_name(job, "Checkout provisional exact adam-core source pair")
    if step.get("uses") != "actions/checkout@v4" or step.get("with") != {
        "repository": CORE_REPOSITORY,
        "ref": "${{ env.ADAM_CORE_SOURCE_REF }}",
        "path": ".source-pair/adam-core",
    }:
        _fail(location, "Core checkout must use the provisional exact source pair")


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
        "source-pair": {"name", "runs-on", "strategy", "steps", "timeout-minutes"},
        "rust-quality": {"name", "runs-on", "steps", "timeout-minutes"},
        "benchmark": {"runs-on", "needs", "steps", "timeout-minutes"},
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
                "bash"
                if (job_id, step.get("name"))
                in {
                    ("source-pair", "Build Core and ASSIST wheels from the exact pair"),
                    ("rust-quality", "Verify source pair and run Rust gates"),
                }
                else None
            )
            if step.get("shell") != expected_shell:
                _fail(
                    f"{NORMAL}:{job_id}:{step.get('name', '<step>')}",
                    "custom or missing shell override",
                )
            env = step.get("env")
            allowed_env = (
                {"GH_TOKEN": "${{ github.token }}"}
                if (job_id, step.get("name")) == ("benchmark", "Run benchmarks")
                else None
            )
            if env is not None and env != allowed_env:
                _fail(
                    f"{NORMAL}:{job_id}:{step.get('name', '<step>')}",
                    "step environment is not allowlisted",
                )

    for (job_id, name), expected in REQUIRED_RUN_SHA256.items():
        actual = hashlib.sha256(
            _run(_step_by_name(jobs[job_id], name)).encode()
        ).hexdigest()
        if actual != expected:
            _fail(f"{NORMAL}:{job_id}:{name}", "required command changed")


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
        "ADAM_CORE_VERSION": "0.7.0",
        "ADAM_CORE_SOURCE_REF": CORE_REF,
    }:
        _fail(NORMAL, "Core version/ref drifted from the provisional source pair")
    jobs = workflow["jobs"]
    if set(jobs) != {"source-pair", "rust-quality", "benchmark"}:
        _fail(NORMAL, "required job inventory is source-pair, rust-quality, benchmark")
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
            run = _run(step)
            if _run_mentions_publication(run):
                _fail(
                    step_location,
                    "package or GitHub-release publication command is forbidden",
                )
    _verify_exact_structure(workflow)
    source = jobs["source-pair"]
    rust = jobs["rust-quality"]
    benchmark = jobs["benchmark"]
    if source.get("strategy") != {
        "fail-fast": False,
        "matrix": {"python-version": ["3.11", "3.12", "3.13"]},
    }:
        _fail(NORMAL, "Python source-pair strategy must remain exactly 3.11-3.13")
    if benchmark.get("needs") != ["source-pair", "rust-quality"]:
        _fail(NORMAL, "benchmark must require source-pair and rust-quality")
    _verify_checkout(source, "source-pair")
    _verify_checkout(rust, "rust-quality")

    required_conditions = {
        "Run non-live wheel tests": "matrix.python-version != '3.12'",
        "Run non-live wheel tests with coverage": "matrix.python-version == '3.12'",
        "Lint and typecheck the source": "matrix.python-version == '3.12'",
        "Upload coverage artifact": "matrix.python-version == '3.12' && github.event_name == 'pull_request'",
        "Coverage report to coveralls": "matrix.python-version == '3.12'",
    }
    for name, condition in required_conditions.items():
        _require_condition(_step_by_name(source, name), condition)
    for name in (
        "Verify source-pair contract",
        "Build Core and ASSIST wheels from the exact pair",
        "Verify transitional workflow policy",
        "Verify installed versions and wheel dependency floor",
        "Verify no generated lock or tracked mutation survived",
    ):
        _require_condition(_step_by_name(source, name), None)
    for name in ("Verify source pair and run Rust gates",):
        _require_condition(_step_by_name(rust, name), None)
    for name in (
        "Install the already-tested source-pair wheels",
        "Run benchmarks",
        "Run current public/native benchmark smoke",
    ):
        _require_condition(_step_by_name(benchmark, name), None)
    _require_condition(
        _step_by_name(benchmark, "Upload current-only benchmark artifact"),
        "always()",
    )
    _require_condition(
        _step_by_name(benchmark, "Upload benchmark artifact"),
        "always()",
    )

    policy = _step_by_name(source, "Verify transitional workflow policy")
    if _run(policy).strip() != "python migration/scripts/verify_workflow_policy.py":
        _fail(NORMAL, "normal CI must execute this policy exactly")
    contract = _run(_step_by_name(source, "Verify source-pair contract"))
    if (
        'test "$(git -C .source-pair/adam-core rev-parse HEAD)" = "$ADAM_CORE_SOURCE_REF"'
        not in contract
        or "verify_source_pair_contract.py" not in contract
        or "--core-tree .source-pair/adam-core" not in contract
    ):
        _fail(NORMAL, "source-pair contract/commit verification is incomplete")
    for job in (source, benchmark):
        combined = "\n".join(_run(step) for step in _steps(job))
        if 'python -m pip install "$core_wheel" "${assist_wheel}[dev]"' not in combined:
            _fail(NORMAL, "wheel-backed jobs must install the Assist dev extra")
        if "verify_dev_environment.py --require-wheel" not in combined:
            _fail(NORMAL, "wheel-backed jobs must verify installed provenance and pins")


def verify_workflow_documents(workflows: Mapping[str, Mapping[str, Any]]) -> None:
    if set(workflows) != {NORMAL}:
        _fail("workflows", f"transitional inventory must be exactly [{NORMAL!r}]")
    workflow = workflows[NORMAL]
    for value in _walk(workflow):
        if isinstance(value, str):
            lowered = value.lower()
            if SECRET_IDENTIFIER.search(value):
                _fail(NORMAL, "any workflow use of secrets is forbidden")
            if REGISTRY_CREDENTIAL.search(value):
                _fail(NORMAL, "registry credential/token references are forbidden")
            if any(helper in lowered for helper in INERT_RELEASE_HELPERS):
                _fail(NORMAL, "inert release helpers are forbidden in workflows")
        if isinstance(value, dict) and (
            "id-token" in value or any(child == "write" for child in value.values())
        ):
            _fail(NORMAL, "write/id-token permissions are forbidden")
    _verify_normal(workflow)


def verify(root: Path = ROOT) -> None:
    contract = json.loads(
        (root / "migration" / "source_pair_contract.json").read_text()
    )
    if contract.get("registry_state") != "awaiting-core-0.7.0-publication":
        _fail(
            "source_pair_contract",
            "workflow policy applies only to registry transition",
        )
    if (
        contract.get("release_automation")
        != "absent-until-core-0.7.0-is-public-and-locked"
    ):
        _fail("source_pair_contract", "release automation must be explicitly absent")
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
    print("verified no-release-automation transition policy")


if __name__ == "__main__":
    main()
