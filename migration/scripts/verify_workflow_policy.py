#!/usr/bin/env python3
"""Fail-closed structural policy for every GitHub Actions workflow.

The verifier intentionally accepts a small set of exact release shapes. Adding a
workflow, action, condition, build command, or publication path requires an
explicit policy update and mutation coverage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_DIR = ROOT / ".github" / "workflows"
CONTRACT = ROOT / "migration" / "source_pair_contract.json"
NORMAL = "pip-build-lint-test-coverage.yml"
RELEASE_WORKFLOWS = {
    "release-candidate-wheel-matrix.yml",
    "rust-crate-release-candidate.yml",
    "publish.yml",
    "publish-rust-crate.yml",
}
GATE = "compatibility-floor"
LOCK_SCRIPT = "migration/scripts/verify_registry_locks.py"
CONTRACT_SCRIPT = "migration/scripts/verify_source_pair_contract.py"
FAILURE_MESSAGE = (
    'echo "Core 0.7.0 is not yet public; registry locks/checksums are unavailable" '
    ">&2"
)
STATUS_FUNCTION = re.compile(
    r"\b(?:always|cancelled|failure|success)\s*\(", re.IGNORECASE
)
CONTROL_OPERATOR = re.compile(r"(?:&&|\|\||(?<![<>]);|(?<!\|)\|(?!\|))")
ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
PROVENANCE_HASHES = {
    ("publish.yml", "collect-tested-artifacts", "Verify acceptance provenance"): (
        "a3117709b81c31e4d13f0e26c4b58c63d5ff724077dabb573a97e4b827310b36"
    ),
    (
        "publish-rust-crate.yml",
        "publish",
        "Verify candidate-run provenance and success",
    ): "eb8676401eeba56cdb68ad54ea0e69f7a9abd97e03571614465a0927042f4737",
}


class WorkflowPolicyError(ValueError):
    """Raised when a workflow can bypass the release policy."""


@dataclass(frozen=True)
class ShellCommand:
    tokens: tuple[str, ...]
    operator_before: str | None = None

    @property
    def executable(self) -> tuple[str, ...]:
        tokens = list(self.tokens)
        while tokens and (ASSIGNMENT.match(tokens[0]) or tokens[0] in {"(", ")"}):
            tokens.pop(0)
        if tokens and tokens[0] == "env":
            tokens.pop(0)
            while tokens and (tokens[0].startswith("-") or ASSIGNMENT.match(tokens[0])):
                tokens.pop(0)
        if tokens and tokens[0] in {"command", "exec", "sudo"}:
            tokens.pop(0)
        if tokens:
            tokens[0] = Path(tokens[0].lstrip("(")).name
            tokens[-1] = tokens[-1].rstrip(")")
        return tuple(tokens)


def _fail(location: str, message: str) -> None:
    raise WorkflowPolicyError(f"{location}: {message}")


def _on(workflow: Mapping[str | bool, Any]) -> Any:
    return workflow.get("on", workflow.get(True))


def load_workflows(root: Path = ROOT) -> dict[str, dict[str, Any]]:
    documents: dict[str, dict[str, Any]] = {}
    paths = sorted((root / ".github" / "workflows").glob("*.yml"))
    paths += sorted((root / ".github" / "workflows").glob("*.yaml"))
    for path in paths:
        loaded = yaml.safe_load(path.read_text())
        if not isinstance(loaded, dict) or not isinstance(loaded.get("jobs"), dict):
            _fail(path.name, "workflow must contain a jobs mapping")
        documents[path.name] = loaded
    return documents


def _needs(job: Mapping[str, Any]) -> set[str]:
    value = job.get("needs", [])
    if isinstance(value, str):
        return {value}
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return set(value)
    if value in (None, []):
        return set()
    raise WorkflowPolicyError(f"invalid needs value: {value!r}")


def _ancestors(jobs: Mapping[str, Mapping[str, Any]], job_id: str) -> set[str]:
    found: set[str] = set()
    pending = list(_needs(jobs[job_id]))
    while pending:
        dependency = pending.pop()
        if dependency in found:
            continue
        if dependency not in jobs:
            _fail(job_id, f"unknown dependency {dependency!r}")
        found.add(dependency)
        pending.extend(_needs(jobs[dependency]))
    return found


def _steps(job: Mapping[str, Any]) -> list[dict[str, Any]]:
    value = job.get("steps", [])
    if not isinstance(value, list) or not all(isinstance(step, dict) for step in value):
        raise WorkflowPolicyError("job steps must be mappings")
    return value


def _run(step: Mapping[str, Any]) -> str:
    value = step.get("run", "")
    return value if isinstance(value, str) else ""


def shell_commands(run: str) -> list[ShellCommand]:
    """Conservatively split shell text on control operators and newlines."""
    commands: list[ShellCommand] = []
    logical = run.replace("\\\n", " ")
    for raw_line in logical.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        lexer = shlex.shlex(line, posix=True, punctuation_chars=";&|")
        lexer.whitespace_split = True
        lexer.commenters = "#"
        try:
            tokens = list(lexer)
        except ValueError:
            # Heredoc bodies and intentionally complex validation snippets are
            # not security gates. Publication/build markers are also scanned
            # from their individual parseable lines.
            continue
        segment: list[str] = []
        preceding: str | None = None
        for token in tokens:
            if token and set(token) <= {";", "&", "|"}:
                if segment:
                    commands.append(ShellCommand(tuple(segment), preceding))
                    segment = []
                preceding = token
            else:
                segment.append(token)
        if segment:
            commands.append(ShellCommand(tuple(segment), preceding))
    return commands


def _python_script(command: ShellCommand, script: str) -> tuple[str, ...] | None:
    tokens = command.executable
    if len(tokens) >= 2 and tokens[0] in {"python", "python3"} and tokens[1] == script:
        return tokens
    return None


def _forbid_continue(location: str, item: Mapping[str, Any]) -> None:
    if "continue-on-error" in item:
        _fail(location, "continue-on-error is forbidden")


def _check_condition(location: str, value: Any, allowed: set[str]) -> None:
    if value is None:
        return
    condition = str(value).strip()
    if STATUS_FUNCTION.search(condition):
        _fail(location, f"status-function condition is forbidden: {condition!r}")
    if condition not in allowed:
        _fail(location, f"condition is not exactly allowlisted: {condition!r}")


def _registry_step(
    workflow_name: str,
    job_id: str,
    index: int,
    step: Mapping[str, Any],
    required_flags: set[str] | None = None,
) -> bool:
    commands = shell_commands(_run(step))
    matches = [
        tokens
        for command in commands
        if (tokens := _python_script(command, LOCK_SCRIPT)) is not None
    ]
    if not matches:
        return False
    location = f"{workflow_name}:{job_id}:step[{index}]"
    _forbid_continue(location, step)
    if "if" in step:
        _fail(location, "registry verification must be unconditional")
    run = _run(step)
    if CONTROL_OPERATOR.search(run) or "|| true" in run:
        _fail(location, "registry verification cannot use command chaining")
    if len(matches) != 1:
        _fail(location, "registry verifier must execute exactly once")
    tokens = set(matches[0][2:])
    required = required_flags or set()
    if "--core-version" not in tokens or not required <= tokens:
        _fail(location, f"registry verifier is missing flags {sorted(required)!r}")
    for command in commands:
        executable = command.executable
        if executable and executable[0] in {"echo", "true", "false", "exit"}:
            _fail(location, "registry verification step contains a neutralizer")
    return True


def _verify_gate(workflow_name: str, jobs: Mapping[str, Mapping[str, Any]]) -> None:
    if GATE not in jobs:
        _fail(workflow_name, "missing transitional compatibility gate")
    gate = jobs[GATE]
    location = f"{workflow_name}:{GATE}"
    _forbid_continue(location, gate)
    if "if" in gate:
        _fail(location, "gate job must be unconditional")
    steps = _steps(gate)
    if len(steps) != 3 or steps[0].get("uses") != "actions/checkout@v4":
        _fail(location, "gate must have the exact checkout/contract/failure shape")
    contract_step, failure_step = steps[1], steps[2]
    for index, step in ((1, contract_step), (2, failure_step)):
        step_location = f"{location}:step[{index}]"
        _forbid_continue(step_location, step)
        if "if" in step:
            _fail(step_location, "gate steps must be unconditional")
    if _run(contract_step).strip() != f"python {CONTRACT_SCRIPT}":
        _fail(location, "gate must execute the source-pair contract exactly")
    lines = [line.strip() for line in _run(failure_step).splitlines() if line.strip()]
    if lines != [FAILURE_MESSAGE, "exit 1"]:
        _fail(location, "gate must end with the exact unconditional failure body")


def _release_allowed_if(workflow: str, job: str, step: Mapping[str, Any]) -> set[str]:
    name = step.get("name")
    if workflow == "release-candidate-wheel-matrix.yml":
        if name == "Build adam-assist manylinux wheel once":
            return {"matrix.platform.family == 'manylinux'"}
        if name == "Build adam-assist native wheel once":
            return {"matrix.platform.family == 'native'"}
    if (
        workflow == "publish.yml"
        and job == "publish-pypi"
        and name == "Upload only missing accepted wheels"
    ):
        return {"steps.publication-state.outputs.missing_count != '0'"}
    return set()


def _release_action_allowed(workflow: str, job: str, step: Mapping[str, Any]) -> bool:
    uses = str(step.get("uses", ""))
    basic = (
        "actions/checkout@v4",
        "actions/setup-python@v5",
        "actions/download-artifact@v4",
        "actions/upload-artifact@v4",
        "dtolnay/rust-toolchain@1.87.0",
        "dtolnay/rust-toolchain@stable",
    )
    if uses in basic:
        return True
    if uses == "PyO3/maturin-action@v1":
        return (
            workflow == "release-candidate-wheel-matrix.yml"
            and job == "artifact-acceptance"
            and step.get("with", {}).get("command") == "build"
        )
    if uses == "pypa/gh-action-pypi-publish@release/v1":
        return workflow == "publish.yml" and job == "publish-pypi"
    if uses == "rust-lang/crates-io-auth-action@v1":
        return workflow == "publish-rust-crate.yml" and job == "publish"
    return False


def _publication_kind(step: Mapping[str, Any]) -> str | None:
    uses = str(step.get("uses", "")).lower()
    command = str(step.get("with", {}).get("command", "")).lower()
    if "pypa/gh-action-pypi-publish" in uses:
        return "pypi-action"
    if "pyo3/maturin-action" in uses and command in {"upload", "publish"}:
        return "maturin-action-publication"
    release_actions = (
        "softprops/action-gh-release",
        "actions/upload-release-asset",
        "ncipollo/release-action",
        "svenstaro/upload-release-action",
    )
    if any(marker in uses for marker in release_actions):
        return "github-release-action"
    for shell_command in shell_commands(_run(step)):
        tokens = shell_command.executable
        if not tokens:
            continue
        if tokens[:2] in {("cargo", "publish"), ("uv", "publish"), ("pdm", "publish")}:
            return f"{tokens[0]}-publication"
        if tokens[:2] in {("maturin", "publish"), ("maturin", "upload")}:
            return "maturin-publication"
        if tokens[:2] == ("twine", "upload"):
            return "twine-publication"
        if (
            len(tokens) >= 4
            and tokens[0] in {"python", "python3"}
            and tokens[1:4] == ("-m", "twine", "upload")
        ):
            return "twine-publication"
        if (
            tokens[:2] == ("gh", "release")
            and len(tokens) >= 3
            and tokens[2] in {"upload", "create"}
        ):
            return "github-release-command"
        if tokens[0] in {"bash", "sh", "zsh"} and "-c" in tokens:
            command_index = tokens.index("-c") + 1
            if command_index < len(tokens):
                nested = {"run": tokens[command_index]}
                if nested_kind := _publication_kind(nested):
                    return nested_kind
        if tokens[0] in {"curl", "wget"}:
            upload_flags = {
                "-t",
                "--upload-file",
                "--data",
                "--data-binary",
                "--form",
                "-f",
            }
            lower = {token.lower() for token in tokens[1:]}
            http = any(
                token.lower().startswith(("http://", "https://")) for token in tokens
            )
            method_upload = any(
                token.upper() in {"PUT", "POST", "PATCH"} for token in tokens
            )
            file_upload = bool(upload_flags & lower) or any(
                token.startswith(
                    (
                        "--post-file=",
                        "--body-file=",
                        "--upload-file=",
                        "--data=",
                        "--data-binary=",
                        "--form=",
                        "-t",
                    )
                )
                for token in lower
            )
            method_upload |= any(
                token.startswith("--method=")
                and token.removeprefix("--method=").upper() in {"PUT", "POST", "PATCH"}
                for token in lower
            )
            if file_upload or method_upload:
                return "http-upload"
            if http and any("upload" in token for token in lower):
                return "http-upload"
        publish = _python_script(
            shell_command, "migration/scripts/publish_crate_archive.py"
        )
        if publish is not None and "--execute" in publish:
            return "crate-archive-publication"
    return None


def _verify_publication_path(
    workflows: Mapping[str, Mapping[str, Any]],
    workflow_name: str,
    job_id: str,
    step_index: int,
    kind: str,
) -> None:
    workflow = workflows[workflow_name]
    jobs = workflow["jobs"]
    job = jobs[job_id]
    allowed = (
        workflow_name == "publish.yml"
        and job_id == "publish-pypi"
        and kind == "pypi-action"
    ) or (
        workflow_name == "publish-rust-crate.yml"
        and job_id == "publish"
        and kind == "crate-archive-publication"
    )
    location = f"{workflow_name}:{job_id}:step[{step_index}]"
    if not allowed:
        _fail(location, f"publication path {kind!r} is not allowlisted")
    if GATE not in _ancestors(jobs, job_id):
        _fail(location, "publication does not depend on compatibility gate")
    ancestors = _ancestors(jobs, job_id)
    protected = any(
        any(
            _registry_step(workflow_name, ancestor, i, step)
            for i, step in enumerate(_steps(jobs[ancestor]))
        )
        for ancestor in ancestors
    ) or any(
        _registry_step(workflow_name, job_id, i, step)
        for i, step in enumerate(_steps(job)[:step_index])
    )
    if not protected:
        _fail(location, "publication does not depend on registry verification")
    if "pypi" in kind:
        expected_environment = {
            "name": "${{ inputs.release_channel == 'stable' && 'pypi' || 'pypi-preview' }}",
            "url": "https://pypi.org/p/adam-assist",
        }
        permissions = job.get("permissions", {})
        if (
            job.get("environment") != expected_environment
            or permissions.get("id-token") != "write"
            or permissions.get("contents") != "read"
        ):
            _fail(
                location, "PyPI publication lacks exact trusted-publishing environment"
            )
        collect = jobs.get("collect-tested-artifacts", {})
        provenance = next(
            (
                step
                for step in _steps(collect)
                if step.get("name") == "Verify acceptance provenance"
            ),
            None,
        )
        expected_hash = PROVENANCE_HASHES[
            (
                "publish.yml",
                "collect-tested-artifacts",
                "Verify acceptance provenance",
            )
        ]
        if (
            provenance is None
            or hashlib.sha256(_run(provenance).encode()).hexdigest() != expected_hash
        ):
            _fail(location, "PyPI publication lacks exact tested-candidate provenance")
    else:
        prior = _steps(job)[:step_index]
        provenance = next(
            (
                step
                for step in prior
                if step.get("name") == "Verify candidate-run provenance and success"
            ),
            None,
        )
        expected_hash = PROVENANCE_HASHES[
            (
                "publish-rust-crate.yml",
                "publish",
                "Verify candidate-run provenance and success",
            )
        ]
        if (
            provenance is None
            or hashlib.sha256(_run(provenance).encode()).hexdigest() != expected_hash
        ):
            _fail(location, "crate publication lacks exact tested-candidate provenance")
        if not any(
            step.get("uses") == "rust-lang/crates-io-auth-action@v1" for step in prior
        ):
            _fail(location, "crate publication lacks trusted-publishing gate")
        expected_environment = (
            "${{ inputs.release_channel == 'stable' && 'crates-io' || "
            "'crates-io-preview' }}"
        )
        permissions = workflow.get("permissions", {})
        dispatch = _on(workflow).get("workflow_dispatch") or {}
        authentication = dispatch.get("inputs", {}).get("authentication", {})
        if (
            job.get("environment") != expected_environment
            or permissions.get("id-token") != "write"
            or permissions.get("contents") != "read"
            or authentication.get("default") != "trusted-publishing"
            or authentication.get("options") != ["trusted-publishing"]
        ):
            _fail(
                location, "crate publication lacks exact trusted-publishing environment"
            )


def _verify_build_commands(workflow: str, job_id: str, step: Mapping[str, Any]) -> None:
    location = f"{workflow}:{job_id}:{step.get('name', '<run>')}"
    run = _run(step)
    if re.search(r"\brm\b[^\n]*\bCargo\.lock\b", run):
        _fail(location, "release workflows must not delete Cargo.lock")
    for command in shell_commands(run):
        tokens = command.executable
        if not tokens:
            continue
        if tokens[:2] in {("cargo", "update"), ("cargo", "generate-lockfile")}:
            _fail(location, "release workflows must not mutate the Cargo lock")
        if (
            tokens[0] == "cargo"
            and len(tokens) >= 2
            and tokens[1] in {"build", "package", "publish"}
            and "--locked" not in tokens
        ):
            _fail(location, f"cargo {tokens[1]} requires --locked")
        if (
            tokens[0] == "maturin"
            and len(tokens) >= 2
            and tokens[1] in {"build", "publish", "upload"}
            and "--locked" not in tokens
        ):
            _fail(location, f"maturin {tokens[1]} requires --locked")
    repository_locked_steps = {
        (
            "rust-crate-release-candidate.yml",
            "package-and-test",
            "Test the public Rust surface at the MSRV",
        ),
        (
            "rust-crate-release-candidate.yml",
            "package-and-test",
            "Package and verify the crate once",
        ),
        (
            "rust-crate-release-candidate.yml",
            "latest-stable-compatibility",
            "Check and test supported feature configurations",
        ),
        (
            "release-candidate-wheel-matrix.yml",
            "full-current-benchmark",
            "Run direct Rust frozen-fixture and kernel integrations",
        ),
    }
    if (workflow, job_id, step.get("name")) in repository_locked_steps:
        for command in shell_commands(run):
            tokens = command.executable
            if (
                tokens
                and tokens[0] == "cargo"
                and tokens[1:2] != ("fmt",)
                and "--locked" not in tokens
            ):
                _fail(location, "repository Cargo command requires --locked")


def _sensitive_before_registry(step: Mapping[str, Any]) -> bool:
    if step.get("uses") == "actions/upload-artifact@v4":
        return True
    if step.get("uses") == "PyO3/maturin-action@v1":
        return True
    if _publication_kind(step) is not None:
        return True
    for command in shell_commands(_run(step)):
        tokens = command.executable
        if tokens and tokens[0] in {"cargo", "maturin"}:
            return True
    return False


def _verify_release_versions(name: str, workflow: Mapping[str, Any]) -> None:
    env = workflow.get("env", {})
    dispatch = _on(workflow).get("workflow_dispatch") or {}
    inputs = dispatch.get("inputs", {}) if isinstance(dispatch, dict) else {}
    if name == "release-candidate-wheel-matrix.yml":
        expected = {
            "ADAM_CORE_RELEASE_VERSION": "0.7.0",
            "ADAM_ASSIST_RELEASE_VERSION": "0.6.0",
        }
        if any(env.get(key) != value for key, value in expected.items()):
            _fail(name, "release versions must remain Core 0.7.0 / Assist 0.6.0")
    elif name == "rust-crate-release-candidate.yml":
        if (
            env.get("CORE_RUST_VERSION") != "0.7.0"
            or env.get("RUST_RELEASE_VERSION") != "0.6.0"
        ):
            _fail(name, "Rust candidate versions must remain Core 0.7.0 / Assist 0.6.0")
    elif name == "publish.yml":
        expected = {
            "expected_version": "0.6.0",
            "expected_rust_version": "0.6.0",
            "expected_adam_core_version": "0.7.0",
            "expected_adam_core_rust_version": "0.7.0",
        }
        if any(
            inputs.get(key, {}).get("default") != value
            for key, value in expected.items()
        ):
            _fail(
                name,
                "Python publication defaults must remain Core 0.7.0 / Assist 0.6.0",
            )
    else:
        expected = {
            "expected_version": "0.6.0",
            "expected_python_version": "0.6.0",
            "expected_core_version": "0.7.0",
        }
        if any(
            inputs.get(key, {}).get("default") != value
            for key, value in expected.items()
        ):
            _fail(
                name, "Rust publication defaults must remain Core 0.7.0 / Assist 0.6.0"
            )


def _verify_release_workflow(name: str, workflow: Mapping[str, Any]) -> None:
    triggers = _on(workflow)
    if not isinstance(triggers, dict) or set(triggers) != {"workflow_dispatch"}:
        _fail(name, "transitional release workflow must be workflow_dispatch-only")
    jobs: Mapping[str, Mapping[str, Any]] = workflow["jobs"]
    _verify_release_versions(name, workflow)
    _verify_gate(name, jobs)
    for job_id, job in jobs.items():
        location = f"{name}:{job_id}"
        _forbid_continue(location, job)
        allowed_job_if = (
            {"inputs.dry_run == false"}
            if (name, job_id) == ("publish.yml", "publish-pypi")
            else set()
        )
        _check_condition(location, job.get("if"), allowed_job_if)
        if job_id != GATE and GATE not in _ancestors(jobs, job_id):
            _fail(location, "job does not depend transitively on compatibility gate")
        for index, step in enumerate(_steps(job)):
            step_location = f"{location}:step[{index}]"
            _forbid_continue(step_location, step)
            _check_condition(
                step_location, step.get("if"), _release_allowed_if(name, job_id, step)
            )
            if "uses" in step and not _release_action_allowed(name, job_id, step):
                _fail(
                    step_location,
                    f"action is not release-allowlisted: {step['uses']!r}",
                )
            run = _run(step)
            if re.search(
                r"(?:^|&&|\|\||;)\s*(?:true|exit\s+0)\b",
                run,
                re.MULTILINE,
            ) or re.search(r"\bif\s+false\s*;", run):
                _fail(step_location, "release step contains a success neutralizer")
            _verify_build_commands(name, job_id, step)
            if step.get("uses") == "PyO3/maturin-action@v1":
                args = str(step.get("with", {}).get("args", ""))
                if "--locked" not in shlex.split(args):
                    _fail(step_location, "maturin action build requires --locked")
    if name == "release-candidate-wheel-matrix.yml":
        env = workflow.get("env", {})
        if env.get("MATURIN_PEP517_ARGS") != "--locked":
            _fail(name, "MATURIN_PEP517_ARGS must be exactly --locked")

    required_locks: dict[str, set[str]]
    if name == "release-candidate-wheel-matrix.yml":
        required_locks = {
            "artifact-acceptance": {"--python-lock", "--rust-lock"},
            "full-current-benchmark": {"--python-lock", "--rust-lock"},
        }
    elif name == "rust-crate-release-candidate.yml":
        required_locks = {"package-and-test": {"--rust-lock"}}
    elif name == "publish.yml":
        required_locks = {"collect-tested-artifacts": {"--python-lock", "--rust-lock"}}
    else:
        required_locks = {"publish": {"--rust-lock"}}
    lock_jobs: set[str] = set()
    for job_id, flags in required_locks.items():
        matches = [
            index
            for index, step in enumerate(_steps(jobs[job_id]))
            if _registry_step(name, job_id, index, step, flags)
        ]
        if len(matches) != 1:
            _fail(
                f"{name}:{job_id}",
                "requires exactly one unconditional registry verification step",
            )
        lock_index = matches[0]
        if any(
            _sensitive_before_registry(step)
            for step in _steps(jobs[job_id])[:lock_index]
        ):
            _fail(
                f"{name}:{job_id}",
                "registry verification must precede every build/package/upload step",
            )
        lock_jobs.add(job_id)
    for job_id in jobs:
        if job_id == GATE:
            continue
        protected = job_id in lock_jobs or bool(_ancestors(jobs, job_id) & lock_jobs)
        if not protected:
            _fail(
                f"{name}:{job_id}",
                "job is not downstream of registry lock verification",
            )


def _normal_allowed_if(job: str, step: Mapping[str, Any]) -> set[str]:
    name = step.get("name")
    if job == "source-pair":
        return {
            "Run non-live wheel tests": {"matrix.python-version != '3.12'"},
            "Run non-live wheel tests with coverage": {
                "matrix.python-version == '3.12'"
            },
            "Lint and typecheck the source": {"matrix.python-version == '3.12'"},
            "Upload coverage artifact": {
                "matrix.python-version == '3.12' && github.event_name == 'pull_request'"
            },
            "Coverage report to coveralls": {"matrix.python-version == '3.12'"},
        }.get(str(name), set())
    if job == "benchmark":
        return {
            "Upload current-only benchmark artifact": {
                "always() && github.event_name == 'pull_request'"
            },
            "Upload benchmark artifact": {"github.event_name == 'pull_request'"},
            "Store main benchmark result": {"github.ref == 'refs/heads/main'"},
            "Compare benchmarks": {"github.event_name == 'pull_request'"},
        }.get(str(name), set())
    return set()


def _verify_normal(workflow: Mapping[str, Any]) -> None:
    name = NORMAL
    triggers = _on(workflow)
    if not isinstance(triggers, dict) or not {"push", "pull_request"} <= set(triggers):
        _fail(name, "normal CI must run on push and pull_request")
    jobs = workflow["jobs"]
    for required in ("source-pair", "rust-quality", "benchmark"):
        if required not in jobs:
            _fail(name, f"missing required normal CI job {required!r}")
        job = jobs[required]
        _forbid_continue(f"{name}:{required}", job)
        if "if" in job:
            _fail(f"{name}:{required}", "normal CI jobs must be unconditional")
        for index, step in enumerate(_steps(job)):
            location = f"{name}:{required}:step[{index}]"
            _forbid_continue(location, step)
            allowed = _normal_allowed_if(required, step)
            value = step.get("if")
            if value is not None and str(value).strip() not in allowed:
                _fail(location, f"normal CI condition is not allowlisted: {value!r}")
            env = step.get("env", {})
            if isinstance(env, dict) and "PYTHONPATH" in env:
                _fail(location, "normal CI must not set PYTHONPATH")
            for command in shell_commands(_run(step)):
                tokens = command.executable
                if tokens[:4] == ("python", "-m", "pip", "install"):
                    if "-e" in tokens or any(
                        token.startswith(".") and "[dev]" in token for token in tokens
                    ):
                        _fail(location, "normal CI must not install repository source")
                    if any("naif-eop-high-prec" in token for token in tokens):
                        _fail(
                            location,
                            "normal CI must not mutate frozen EOP after verification",
                        )
    matrix = (
        jobs["source-pair"].get("strategy", {}).get("matrix", {}).get("python-version")
    )
    if matrix != ["3.11", "3.12", "3.13"]:
        _fail(name, "source-pair matrix must cover exactly Python 3.11-3.13")
    if _needs(jobs["benchmark"]) != {"source-pair", "rust-quality"}:
        _fail(name, "benchmark must require source-pair and rust-quality")
    required_unconditional = {
        "source-pair": {
            "Verify source-pair contract",
            "Install build and test tooling",
            "Build Core and ASSIST wheels from the exact pair",
            "Verify fail-closed workflow policy",
            "Verify installed versions and wheel dependency floor",
            "Verify no generated lock or tracked mutation survived",
        },
        "rust-quality": {"Verify source pair and run Rust gates"},
        "benchmark": {
            "Install the already-tested source-pair wheels",
            "Run benchmarks",
            "Run current public/native benchmark smoke",
        },
    }
    for job_id, names in required_unconditional.items():
        found = {step.get("name"): step for step in _steps(jobs[job_id])}
        for step_name in names:
            if step_name not in found or "if" in found[step_name]:
                _fail(
                    f"{name}:{job_id}",
                    f"required step {step_name!r} must be unconditional",
                )
    policy_step = next(
        step
        for step in _steps(jobs["source-pair"])
        if step.get("name") == "Verify fail-closed workflow policy"
    )
    if (
        _run(policy_step).strip()
        != "python migration/scripts/verify_workflow_policy.py"
    ):
        _fail(name, "normal CI must execute the structural workflow policy exactly")
    install_names = {
        ("source-pair", "Build Core and ASSIST wheels from the exact pair"),
        ("benchmark", "Install the already-tested source-pair wheels"),
    }
    for job_id, step_name in install_names:
        step = next(
            step for step in _steps(jobs[job_id]) if step.get("name") == step_name
        )
        commands = [command.executable for command in shell_commands(_run(step))]
        expected = (
            "python",
            "-m",
            "pip",
            "install",
            "$core_wheel",
            "${assist_wheel}[dev]",
        )
        verifier = (
            "python",
            "migration/scripts/verify_dev_environment.py",
            "--require-wheel",
        )
        if expected not in commands or verifier not in commands:
            _fail(
                f"{name}:{job_id}:{step_name}",
                "must install wheel[dev] and verify provenance",
            )


def verify_workflow_documents(
    workflows: Mapping[str, Mapping[str, Any]], *, awaiting_core: bool = True
) -> None:
    if NORMAL not in workflows:
        _fail("workflows", f"missing {NORMAL}")
    if awaiting_core and not RELEASE_WORKFLOWS <= set(workflows):
        _fail("workflows", "missing transitional release workflow")
    _verify_normal(workflows[NORMAL])
    for name in RELEASE_WORKFLOWS:
        _verify_release_workflow(name, workflows[name])
    for workflow_name, workflow in workflows.items():
        jobs = workflow.get("jobs", {})
        for job_id, job in jobs.items():
            for index, step in enumerate(_steps(job)):
                if kind := _publication_kind(step):
                    _verify_publication_path(
                        workflows, workflow_name, job_id, index, kind
                    )


def verify(root: Path = ROOT) -> None:
    contract = json.loads(
        (root / "migration" / "source_pair_contract.json").read_text()
    )
    awaiting = contract.get("registry_state") == "awaiting-core-0.7.0-publication"
    verify_workflow_documents(load_workflows(root), awaiting_core=awaiting)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    verify(args.root.resolve())
    print("verified fail-closed workflow policy")


if __name__ == "__main__":
    main()
