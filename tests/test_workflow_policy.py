from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import pytest

from migration.scripts.verify_workflow_policy import (
    NORMAL,
    WorkflowPolicyError,
    load_workflows,
    verify_workflow_documents,
)

WorkflowMap = dict[str, dict[str, Any]]
Mutator = Callable[[WorkflowMap], None]
BASE = load_workflows()


@dataclass(frozen=True)
class Mutation:
    id: str
    apply: Mutator


def _step(workflows: WorkflowMap, job: str, name: str) -> dict[str, Any]:
    return next(
        step
        for step in workflows[NORMAL]["jobs"][job]["steps"]
        if step.get("name") == name
    )


def _new_workflow(w: WorkflowMap) -> None:
    w["release.yml"] = {
        "on": {"workflow_dispatch": None},
        "permissions": {"contents": "read"},
        "jobs": {
            "release": {
                "runs-on": "ubuntu-latest",
                "steps": [{"run": "echo disabled"}],
            }
        },
    }


def _new_yaml_workflow(w: WorkflowMap) -> None:
    w["hidden.yaml"] = {
        "on": {"push": None},
        "jobs": {"hidden": {"runs-on": "ubuntu-latest", "steps": []}},
    }


def _publisher_action(uses: str, with_: dict[str, Any] | None = None) -> Mutator:
    def mutate(w: WorkflowMap) -> None:
        step: dict[str, Any] = {"uses": uses}
        if with_ is not None:
            step["with"] = with_
        w[NORMAL]["jobs"]["benchmark"]["steps"].append(step)

    return mutate


def _publisher_command(command: str) -> Mutator:
    return lambda w: w[NORMAL]["jobs"]["benchmark"]["steps"].append({"run": command})


def _remove_step(job: str, name: str) -> Mutator:
    def mutate(w: WorkflowMap) -> None:
        steps = w[NORMAL]["jobs"][job]["steps"]
        steps[:] = [step for step in steps if step.get("name") != name]

    return mutate


MUTATIONS = [
    Mutation("new-workflow", _new_workflow),
    Mutation("new-yaml-workflow", _new_yaml_workflow),
    Mutation(
        "top-contents-write",
        lambda w: w[NORMAL].__setitem__("permissions", {"contents": "write"}),
    ),
    Mutation(
        "top-id-token",
        lambda w: w[NORMAL]["permissions"].__setitem__("id-token", "write"),
    ),
    Mutation(
        "job-write-permission",
        lambda w: w[NORMAL]["jobs"]["benchmark"].__setitem__(
            "permissions", {"contents": "write"}
        ),
    ),
    Mutation(
        "pypi-secret",
        lambda w: w[NORMAL]
        .setdefault("env", {})
        .__setitem__("PYPI_TOKEN", "${{ secrets.PYPI_TOKEN }}"),
    ),
    Mutation(
        "cargo-token",
        lambda w: _step(w, "rust-quality", "Verify source pair and run Rust gates")
        .setdefault("env", {})
        .__setitem__("CARGO_REGISTRY_TOKEN", "token"),
    ),
    Mutation(
        "generic-secret",
        lambda w: _step(w, "benchmark", "Run benchmarks")
        .setdefault("env", {})
        .__setitem__("TOKEN", "${{ secrets.GITHUB_TOKEN }}"),
    ),
    Mutation(
        "pypi-action", _publisher_action("pypa/gh-action-pypi-publish@release/v1")
    ),
    Mutation(
        "maturin-upload-action",
        _publisher_action("PyO3/maturin-action@v1", {"command": "upload"}),
    ),
    Mutation(
        "github-release-action", _publisher_action("softprops/action-gh-release@v2")
    ),
    Mutation(
        "crates-auth-action", _publisher_action("rust-lang/crates-io-auth-action@v1")
    ),
    Mutation("twine", _publisher_command("twine upload dist/*")),
    Mutation("python-twine", _publisher_command("python3 -m twine upload dist/*")),
    Mutation(
        "hidden-shell-publisher",
        _publisher_command('bash -c "cd dist && python -m twine upload *"'),
    ),
    Mutation("uv-publish", _publisher_command("uv publish dist/*")),
    Mutation("pdm-publish", _publisher_command("pdm publish")),
    Mutation("cargo-publish", _publisher_command("cd rust && cargo publish")),
    Mutation("maturin-publish", _publisher_command("maturin publish")),
    Mutation("gh-release", _publisher_command("gh release create v0.6.0 dist/*")),
    Mutation(
        "http-upload",
        _publisher_command("curl --upload-file=dist/x https://example.invalid/upload"),
    ),
    Mutation(
        "release-trigger",
        lambda w: w[NORMAL].__setitem__(
            "on",
            {
                **w[NORMAL].get("on", w[NORMAL].get(True)),
                "release": {"types": ["published"]},
            },
        ),
    ),
    Mutation(
        "schedule-trigger",
        lambda w: w[NORMAL].__setitem__(
            "on",
            {
                **w[NORMAL].get("on", w[NORMAL].get(True)),
                "schedule": [{"cron": "0 0 * * *"}],
            },
        ),
    ),
    Mutation("remove-test", _remove_step("source-pair", "Run non-live wheel tests")),
    Mutation(
        "disable-test",
        lambda w: _step(w, "source-pair", "Run non-live wheel tests").__setitem__(
            "if", "false"
        ),
    ),
    Mutation(
        "continue-test",
        lambda w: _step(w, "source-pair", "Run non-live wheel tests").__setitem__(
            "continue-on-error", True
        ),
    ),
    Mutation(
        "disable-source-job",
        lambda w: w[NORMAL]["jobs"]["source-pair"].__setitem__("if", "false"),
    ),
    Mutation(
        "remove-rust-job",
        lambda w: w[NORMAL]["jobs"].pop("rust-quality"),
    ),
    Mutation(
        "disable-benchmark",
        lambda w: _step(w, "benchmark", "Run benchmarks").__setitem__("if", "false"),
    ),
    Mutation(
        "pin-drift",
        lambda w: w[NORMAL]["env"].__setitem__("ADAM_CORE_SOURCE_REF", "0" * 40),
    ),
    Mutation(
        "checkout-drift",
        lambda w: _step(
            w, "source-pair", "Checkout exact reviewed adam-core source pair"
        )["with"].__setitem__("ref", "main"),
    ),
    Mutation(
        "drop-source-contract",
        lambda w: _step(w, "source-pair", "Verify source-pair contract").__setitem__(
            "run", "echo skipped"
        ),
    ),
    Mutation(
        "drop-dev-extra",
        lambda w: _step(
            w, "source-pair", "Build Core and ASSIST wheels from the exact pair"
        ).__setitem__(
            "run",
            _step(
                w,
                "source-pair",
                "Build Core and ASSIST wheels from the exact pair",
            )["run"].replace("${assist_wheel}[dev]", "$assist_wheel"),
        ),
    ),
    Mutation(
        "editable-install",
        lambda w: _step(
            w, "benchmark", "Install the already-tested source-pair wheels"
        ).__setitem__("run", 'python -m pip install -e ".[dev]"'),
    ),
    Mutation(
        "benchmark-auto-push",
        lambda w: w[NORMAL]["jobs"]["benchmark"]["steps"].append(
            {
                "uses": "benchmark-action/github-action-benchmark@v1",
                "with": {"auto-push": True},
            }
        ),
    ),
]


def test_transition_policy_accepts_exact_source_ci() -> None:
    verify_workflow_documents(BASE)


@pytest.mark.parametrize("mutation", MUTATIONS, ids=lambda value: value.id)
def test_transition_policy_rejects_mutations(mutation: Mutation) -> None:
    workflows = deepcopy(BASE)
    mutation.apply(workflows)
    with pytest.raises(WorkflowPolicyError):
        verify_workflow_documents(workflows)


def test_normal_artifact_and_reporting_actions_are_explicitly_allowed() -> None:
    actions = {
        step.get("uses")
        for job in BASE[NORMAL]["jobs"].values()
        for step in job["steps"]
        if "uses" in step
    }
    assert "actions/upload-artifact@v4" in actions
    assert "coverallsapp/github-action@v2.0.0" in actions
    verify_workflow_documents(BASE)


REVIEW_GAP_MUTATIONS = [
    Mutation(
        "secret-no-space",
        lambda w: _step(w, "benchmark", "Run benchmarks")
        .setdefault("env", {})
        .__setitem__("TOKEN", "${{secrets.PUBLISH}}"),
    ),
    Mutation(
        "secret-index",
        lambda w: _step(w, "benchmark", "Run benchmarks")
        .setdefault("env", {})
        .__setitem__("TOKEN", "${{ secrets['PUBLISH'] }}"),
    ),
    Mutation(
        "secret-to-json",
        lambda w: _step(w, "benchmark", "Run benchmarks")
        .setdefault("env", {})
        .__setitem__("TOKEN", "${{ toJSON(secrets) }}"),
    ),
    Mutation(
        "secret-format",
        lambda w: _step(w, "benchmark", "Run benchmarks")
        .setdefault("env", {})
        .__setitem__("TOKEN", "${{ format('{0}', secrets.PUBLISH) }}"),
    ),
    Mutation(
        "uv-publish-token",
        lambda w: _step(w, "benchmark", "Run benchmarks")
        .setdefault("env", {})
        .__setitem__("UV_PUBLISH_TOKEN", "literal"),
    ),
    Mutation(
        "pypi-environment",
        lambda w: w[NORMAL]["jobs"]["benchmark"].__setitem__("environment", "pypi"),
    ),
    Mutation(
        "mapped-release-environment",
        lambda w: w[NORMAL]["jobs"]["benchmark"].__setitem__(
            "environment", {"name": "release", "url": "https://pypi.org"}
        ),
    ),
    Mutation(
        "workflow-defaults",
        lambda w: w[NORMAL].__setitem__("defaults", {"run": {"shell": "false"}}),
    ),
    Mutation(
        "job-defaults",
        lambda w: w[NORMAL]["jobs"]["benchmark"].__setitem__(
            "defaults", {"run": {"shell": "false"}}
        ),
    ),
    Mutation(
        "replace-test-with-true",
        lambda w: _step(w, "source-pair", "Run non-live wheel tests").__setitem__(
            "run", "true"
        ),
    ),
    Mutation(
        "append-or-true",
        lambda w: _step(w, "source-pair", "Run non-live wheel tests").__setitem__(
            "run", "python -m pytest --benchmark-skip -q tests || true"
        ),
    ),
    Mutation(
        "noop-shell",
        lambda w: _step(w, "source-pair", "Run non-live wheel tests").__setitem__(
            "shell", "true {0}"
        ),
    ),
    Mutation(
        "replace-lint-with-true",
        lambda w: _step(w, "source-pair", "Lint and typecheck the source").__setitem__(
            "run", "true"
        ),
    ),
    Mutation(
        "drop-cargo-test",
        lambda w: _step(
            w, "rust-quality", "Verify source pair and run Rust gates"
        ).__setitem__(
            "run",
            _step(w, "rust-quality", "Verify source pair and run Rust gates")[
                "run"
            ].replace("cargo test", "echo skipped"),
        ),
    ),
    Mutation(
        "exclude-python-313",
        lambda w: w[NORMAL]["jobs"]["source-pair"]["strategy"]["matrix"].__setitem__(
            "exclude", [{"python-version": "3.13"}]
        ),
    ),
    Mutation(
        "job-core-ref-override",
        lambda w: w[NORMAL]["jobs"]["rust-quality"].__setitem__(
            "env", {"ADAM_CORE_SOURCE_REF": "main"}
        ),
    ),
    Mutation(
        "drop-rust-core-tree-check",
        lambda w: _step(
            w, "rust-quality", "Verify source pair and run Rust gates"
        ).__setitem__(
            "run",
            _step(w, "rust-quality", "Verify source pair and run Rust gates")[
                "run"
            ].replace("--core-tree .source-pair/adam-core", ""),
        ),
    ),
    Mutation(
        "execute-inert-helper-module",
        _publisher_command(
            "python -m migration.scripts.publish_crate_archive --execute"
        ),
    ),
    Mutation(
        "invoke-inert-pypi-helper",
        _publisher_command("python migration/scripts/prepare_pypi_upload.py"),
    ),
]


@pytest.mark.parametrize("mutation", REVIEW_GAP_MUTATIONS, ids=lambda value: value.id)
def test_transition_policy_rejects_review_gap_mutations(mutation: Mutation) -> None:
    workflows = deepcopy(BASE)
    mutation.apply(workflows)
    with pytest.raises(WorkflowPolicyError):
        verify_workflow_documents(workflows)
