from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import pytest

from migration.scripts.verify_workflow_policy import (
    EXPECTED_RELEASE_JOB_CONDITIONS,
    NORMAL,
    PUBLISH_PYPI_CONDITION,
    READINESS_CONDITION,
    RELEASE_WORKFLOWS,
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
        lambda w: w[NORMAL]["env"].__setitem__("ADAM_CORE_PUBLIC_REF", "0" * 40),
    ),
    Mutation(
        "checkout-drift",
        lambda w: _step(w, "source-pair", "Checkout exact adam-core source")[
            "with"
        ].__setitem__("ref", "main"),
    ),
    Mutation(
        "drop-source-contract",
        lambda w: _step(
            w, "source-pair", "Verify exact source-pair identity"
        ).__setitem__("run", "echo skipped"),
    ),
    Mutation(
        "drop-dev-extra",
        lambda w: _step(
            w, "source-pair", "Build and install exact source-pair wheels"
        ).__setitem__(
            "run",
            _step(
                w,
                "source-pair",
                "Build and install exact source-pair wheels",
            )[
                "run"
            ].replace("${assist_wheel}[dev]", "$assist_wheel"),
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


def test_release_ready_policy_accepts_active_ci_and_gated_release_workflows() -> None:
    verify_workflow_documents(BASE)


def _replace_workflow_string(value: Any, old: str, new: str) -> Any:
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, list):
        return [_replace_workflow_string(item, old, new) for item in value]
    if isinstance(value, dict):
        return {
            key: _replace_workflow_string(item, old, new) for key, item in value.items()
        }
    return value


@pytest.mark.parametrize("name", RELEASE_WORKFLOWS)
def test_release_workflow_inventory_is_required(name: str) -> None:
    workflows = deepcopy(BASE)
    workflows.pop(name)
    with pytest.raises(WorkflowPolicyError, match="inventory"):
        verify_workflow_documents(workflows)


@pytest.mark.parametrize("name", RELEASE_WORKFLOWS)
def test_every_release_job_must_retain_its_readiness_condition(name: str) -> None:
    workflows = deepcopy(BASE)
    job_id, job = next(iter(workflows[name]["jobs"].items()))
    assert job.pop("if") == EXPECTED_RELEASE_JOB_CONDITIONS[name][job_id]
    with pytest.raises(WorkflowPolicyError, match="condition drifted"):
        verify_workflow_documents(workflows)


@pytest.mark.parametrize("name", RELEASE_WORKFLOWS)
def test_release_workflows_reject_automatic_triggers(name: str) -> None:
    workflows = deepcopy(BASE)
    workflow = workflows[name]
    key: Any = True if True in workflow else "on"
    dispatch = workflow[key]["workflow_dispatch"]
    workflow[key] = {"push": None, "workflow_dispatch": dispatch}
    with pytest.raises(WorkflowPolicyError, match="workflow_dispatch-only"):
        verify_workflow_documents(workflows)


@pytest.mark.parametrize("name", RELEASE_WORKFLOWS)
def test_release_workflows_reject_negated_readiness_gate(name: str) -> None:
    workflows = deepcopy(BASE)
    job = next(iter(workflows[name]["jobs"].values()))
    job["if"] = READINESS_CONDITION.replace("==", "!=")
    with pytest.raises(WorkflowPolicyError, match="condition drifted"):
        verify_workflow_documents(workflows)


def test_pypi_publish_retains_dry_run_composition() -> None:
    workflows = deepcopy(BASE)
    job = workflows["publish.yml"]["jobs"]["publish-pypi"]
    assert job["if"] == PUBLISH_PYPI_CONDITION
    job["if"] = READINESS_CONDITION
    with pytest.raises(WorkflowPolicyError, match="condition drifted"):
        verify_workflow_documents(workflows)


def test_release_workflows_reject_secret_insertion() -> None:
    workflows = deepcopy(BASE)
    step = workflows["publish-rust-crate.yml"]["jobs"]["publish"]["steps"][0]
    step["env"] = {"CARGO_TOKEN": "${{ secrets.CARGO_TOKEN }}"}
    with pytest.raises(WorkflowPolicyError, match="may not use repository secrets"):
        verify_workflow_documents(workflows)


def test_release_workflows_reject_partially_stale_versions() -> None:
    workflows = deepcopy(BASE)
    inputs = workflows["publish.yml"][True]["workflow_dispatch"]["inputs"]
    inputs["expected_version"]["default"] = "0.5.2"
    with pytest.raises(WorkflowPolicyError, match="expected_version default"):
        verify_workflow_documents(workflows)


@pytest.mark.parametrize("name", RELEASE_WORKFLOWS)
def test_release_workflows_reject_stale_core_versions(name: str) -> None:
    workflows = deepcopy(BASE)
    workflows[name] = _replace_workflow_string(workflows[name], "0.7.0", "0.6.1")
    with pytest.raises(WorkflowPolicyError):
        verify_workflow_documents(workflows)


@pytest.mark.parametrize("name", RELEASE_WORKFLOWS)
def test_release_workflows_require_public_source_and_registry_floor(name: str) -> None:
    workflows = deepcopy(BASE)
    floor = workflows[name]["jobs"]["compatibility-floor"]
    floor["steps"] = [
        step
        for step in floor["steps"]
        if step.get("name") != "Verify authentic Core registry locks"
    ]
    with pytest.raises(WorkflowPolicyError, match="authentic Core registry locks"):
        verify_workflow_documents(workflows)

    workflows = deepcopy(BASE)
    floor = workflows[name]["jobs"]["compatibility-floor"]
    next(
        step
        for step in floor["steps"]
        if step.get("name") == "Verify exact public-release source-pair state"
    )["run"] = "echo skipped"
    with pytest.raises(WorkflowPolicyError, match="exact public source pair"):
        verify_workflow_documents(workflows)


@pytest.mark.parametrize("mutation", MUTATIONS, ids=lambda value: value.id)
def test_release_ready_policy_rejects_mutations(mutation: Mutation) -> None:
    workflows = deepcopy(BASE)
    mutation.apply(workflows)
    with pytest.raises(WorkflowPolicyError):
        verify_workflow_documents(workflows)


@pytest.mark.parametrize("job", ["package-and-test", "latest-stable-compatibility"])
def test_rust_release_python_feature_jobs_require_cpython_314(job: str) -> None:
    workflow = "rust-crate-release-candidate.yml"

    missing = deepcopy(BASE)
    steps = missing[workflow]["jobs"][job]["steps"]
    steps[:] = [step for step in steps if step.get("name") != "Set up supported Python"]
    with pytest.raises(WorkflowPolicyError):
        verify_workflow_documents(missing)

    unsupported = deepcopy(BASE)
    setup = next(
        step
        for step in unsupported[workflow]["jobs"][job]["steps"]
        if step.get("name") == "Set up supported Python"
    )
    setup["with"]["python-version"] = "3.13"
    with pytest.raises(WorkflowPolicyError):
        verify_workflow_documents(unsupported)


@pytest.mark.parametrize(
    "step_name",
    (
        "Prepare only missing checksum-verified files",
        "Verify the complete public release matches accepted bytes",
    ),
)
def test_publication_expected_counts_follow_acceptance_matrix(step_name: str) -> None:
    workflows = deepcopy(BASE)
    step = next(
        item
        for item in workflows["publish.yml"]["jobs"]["publish-pypi"]["steps"]
        if item.get("name") == step_name
    )
    step["run"] = step["run"].replace("--expected-count 16", "--expected-count 12")
    with pytest.raises(WorkflowPolicyError, match="every publication expected count"):
        verify_workflow_documents(workflows)


def test_publication_aggregation_count_and_tags_follow_acceptance_matrix() -> None:
    aggregate_drift = deepcopy(BASE)
    step = next(
        item
        for item in aggregate_drift["publish.yml"]["jobs"]["collect-tested-artifacts"][
            "steps"
        ]
        if item.get("name")
        == "Aggregate and verify adam-assist wheels without rebuilding"
    )
    step["run"] = step["run"].replace(')" = 16', ')" = 12', 1)
    with pytest.raises(WorkflowPolicyError, match="aggregate wheel count"):
        verify_workflow_documents(aggregate_drift)

    tags_drift = deepcopy(BASE)
    step = next(
        item
        for item in tags_drift["publish.yml"]["jobs"]["collect-tested-artifacts"][
            "steps"
        ]
        if item.get("name")
        == "Aggregate and verify adam-assist wheels without rebuilding"
    )
    step["run"] = step["run"].replace(', "cp314"', "", 1)
    with pytest.raises(WorkflowPolicyError, match="publication Python tags"):
        verify_workflow_documents(tags_drift)

    matrix_drift = deepcopy(BASE)
    matrix_drift["release-candidate-wheel-matrix.yml"]["jobs"]["artifact-acceptance"][
        "strategy"
    ]["matrix"]["platform"].pop()
    with pytest.raises(WorkflowPolicyError, match="acceptance matrix size"):
        verify_workflow_documents(matrix_drift)


def test_wheel_release_requires_supported_macos_arm64_runner() -> None:
    workflows = deepcopy(BASE)
    workflow = workflows["release-candidate-wheel-matrix.yml"]
    platforms = workflow["jobs"]["artifact-acceptance"]["strategy"]["matrix"][
        "platform"
    ]
    macos_arm64 = next(
        platform for platform in platforms if platform["name"] == "macos-arm64"
    )
    macos_arm64["runner"] = "macos-14"
    with pytest.raises(WorkflowPolicyError, match="supported macos-15 runner"):
        verify_workflow_documents(workflows)


def test_wheel_benchmark_is_non_gating_and_artifact_dependent() -> None:
    workflow_name = "release-candidate-wheel-matrix.yml"

    missing_non_gating = deepcopy(BASE)
    del missing_non_gating[workflow_name]["jobs"]["full-current-benchmark"][
        "continue-on-error"
    ]
    with pytest.raises(WorkflowPolicyError, match="diagnostic must be non-gating"):
        verify_workflow_documents(missing_non_gating)

    weakened_artifacts = deepcopy(BASE)
    weakened_artifacts[workflow_name]["jobs"]["artifact-acceptance"][
        "continue-on-error"
    ] = True
    with pytest.raises(WorkflowPolicyError, match="continue-on-error is forbidden"):
        verify_workflow_documents(weakened_artifacts)

    detached = deepcopy(BASE)
    detached[workflow_name]["jobs"]["full-current-benchmark"][
        "needs"
    ] = "compatibility-floor"
    with pytest.raises(WorkflowPolicyError, match="depend on artifact acceptance"):
        verify_workflow_documents(detached)

    unsupported_runner = deepcopy(BASE)
    unsupported_runner[workflow_name]["jobs"]["full-current-benchmark"][
        "runs-on"
    ] = "macos-14"
    with pytest.raises(WorkflowPolicyError, match="supported macos-15 runner"):
        verify_workflow_documents(unsupported_runner)


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
        "remove-merged-core-row",
        lambda w: w[NORMAL]["jobs"]["source-pair"]["strategy"]["matrix"][
            "include"
        ].pop(),
    ),
    Mutation(
        "merged-core-ref-drift",
        lambda w: w[NORMAL]["jobs"]["source-pair"]["strategy"]["matrix"]["include"][
            -1
        ].__setitem__("core-ref", "main"),
    ),
    Mutation(
        "public-core-kind-drift",
        lambda w: w[NORMAL]["jobs"]["source-pair"]["strategy"]["matrix"]["include"][
            0
        ].__setitem__("core-kind", "merged-source"),
    ),
    Mutation(
        "add-version-specific-job",
        lambda w: w[NORMAL]["jobs"].__setitem__(
            "version-specific-source-pair",
            {"runs-on": "ubuntu-latest", "steps": []},
        ),
    ),
    Mutation(
        "job-core-ref-override",
        lambda w: w[NORMAL]["jobs"]["rust-quality"].__setitem__(
            "env", {"ADAM_CORE_PUBLIC_REF": "main"}
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
def test_release_ready_policy_rejects_review_gap_mutations(mutation: Mutation) -> None:
    workflows = deepcopy(BASE)
    mutation.apply(workflows)
    with pytest.raises(WorkflowPolicyError):
        verify_workflow_documents(workflows)
