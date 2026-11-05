import ast
import hashlib
import importlib.util
import json
import re
import tomllib
from importlib.metadata import version
from pathlib import Path

from adam_core.utils.spice import DEFAULT_KERNELS

from adam_assist.version import __version__
from migration.scripts import benchmark_current
from migration.scripts.verify_source_pair_contract import verify as verify_source_pair

ROOT = Path(__file__).resolve().parents[1]


def _project_dependencies() -> list[str]:
    with (ROOT / "pyproject.toml").open("rb") as pyproject_file:
        pyproject = tomllib.load(pyproject_file)
    return list(pyproject["project"]["dependencies"])


def _cargo_manifest() -> dict:
    with (ROOT / "rust" / "adam_assist_rs" / "Cargo.toml").open("rb") as cargo_file:
        return tomllib.load(cargo_file)


def _python_docstrings() -> list[str]:
    docs: list[str] = []
    for path in sorted((ROOT / "src" / "adam_assist").rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        module_doc = ast.get_docstring(tree, clean=False)
        if module_doc:
            docs.append(module_doc)
        for node in ast.walk(tree):
            if not isinstance(
                node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                continue
            docstring = ast.get_docstring(node, clean=False)
            if docstring:
                docs.append(docstring)
    return docs


def _public_rustdoc() -> list[str]:
    docs: list[str] = []
    for path in sorted((ROOT / "rust" / "adam_assist_rs" / "src").rglob("*.rs")):
        source = path.read_text()
        line_docs = []
        for line in source.splitlines():
            stripped = line.lstrip()
            if stripped.startswith(("///", "//!")):
                line_docs.append(stripped[3:].lstrip())
        docs.extend(line_docs)
        docs.extend(re.findall(r"/\*\*.*?\*/|/\*!.*?\*/", source, flags=re.DOTALL))
    return docs


def test_legacy_python_assist_stack_is_not_a_runtime_dependency() -> None:
    names = {
        dependency.split("=")[0].split(">")[0] for dependency in _project_dependencies()
    }
    assert {"assist", "rebound", "ray", "spiceypy"}.isdisjoint(names)


def test_public_rust_crate_metadata_and_dependencies() -> None:
    manifest = _cargo_manifest()
    package = manifest["package"]
    assert package["name"] == "adam_assist"
    assert package["version"] == "0.6.0"
    assert package["rust-version"] == "1.87"
    assert package.get("publish", True) is True
    assert package["license"] == "GPL-3.0"
    assert package["readme"] == "README.md"
    crate_readme = (ROOT / "rust" / "adam_assist_rs" / package["readme"]).read_text()
    assert "ADAM_CORE_KERNEL_DE440" in crate_readme
    assert "ADAM_CORE_KERNEL_SB441_N16" in crate_readme
    assert "ADAM_CORE_RS_ASSIST_PLANETS_PATH" not in crate_readme

    assert manifest["features"]["default"] == ["kernel-data"]
    assert manifest["features"]["kernel-data"] == ["dep:adam_core_rs_kernel_data"]
    dependencies = manifest["dependencies"]
    assert "assist-rs" not in dependencies
    assert dependencies["libassist-sys"] == "=1.2.1"
    assert dependencies["librebound-sys"] == "=4.6.0"
    assert dependencies["sha2"] == "0.10"
    kernel_dependency = {
        "version": "0.7",
        "default-features": False,
    }
    assert dependencies["adam_core_rs_kernel_data"] == {
        **kernel_dependency,
        "optional": True,
    }
    assert manifest["dev-dependencies"]["adam_core"] == "0.7"
    assert manifest["dev-dependencies"]["adam_core_rs_kernel_data"] == (
        kernel_dependency
    )
    assert manifest["dev-dependencies"]["filetime"] == "0.2"
    assert {
        "icu_locale_core",
        "icu_normalizer",
        "icu_properties",
        "icu_provider",
    }.isdisjoint(dependencies)


def test_dev_lint_tool_is_pinned() -> None:
    with (ROOT / "pyproject.toml").open("rb") as pyproject_file:
        pyproject = tomllib.load(pyproject_file)
    dev = pyproject["project"]["optional-dependencies"]["dev"]
    assert "ruff==0.16.4" in dev
    assert "pdm==2.29.2" in dev
    assert "packaging" in dev
    assert "PyYAML" in dev


def test_dependencies_declare_core_070_floor() -> None:
    dependencies = _project_dependencies()
    assert dependencies == [
        "adam-core>=0.7.0,<0.8.0",
        "naif-de440==2020.12.21.1",
        "jpl-small-bodies-de441-n16==2021.3.31.1",
    ]
    with (ROOT / "pyproject.toml").open("rb") as pyproject_file:
        dev_dependencies = tomllib.load(pyproject_file)["project"][
            "optional-dependencies"
        ]["dev"]
    for requirement in (
        "naif-leapseconds==2025.4.22",
        "naif-eop-predict==2024.8.28.1",
        "naif-eop-historical==2024.8.28.1",
        "naif-eop-high-prec==2026.5.9",
        "naif-earth-itrf93==2007.4.3.1",
    ):
        assert requirement in dev_dependencies
    manifest = _cargo_manifest()
    dependencies = manifest["dependencies"]
    assert dependencies["adam_core_rs_coords"] == "0.7"
    assert dependencies["adam_core_rs_spice"] == "0.7"
    assert "patch" not in manifest
    assert not (ROOT / "rust" / "vendor").exists()


def test_prerelease_source_pair_has_no_stale_registry_locks() -> None:
    assert not (ROOT / "pdm.lock").exists()
    assert not (ROOT / "rust" / "adam_assist_rs" / "Cargo.lock").exists()
    contract = verify_source_pair()
    assert contract["core_version"] == "0.7.0"
    assert contract["core_commit"] == "0b49938eb0be04214a8d07b206227d31b248394a"
    assert contract["lock_state"] == "absent-until-core-0.7.0-is-public"
    assert contract["source_pair_status"] == "exact-reviewed-source"
    assert contract["release_automation"] == (
        "present-but-disabled-until-core-0.7.0-is-public-and-locked"
    )


def test_packaged_rust_fixture_matches_frozen_migration_evidence() -> None:
    historical = (
        ROOT
        / "migration"
        / "artifacts"
        / "assist_public_semantics_fixture_2026-05-20.json"
    ).read_bytes()
    packaged = (
        ROOT
        / "rust"
        / "adam_assist_rs"
        / "testdata"
        / "assist_public_semantics_fixture_2026-05-20.json"
    ).read_bytes()
    assert hashlib.sha256(historical).hexdigest() == (
        "31de414652f86a4d23399610e32407bb11e37022e5c28af27e1fbf85dc1aa913"
    )
    assert packaged == historical


def test_core_07_migration_inventory_is_complete() -> None:
    inventory = (ROOT / "MIGRATION.md").read_text()
    for required in (
        "a8ec6d22b422da055730ff7c0e076f51afe8b0ba",
        "00ec21e3888d737becc6b180d1cb7ea3d7524df3",
        "6ef5c5a^{tree}",
        "11 of the 12 replayed commits",
        "spicekit",
        "Propagation, science, and runtime inventory",
        "Core-owned OD surface removal and adapter boundary",
        "Tests, workflows, documentation, and metadata",
        "Current-main test disposition",
        "test_core_refine_preserves_physical_and_custom_nongrav_metadata",
        "test_core_refine_preserves_but_does_not_activate_source_only_nongrav",
        "No Windows or musllinux support",
    ):
        assert required in inventory


def test_all_removed_current_main_tests_have_explicit_dispositions() -> None:
    inventory = (ROOT / "MIGRATION.md").read_text()
    removed = (
        "test_fit_least_squares_evaluated_matches_composed_crossings",
        "test_initial_orbit_determination_fused_batch_and_timing",
        "test_native_od_timing_lanes",
        "test_od_fit_below_min_obs_returns_empty",
        "test_od_fit_matches_frozen_legacy_solution",
        "test_public_fit_least_squares_dispatches_to_native",
        "test_rust_od_recovers_truth",
        "test_vallado_least_squares_matches_frozen_parity",
        "test_every_rust_workflow_uses_the_msrv_toolchain_and_components",
        "test_normal_ci_runs_direct_rust_quality_and_current_only_benchmark",
        "test_release_matrix_builds_once_and_clean_room_accepts_registry_wheels",
        "test_rust_crate_workflows_package_once_and_publish_tested_bytes",
        "test_same_wheel_reports_accept_compatible_core_06_outputs",
        "test_cargo_lock_uses_public_core_registry_artifacts",
        "test_current_benchmark_ci_covers_product_and_core_od_backend_workloads",
        "test_python_lock_uses_public_core_registry_artifacts",
        "test_stable_dependencies_declare_core_06_compatibility",
        "test_partial_constants_are_rejected_for_active_force",
        "test_stable_version_module_is_preserved",
    )
    assert len(removed) == 19
    assert len(set(removed)) == len(removed)
    for test_name in removed:
        assert inventory.count(test_name) == 1, test_name


def test_stale_assist_owned_od_benchmark_is_retired() -> None:
    assert not (ROOT / "migration" / "scripts" / "benchmark_assist_od.py").exists()
    note = (
        ROOT / "migration" / "artifacts" / "assist_od_benchmark_2026-08-12.README.md"
    ).read_text()
    assert "Frozen historical" in note
    normalized = note.replace("**", "").replace("\n", " ")
    assert "not a current benchmark" in normalized


def test_public_extension_is_packaged_inside_adam_assist() -> None:
    with (ROOT / "pyproject.toml").open("rb") as pyproject_file:
        pyproject = tomllib.load(pyproject_file)
    assert pyproject["tool"]["maturin"]["module-name"] == "adam_assist._native"


def test_python_stable_version_matches_cargo_semver() -> None:
    script = ROOT / "migration" / "scripts" / "write_maturin_version.py"
    spec = importlib.util.spec_from_file_location("assist_write_version", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cargo_version = _cargo_manifest()["package"]["version"]
    assert cargo_version == "0.6.0"
    assert module.cargo_version_to_pep440(cargo_version) == __version__ == "0.6.0"


def test_current_benchmark_ci_covers_provider_product_workloads() -> None:
    with (ROOT / "pyproject.toml").open("rb") as pyproject_file:
        command = tomllib.load(pyproject_file)["tool"]["pdm"]["scripts"][
            "benchmark-current-ci"
        ]

    assert "--lanes tiny small large" in command
    assert "--repeats 5" in command
    assert "--require-native" in command
    assert "--quick" not in command

    propagation_count = len(benchmark_current.propagation._workloads())
    nongrav_count = len(benchmark_current.nongrav._workloads())
    ephemeris_workloads = benchmark_current.ephemeris._workloads()
    ephemeris_count = sum(not workload.covariance for workload in ephemeris_workloads)
    covariance_count = sum(workload.covariance for workload in ephemeris_workloads)
    collision_count = 3

    assert (
        propagation_count,
        nongrav_count,
        ephemeris_count,
        covariance_count,
        collision_count,
    ) == (17, 3, 5, 2, 3)
    assert (
        sum(
            (
                propagation_count,
                nongrav_count,
                ephemeris_count,
                covariance_count,
                collision_count,
            )
        )
        == 30
    )


def test_rust_public_semantics_fixture_uses_the_final_legacy_authority() -> None:
    fixture_path = (
        ROOT
        / "migration"
        / "artifacts"
        / "assist_public_semantics_fixture_2026-05-20.json"
    )
    assert hashlib.sha256(fixture_path.read_bytes()).hexdigest() == (
        "31de414652f86a4d23399610e32407bb11e37022e5c28af27e1fbf85dc1aa913"
    )
    assert json.loads(fixture_path.read_text())["packages"] == {
        "adam-assist": "0.3.11.dev12+gcb5bb14",
        "assist": "1.2.3",
        "rebound": "4.6.0",
        "adam-core": "0.5.7.dev39+g757c09fc",
    }


def test_frozen_regressions_are_complete_and_legacy_runtime_free() -> None:
    fixture_dir = ROOT / "tests" / "fixtures"
    metadata = json.loads(
        (fixture_dir / "assist_legacy_regression_v1.json").read_text()
    )
    fixture = fixture_dir / metadata["fixture"]

    assert (
        hashlib.sha256(fixture.read_bytes()).hexdigest() == metadata["fixture_sha256"]
    )
    assert metadata["array_count"] == 53
    current_kernel_provenance = {
        "naif_eop_high_prec_version": version("naif-eop-high-prec"),
        "kernels": [
            {
                "name": Path(kernel).name,
                "size_bytes": Path(kernel).stat().st_size,
                "sha256": hashlib.sha256(Path(kernel).read_bytes()).hexdigest(),
            }
            for kernel in DEFAULT_KERNELS
        ],
    }
    assert metadata["spice_kernel_provenance"] == current_kernel_provenance

    rust_tests = ROOT / "tests" / "rust"
    source = "\n".join(path.read_text() for path in rust_tests.glob("*.py"))
    assert "python_reference_propagator" not in source
    assert "_assist_oracle" not in source
    assert ".legacy-assist-venv" not in source


def test_runtime_docs_and_release_tooling_have_separate_authorities() -> None:
    core_ref = "0b49938eb0be04214a8d07b206227d31b248394a"
    readme = (ROOT / "README.md").read_text()
    crate_readme = (ROOT / "rust" / "adam_assist_rs" / "README.md").read_text()
    migration = (ROOT / "MIGRATION.md").read_text()
    release_tooling = (
        ROOT / "migration" / "scripts" / "RELEASE_TOOLING.md"
    ).read_text()
    normal_ci = (
        ROOT / ".github" / "workflows" / "pip-build-lint-test-coverage.yml"
    ).read_text()

    public_runtime_docs = "\n".join(
        [readme, crate_readme, *_python_docstrings(), *_public_rustdoc()]
    )
    public_runtime_docs_lower = public_runtime_docs.lower()
    for transition_detail in (
        core_ref,
        "adam_assist_releases_enabled",
        "exact-source",
        "source-pair",
        "registry lock",
        "authentic lock",
        "not yet available",
        "not yet on crates.io",
        "release candidate",
        "reviewed core",
        "reviewed source",
    ):
        assert transition_detail not in public_runtime_docs_lower
    forbidden_doc_patterns = {
        "exact source revision": re.compile(
            r"(?i)\b(?=[0-9a-f]{7,40}\b)(?=[0-9a-f]*[a-f])[0-9a-f]{7,40}\b"
        ),
        "Core/package version history": re.compile(
            r"(?i)\b(?:adam[-_ ]?core|adam[-_ ]?assist|core)\b"
            r"[^\n]{0,24}\bv?\d+\.\d+(?:\.\d+)?\b"
        ),
        "transition terminology": re.compile(
            r"(?i)\b(?:exact[- ]source|source[- ]pair|registry locks?|"
            r"release candidates?|authentic locks?)\b"
        ),
        "internal Beads task": re.compile(r"(?i)\bpersonal-[a-z0-9.]+\b"),
    }
    for description, pattern in forbidden_doc_patterns.items():
        assert pattern.search(public_runtime_docs) is None, description

    assert "`pyproject.toml`" in readme
    assert "`rust/adam_assist_rs/Cargo.toml`" in readme
    assert "`migration/scripts/RELEASE_TOOLING.md`" in readme
    assert "`Cargo.toml` is" in crate_readme
    assert "the authority for compatible adam-core crate versions" in crate_readme
    assert "packaged-crate consumers" not in crate_readme
    assert "kernel-backed integration tests" not in crate_readme
    assert "adam-core>=0.7.0,<0.8.0" in _project_dependencies()
    assert _cargo_manifest()["dependencies"]["adam_core_rs_coords"] == "0.7"

    assert core_ref in migration
    assert core_ref in release_tooling
    assert core_ref in normal_ci
    assert "f66517aa" in migration
    assert "src/{propagate.rs, coordinates.rs, orbit.rs, assist_data.rs}" in migration
    assert "ADAM_ASSIST_RELEASES_ENABLED" in release_tooling
    assert "verify_source_pair_contract.py --core-tree" in release_tooling
    for crate in (
        "adam_core",
        "adam_core_rs_coords",
        "adam_core_rs_spice",
        "adam_core_rs_kernel_data",
    ):
        assert f"patch.crates-io.{crate}.path" in release_tooling
    assert "rm -f rust/adam_assist_rs/Cargo.lock" in release_tooling
    assert 'test -z "$(git status --porcelain)"' in release_tooling
    assert "Exact Core source pair" in normal_ci
    assert "exit 1" not in normal_ci
    workflows = sorted(
        path.name for path in (ROOT / ".github" / "workflows").glob("*.yml")
    )
    assert workflows == [
        "pip-build-lint-test-coverage.yml",
        "publish-rust-crate.yml",
        "publish.yml",
        "release-candidate-wheel-matrix.yml",
        "rust-crate-release-candidate.yml",
    ]
