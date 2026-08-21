import hashlib
import importlib.util
import json
import tomllib
from importlib.metadata import version
from pathlib import Path

from adam_core.utils.spice import DEFAULT_KERNELS

from adam_assist.version import __version__
from migration.scripts import benchmark_current

ROOT = Path(__file__).resolve().parents[1]


def _project_dependencies() -> list[str]:
    with (ROOT / "pyproject.toml").open("rb") as pyproject_file:
        pyproject = tomllib.load(pyproject_file)
    return list(pyproject["project"]["dependencies"])


def _cargo_manifest() -> dict:
    with (ROOT / "rust" / "adam_assist_rs" / "Cargo.toml").open("rb") as cargo_file:
        return tomllib.load(cargo_file)


def _cargo_lock_packages() -> dict[str, dict]:
    with (ROOT / "rust" / "adam_assist_rs" / "Cargo.lock").open("rb") as lock_file:
        packages = tomllib.load(lock_file)["package"]
    return {package["name"]: package for package in packages}


def test_legacy_python_assist_stack_is_not_a_runtime_dependency() -> None:
    names = {
        dependency.split("=")[0].split(">")[0] for dependency in _project_dependencies()
    }
    assert {"assist", "rebound", "ray", "spiceypy"}.isdisjoint(names)


def test_canonical_sys_crates_are_pinned_without_assist_rs() -> None:
    dependencies = _cargo_manifest()["dependencies"]
    assert "assist-rs" not in dependencies
    assert dependencies["libassist-sys"] == "=1.2.1"
    assert dependencies["librebound-sys"] == "=4.6.0"


def test_preview_dependencies_are_exact_public_releases() -> None:
    assert "adam-core==0.5.6rc5" in _project_dependencies()
    manifest = _cargo_manifest()
    dependencies = manifest["dependencies"]
    assert dependencies["adam_core_rs_coords"] == "=0.1.0-rc.4"
    assert dependencies["adam_core_rs_spice"] == "=0.1.0-rc.4"
    assert manifest["dev-dependencies"]["adam_core_rs_kernel_data"] == {
        "version": "=0.1.0-rc.4",
        "default-features": False,
    }
    assert not (ROOT / "rust" / "vendor").exists()


def test_lock_matches_exact_published_core_crates() -> None:
    packages = _cargo_lock_packages()
    assert packages["adam_assist_rs"]["version"] == "0.4.0-rc.6"
    for name in (
        "adam_core_rs_autodiff",
        "adam_core_rs_coords",
        "adam_core_rs_kernel_data",
        "adam_core_rs_orbit_determination",
        "adam_core_rs_spice",
    ):
        assert packages[name]["version"] == "0.1.0-rc.4"
        assert packages[name]["source"] == (
            "registry+https://github.com/rust-lang/crates.io-index"
        )


def test_public_extension_is_packaged_inside_adam_assist() -> None:
    with (ROOT / "pyproject.toml").open("rb") as pyproject_file:
        pyproject = tomllib.load(pyproject_file)
    assert pyproject["tool"]["maturin"]["module-name"] == "adam_assist._native"


def test_python_preview_version_matches_cargo_semver() -> None:
    script = ROOT / "migration" / "scripts" / "write_maturin_version.py"
    spec = importlib.util.spec_from_file_location("assist_write_version", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cargo_version = _cargo_manifest()["package"]["version"]
    assert cargo_version == "0.4.0-rc.6"
    assert module.cargo_version_to_pep440(cargo_version) == __version__ == "0.4.0rc6"


def test_current_benchmark_ci_is_the_complete_35_workload_grid() -> None:
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
    od_count = 5

    assert (
        propagation_count,
        nongrav_count,
        ephemeris_count,
        covariance_count,
        collision_count,
        od_count,
    ) == (17, 3, 5, 2, 3, 5)
    assert (
        sum(
            (
                propagation_count,
                nongrav_count,
                ephemeris_count,
                covariance_count,
                collision_count,
                od_count,
            )
        )
        == 35
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
