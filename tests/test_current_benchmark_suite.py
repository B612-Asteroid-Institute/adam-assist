import inspect

from migration.scripts import benchmark_assist_ephemeris as ephemeris
from migration.scripts import benchmark_assist_nongrav_propagation as nongrav
from migration.scripts import benchmark_assist_public_semantics as propagation
from migration.scripts import benchmark_current


def test_current_suite_has_no_legacy_runtime_arguments() -> None:
    parser = benchmark_current._build_arg_parser()
    args = parser.parse_args([])
    help_text = parser.format_help().lower()

    assert args.domains == list(benchmark_current.DOMAINS)
    assert args.lanes == list(benchmark_current.LANES)
    source = inspect.getsource(benchmark_current)
    assert "--legacy" not in help_text
    assert "--oracle" not in help_text
    assert "LegacyAssistPropagator" not in source
    assert "LEGACY_ASSIST_VENV_PYTHON" not in source
    assert "performance_timing_payload" not in source


def test_current_suite_can_require_native_timing() -> None:
    args = benchmark_current._build_arg_parser().parse_args(["--require-native"])

    assert args.require_native is True


def test_current_suite_reuses_existing_workload_builders() -> None:
    assert benchmark_current.propagation._workloads is propagation._workloads
    assert benchmark_current.nongrav._workloads is nongrav._workloads
    assert benchmark_current.ephemeris._workloads is ephemeris._workloads


def test_current_suite_forces_single_process_assist_options() -> None:
    source = inspect.getsource(benchmark_current)

    assert '"max_processes": 1' in source
    assert "--max-processes" not in benchmark_current._build_arg_parser().format_help()
