"""Three-boundary ASSIST non-gravitational propagation benchmark.

Compares frozen updated-upstream ``adam_assist``, the current public Rust-backed
facade, and the genuine Rust-owned ``std::time::Instant`` work unit on identical
same-epoch A1/A2 workloads. These lanes guard the model-aware multi-particle
batching path separately from the gravity-only propagation matrix.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adam_assist import ASSISTPropagator  # noqa: E402
from migration.parity._assist_bench import (  # noqa: E402
    PERFORMANCE_COLUMNS,
    TWO_RUNTIME_COMPARISON_MODE,
    performance_timing_payload,
    time_native_rust,
    time_rust,
)
from migration.parity._assist_oracle import (  # noqa: E402
    LEGACY_ASSIST_VENV_PYTHON,
    LegacyAssistPropagator,
)
from migration.scripts import benchmark_assist_ephemeris as ephemeris  # noqa: E402
from migration.scripts import benchmark_assist_public_semantics as base  # noqa: E402

DEFAULT_OUTPUT = (
    REPO_ROOT
    / "migration"
    / "artifacts"
    / "assist_nongrav_propagation_benchmark_2026-08-12.json"
)


def _workloads() -> list[base.Workload]:
    return [
        base.Workload(
            lane="tiny",
            name="nongrav_5x2",
            description=(
                "A1/A2 non-gravitational propagation: 5 same-epoch orbits by "
                "2 target epochs."
            ),
            orbits=ephemeris._with_nongrav(
                base._base_sun_ecliptic_orbits(5, mixed_epochs=False)
            ),
            times=base._target_times(2, scale="tdb", span_days=1.0),
            chunk_size=5,
        ),
        base.Workload(
            lane="small",
            name="nongrav_40x50",
            description=(
                "A1/A2 non-gravitational propagation: 40 same-epoch orbits by "
                "50 target epochs over 30 days."
            ),
            orbits=ephemeris._with_nongrav(
                base._base_sun_ecliptic_orbits(40, mixed_epochs=False)
            ),
            times=base._target_times(50, scale="tdb", span_days=30.0),
            chunk_size=40,
        ),
        base.Workload(
            lane="large",
            name="nongrav_200x50_1yr",
            description=(
                "A1/A2 non-gravitational propagation: 200 same-epoch orbits by "
                "50 target epochs over one year."
            ),
            orbits=ephemeris._with_nongrav(
                base._base_sun_ecliptic_orbits(200, mixed_epochs=False)
            ),
            times=base._long_horizon_target_times(50, scale="tdb"),
            chunk_size=200,
        ),
    ]


def _benchmark(
    workload: base.Workload,
    legacy: LegacyAssistPropagator,
    current: ASSISTPropagator,
    *,
    repeats: int,
    warmups: int,
) -> dict[str, Any]:
    kwargs = {
        "covariance": False,
        "chunk_size": workload.chunk_size,
        "max_processes": 1,
        "include_nongrav": True,
    }
    legacy_samples = legacy.time_propagate_orbits(
        workload.orbits,
        workload.times,
        repeats=repeats,
        warmups=warmups,
        **kwargs,
    )
    legacy_output = legacy.propagate_orbits(workload.orbits, workload.times, **kwargs)
    current_samples, current_output = time_rust(
        lambda: current.propagate_orbits(workload.orbits, workload.times, **kwargs),
        repeats=repeats,
        warmups=warmups,
    )
    operation, native_samples = time_native_rust(
        current, repeats=repeats, warmups=warmups
    )
    return {
        "lane": workload.lane,
        "name": workload.name,
        "description": workload.description,
        "workload_shape": {
            "n_orbits": len(workload.orbits),
            "n_target_times": len(workload.times),
            "output_rows": len(workload.orbits) * len(workload.times),
            "unique_input_epochs": len(workload.orbits.coordinates.time.unique()),
            "marsden_models": 1,
        },
        "options": kwargs,
        "timing_seconds": performance_timing_payload(
            legacy_samples,
            current_samples,
            native_samples,
            native_operation=operation,
        ),
        "residuals": base._state_residuals(current_output, legacy_output),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.repeats < 3:
        raise ValueError("--repeats must be at least 3 to report p50/p95")
    if not LEGACY_ASSIST_VENV_PYTHON.exists():
        raise FileNotFoundError(LEGACY_ASSIST_VENV_PYTHON)

    legacy = LegacyAssistPropagator()
    current = ASSISTPropagator()
    rows = [
        _benchmark(
            workload,
            legacy,
            current,
            repeats=args.repeats,
            warmups=args.warmups,
        )
        for workload in _workloads()
    ]
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "comparison_mode": TWO_RUNTIME_COMPARISON_MODE,
        "performance_columns": PERFORMANCE_COLUMNS,
        "repeats": args.repeats,
        "warmups": args.warmups,
        "workloads": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
