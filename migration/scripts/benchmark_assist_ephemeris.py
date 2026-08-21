"""Three-boundary ASSIST ephemeris parity and performance matrix.

Times frozen updated-upstream ``adam_assist`` inside the isolated oracle
runtime, the current public Python facade, and the native Rust work unit timed
with ``std::time::Instant``. Workloads cover gravity-only, non-gravitational,
6D covariance, and 9D covariance ephemerides.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from adam_core.coordinates import CoordinateCovariances
from adam_core.observers import Observers
from adam_core.orbits import NonGravitationalParameters, Orbits
from adam_core.time import Timestamp

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adam_assist import ASSISTPropagator  # noqa: E402

if TYPE_CHECKING:
    from migration.parity._assist_oracle import LegacyAssistPropagator
from migration.parity._assist_bench import (  # noqa: E402
    PERFORMANCE_COLUMNS,
    TWO_RUNTIME_COMPARISON_MODE,
    performance_timing_payload,
    time_native_rust,
    time_rust,
)
from migration.scripts import benchmark_assist_public_semantics as base  # noqa: E402

DEFAULT_OUTPUT = (
    REPO_ROOT / "migration" / "artifacts" / "assist_ephemeris_benchmark_2026-08-12.json"
)


@dataclass(frozen=True)
class Workload:
    lane: str
    name: str
    description: str
    orbits: Orbits
    observer_times: Timestamp
    covariance: bool = False
    include_nongrav: bool = True


def _with_diagonal_covariance(orbits: Orbits, *, dimension: int) -> Orbits:
    if dimension not in (6, 9):
        raise ValueError(f"expected covariance dimension 6 or 9, got {dimension}")
    sigma = np.array(
        [1e-8, 1e-8, 1e-8, 1e-10, 1e-10, 1e-10, 2e-14, 3e-14, 1e-14],
        dtype=np.float64,
    )[:dimension]
    matrix = np.tile(np.diag(sigma**2), (len(orbits), 1, 1))
    return orbits.set_column(
        "coordinates.covariance", CoordinateCovariances.from_matrix(matrix)
    )


def _with_nongrav(orbits: Orbits) -> Orbits:
    rows = len(orbits)
    index = np.arange(rows)
    return orbits.set_column(
        "non_gravitational_parameters",
        NonGravitationalParameters.from_kwargs(
            source=["benchmark"] * rows,
            A1=(5e-13 * (1.0 + 0.01 * index)).tolist(),
            A2=(-2.9e-14 * (1.0 + 0.01 * index)).tolist(),
            A3=[None] * rows,
        ),
    )


def _workloads() -> list[Workload]:
    tiny = base._base_sun_ecliptic_orbits(5, mixed_epochs=False)
    small = base._base_sun_ecliptic_orbits(40, mixed_epochs=False)
    large = base._base_sun_ecliptic_orbits(200, mixed_epochs=False)
    nongrav_small = _with_nongrav(small)
    nongrav_large = _with_nongrav(large)
    return [
        Workload(
            "tiny",
            "gravity_5x2",
            "Gravity-only ephemeris: 5 orbits by 2 X05 observer epochs.",
            tiny,
            base._target_times(2, scale="utc", span_days=1.0),
        ),
        Workload(
            "small",
            "gravity_40x20",
            "Gravity-only ephemeris: 40 orbits by 20 X05 observer epochs.",
            small,
            base._target_times(20, scale="utc", span_days=30.0),
        ),
        Workload(
            "large",
            "gravity_200x50_1yr",
            "Gravity-only ephemeris: 200 orbits by 50 X05 observer epochs over one year.",
            large,
            base._long_horizon_target_times(50, scale="utc"),
        ),
        Workload(
            "small",
            "nongrav_40x20",
            "A1/A2 non-gravitational ephemeris: 40 orbits by 20 X05 observer epochs.",
            nongrav_small,
            base._target_times(20, scale="utc", span_days=30.0),
        ),
        Workload(
            "large",
            "nongrav_200x50_1yr",
            "A1/A2 non-gravitational ephemeris: 200 orbits by 50 X05 observer epochs over one year.",
            nongrav_large,
            base._long_horizon_target_times(50, scale="utc"),
        ),
        Workload(
            "small",
            "covariance_6d_sigma_point_10x10",
            "Deterministic 6D sigma-point covariance ephemeris: 10 orbits by 10 X05 epochs.",
            _with_diagonal_covariance(
                base._base_sun_ecliptic_orbits(10, mixed_epochs=False), dimension=6
            ),
            base._target_times(10, scale="utc", span_days=10.0),
            covariance=True,
        ),
        Workload(
            "small",
            "covariance_9d_nongrav_sigma_point_10x10",
            "Deterministic full 9D A1/A2/A3 sigma-point covariance ephemeris: 10 orbits by 10 X05 epochs.",
            _with_diagonal_covariance(
                _with_nongrav(base._base_sun_ecliptic_orbits(10, mixed_epochs=False)),
                dimension=9,
            ),
            base._target_times(10, scale="utc", span_days=10.0),
            covariance=True,
        ),
    ]


def _residuals(actual: Any, expected: Any) -> dict[str, Any]:
    delta = np.abs(actual.coordinates.values - expected.coordinates.values)
    a_cov = np.asarray(actual.coordinates.covariance.to_matrix(), dtype=np.float64)
    e_cov = np.asarray(expected.coordinates.covariance.to_matrix(), dtype=np.float64)
    finite = np.isfinite(a_cov) & np.isfinite(e_cov)
    covariance_max_abs = (
        float(np.max(np.abs(a_cov[finite] - e_cov[finite]))) if np.any(finite) else None
    )
    actual_time = actual.coordinates.time
    expected_time = expected.coordinates.time
    time_delta_ns = np.abs(
        (actual_time.days.to_numpy() - expected_time.days.to_numpy())
        * 86_400_000_000_000
        + actual_time.nanos.to_numpy()
        - expected_time.nanos.to_numpy()
    )
    return {
        "rows": len(actual),
        "range_abs_au": float(delta[:, 0].max(initial=0.0)),
        "range_abs_m": float(delta[:, 0].max(initial=0.0) * base.AU_METERS),
        "longitude_abs_deg": float(delta[:, 1].max(initial=0.0)),
        "latitude_abs_deg": float(delta[:, 2].max(initial=0.0)),
        "range_rate_abs_au_per_day": float(delta[:, 3].max(initial=0.0)),
        "range_rate_abs_m_per_s": float(
            delta[:, 3].max(initial=0.0) * base.AU_METERS / base.SECONDS_PER_DAY
        ),
        "longitude_rate_abs_deg_per_day": float(delta[:, 4].max(initial=0.0)),
        "latitude_rate_abs_deg_per_day": float(delta[:, 5].max(initial=0.0)),
        "time_abs_ns": int(time_delta_ns.max(initial=0)),
        "covariance_max_abs": covariance_max_abs,
    }


def _benchmark(
    workload: Workload,
    legacy: LegacyAssistPropagator,
    current: ASSISTPropagator,
    *,
    repeats: int,
    warmups: int,
) -> dict[str, Any]:
    observers = Observers.from_code("X05", workload.observer_times)
    kwargs: dict[str, Any] = {
        "covariance": workload.covariance,
        "max_processes": 1,
        "predict_magnitudes": False,
        "predict_phase_angle": False,
        "include_nongrav": workload.include_nongrav,
    }
    if workload.covariance:
        kwargs.update(covariance_method="sigma-point", num_samples=1000, seed=None)

    legacy_samples = legacy.time_generate_ephemeris(
        workload.orbits,
        observers,
        repeats=repeats,
        warmups=warmups,
        **kwargs,
    )
    legacy_output = legacy.generate_ephemeris(workload.orbits, observers, **kwargs)
    current_samples, current_output = time_rust(
        lambda: current.generate_ephemeris(workload.orbits, observers, **kwargs),
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
            "n_observers": len(observers),
            "output_rows": len(workload.orbits) * len(observers),
            "covariance_dimension": (
                None
                if not workload.covariance
                else (
                    9
                    if workload.orbits.coordinates.covariance.has_nongrav_block()
                    else 6
                )
            ),
        },
        "options": kwargs,
        "timing_seconds": performance_timing_payload(
            legacy_samples,
            current_samples,
            native_samples,
            native_operation=operation,
        ),
        "residuals": _residuals(current_output, legacy_output),
    }


def main() -> int:
    from migration.parity._assist_oracle import (
        LEGACY_ASSIST_VENV_PYTHON,
        LegacyAssistPropagator,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
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
