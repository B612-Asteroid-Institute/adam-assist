"""Three-boundary ASSIST OD/LSQ performance and parity matrix.

The frozen updated-upstream side runs its public Python/SciPy orchestration in
the isolated ASSIST oracle runtime. The current side enters the corresponding
public Rust-backed facade once. Native samples come only from the propagator's
Rust-owned ``std::time::Instant`` timing hook.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from adam_core.coordinates import (
    CartesianCoordinates,
    CoordinateCovariances,
    Origin,
    SphericalCoordinates,
)
from adam_core.observers import Observers
from adam_core.orbit_determination.evaluate import OrbitDeterminationObservations
from adam_core.orbits import Orbits
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
    percentiles,
    performance_timing_payload,
    time_native_rust,
    time_rust,
)

DEFAULT_OUTPUT = (
    REPO_ROOT / "migration" / "artifacts" / "assist_od_benchmark_2026-08-12.json"
)
EPOCH_MJD = 60000.0
TRUTH_STATE = np.array([1.2, 0.1, 0.05, -0.002, 0.016, 0.001])


def _make_orbit(state: np.ndarray, orbit_id: str) -> Orbits:
    return Orbits.from_kwargs(
        orbit_id=[orbit_id],
        object_id=[orbit_id],
        coordinates=CartesianCoordinates.from_kwargs(
            x=[state[0]],
            y=[state[1]],
            z=[state[2]],
            vx=[state[3]],
            vy=[state[4]],
            vz=[state[5]],
            time=Timestamp.from_mjd([EPOCH_MJD], scale="tdb"),
            origin=Origin.from_kwargs(code=["SUN"]),
            frame="ecliptic",
        ),
    )


def _problem(rows: int) -> tuple[OrbitDeterminationObservations, Orbits]:
    times = Timestamp.from_mjd(
        [EPOCH_MJD + 2.0 + i * 3.0 for i in range(rows)], scale="utc"
    )
    observers = Observers.from_code("X05", times)
    truth = _make_orbit(TRUTH_STATE, "truth")
    predicted = (
        ASSISTPropagator()
        .generate_ephemeris(
            truth,
            observers,
            covariance=False,
            max_processes=1,
            predict_magnitudes=False,
            predict_phase_angle=False,
        )
        .coordinates
    )
    arcsec_variance = (1.0 / 3600.0) ** 2
    covariance = np.tile(
        np.diag([1.0, arcsec_variance, arcsec_variance, 1.0, 1.0, 1.0]),
        (rows, 1, 1),
    )
    observed = SphericalCoordinates.from_kwargs(
        rho=predicted.rho,
        lon=predicted.lon,
        lat=predicted.lat,
        vrho=predicted.vrho,
        vlon=predicted.vlon,
        vlat=predicted.vlat,
        time=predicted.time,
        origin=predicted.origin,
        frame=predicted.frame,
        covariance=CoordinateCovariances.from_matrix(covariance),
    )
    observations = OrbitDeterminationObservations.from_kwargs(
        id=[f"obs-{i}" for i in range(rows)],
        coordinates=observed,
        observers=observers,
    )
    initial = _make_orbit(
        TRUTH_STATE + np.array([1e-3, -1e-3, 5e-4, 1e-5, -1e-5, 1e-5]),
        "fit",
    )
    return observations, initial


def _state_from_legacy(result: Any) -> np.ndarray:
    if isinstance(result, tuple):
        orbit = result[0]
    else:
        orbit = result
    if orbit is None or len(orbit) == 0:
        return np.full(6, np.nan)
    return np.asarray(orbit.coordinates.values[0], dtype=np.float64)


def _state_from_current(operation: str, result: Any) -> np.ndarray:
    if operation == "fit_least_squares":
        return np.asarray(result[0].coordinates.values[0], dtype=np.float64)
    return np.asarray(result["state"], dtype=np.float64)


def _benchmark_operation(
    operation: str,
    observations: OrbitDeterminationObservations,
    initial: Orbits,
    legacy: LegacyAssistPropagator,
    current: ASSISTPropagator,
    *,
    repeats: int,
    warmups: int,
) -> dict[str, Any]:
    if operation == "fit_least_squares":

        def legacy_call() -> Any:
            return legacy.fit_least_squares_public(initial, observations)

        def legacy_time() -> list[float]:
            return legacy.time_fit_least_squares_public(
                initial, observations, repeats=repeats, warmups=warmups
            )

        def current_call() -> Any:
            return current.fit_least_squares(initial, observations)

    elif operation == "od_fit":
        kwargs = {
            "rchi2_threshold": 100.0,
            "min_obs": 5,
            "min_arc_length": 1.0,
            "contamination_percentage": 0.0,
            "delta": 1e-6,
            "max_iter": 20,
            "method": "central",
        }

        def legacy_call() -> Any:
            return legacy.od(initial, observations, **kwargs)

        def legacy_time() -> list[float]:
            return legacy.time_od(
                initial, observations, repeats=repeats, warmups=warmups, **kwargs
            )

        def current_call() -> Any:
            return current.od_fit(initial, observations, **kwargs)

    elif operation == "vallado_least_squares":

        def legacy_call() -> Any:
            return legacy.vallado_least_squares(initial, observations, True)

        def legacy_time() -> list[float]:
            return legacy.time_vallado_least_squares(
                initial,
                observations,
                True,
                repeats=repeats,
                warmups=warmups,
            )

        def current_call() -> Any:
            return current.vallado_least_squares(
                initial, observations, use_central_difference=True
            )

    else:
        raise ValueError(operation)

    legacy_samples = legacy_time()
    legacy_result = legacy_call()
    current_samples, current_result = time_rust(
        current_call, repeats=repeats, warmups=warmups
    )
    try:
        native_operation, native_samples = time_native_rust(
            current, repeats=repeats, warmups=warmups
        )
        timing = performance_timing_payload(
            legacy_samples,
            current_samples,
            native_samples,
            native_operation=native_operation,
        )
    except ValueError as exc:
        legacy_p50, legacy_p95 = percentiles(legacy_samples)
        current_p50, current_p95 = percentiles(current_samples)
        timing = {
            "legacy_adam_core": {"p50": legacy_p50, "p95": legacy_p95},
            "current_python": {"p50": current_p50, "p95": current_p95},
            "native_rust": {
                "status": "unavailable",
                "reason": str(exc),
            },
            "speedup": {
                "p50_legacy_over_current_python": legacy_p50 / current_p50,
                "p95_legacy_over_current_python": legacy_p95 / current_p95,
            },
        }
    current_state = _state_from_current(operation, current_result)
    legacy_state = _state_from_legacy(legacy_result)
    return {
        "lane": "small",
        "name": operation,
        "description": (
            f"Noise-free one-orbit {operation} with {len(observations)} X05 "
            "astrometric observations over a 21-day arc."
        ),
        "workload_shape": {
            "n_starting_orbits": 1,
            "n_observations": len(observations),
            "arc_days": 3.0 * (len(observations) - 1),
            "state_parameters": 6,
        },
        "timing_seconds": timing,
        "residuals": {
            "state_max_abs_current_vs_updated_upstream": float(
                np.nanmax(np.abs(current_state - legacy_state))
            ),
            "state_max_abs_current_vs_truth": float(
                np.nanmax(np.abs(current_state - TRUTH_STATE))
            ),
        },
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
    observations, initial = _problem(8)
    legacy = LegacyAssistPropagator()
    current = ASSISTPropagator()
    rows = [
        _benchmark_operation(
            operation,
            observations,
            initial,
            legacy,
            current,
            repeats=args.repeats,
            warmups=args.warmups,
        )
        for operation in ("fit_least_squares", "od_fit", "vallado_least_squares")
    ]
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "comparison_mode": TWO_RUNTIME_COMPARISON_MODE,
        "performance_columns": PERFORMANCE_COLUMNS,
        "repeats": args.repeats,
        "warmups": args.warmups,
        "workloads": rows,
        "iod": {
            "status": "parity-and-native-timing-tested; no frozen updated-upstream performance row",
            "reason": (
                "The frozen oracle adapter has no timing/result request for public "
                "initial_orbit_determination; do not mislabel a non-equivalent Python "
                "composition as an apples-to-apples performance comparison."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
