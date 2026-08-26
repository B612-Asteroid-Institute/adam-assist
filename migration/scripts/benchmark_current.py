"""Benchmark current adam-assist public and native Rust boundaries only.

Workloads are imported from the existing propagation, nongrav, ephemeris,
collision, and orbit-determination benchmark modules. The suite never starts
the frozen adam-assist oracle and always uses ``max_processes=1``.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Callable

import numpy as np
from adam_core.observers import Observers

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adam_assist import ASSISTPropagator  # noqa: E402
from migration.parity._assist_bench import (  # noqa: E402
    percentiles,
    time_native_rust,
    time_rust,
)
from migration.scripts import benchmark_assist_ephemeris as ephemeris  # noqa: E402
from migration.scripts import benchmark_assist_impacts as impacts  # noqa: E402
from migration.scripts import (  # noqa: E402
    benchmark_assist_nongrav_propagation as nongrav,
)
from migration.scripts import benchmark_assist_od as od  # noqa: E402
from migration.scripts import (  # noqa: E402
    benchmark_assist_public_semantics as propagation,
)

DEFAULT_OUTPUT = REPO_ROOT / "migration" / "artifacts" / "benchmark_current_assist.json"
DEFAULT_MARKDOWN = REPO_ROOT / "migration" / "artifacts" / "benchmark_current_assist.md"
DOMAINS = (
    "propagation",
    "nongrav",
    "ephemeris",
    "covariance",
    "collisions",
    "od",
)
LANES = ("tiny", "small", "large")


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _duration(value: float | None) -> str:
    if value is None:
        return "—"
    if value < 1e-3:
        return f"{value * 1e6:.1f} µs"
    if value < 1.0:
        return f"{value * 1e3:.2f} ms"
    return f"{value:.3f} s"


def _timing(
    current: ASSISTPropagator,
    call: Callable[[], Any],
    *,
    repeats: int,
    warmups: int,
) -> tuple[dict[str, Any], Any]:
    current_samples, result = time_rust(call, repeats=repeats, warmups=warmups)
    current_p50, current_p95 = percentiles(current_samples)
    try:
        operation, native_samples = time_native_rust(
            current, repeats=repeats, warmups=warmups
        )
        native_p50, native_p95 = percentiles(native_samples)
        native = {
            "status": "measured",
            "operation": operation,
            "timer": "std::time::Instant",
            "values": native_samples,
            "p50": native_p50,
            "p95": native_p95,
        }
    except ValueError as exc:
        native_p50 = native_p95 = None
        native = {"status": "unavailable", "reason": str(exc)}
    return {
        "current_python": {
            "values": current_samples,
            "p50": current_p50,
            "p95": current_p95,
        },
        "native_rust": native,
        "public_over_native": {
            "p50": current_p50 / native_p50 if native_p50 else None,
            "p95": current_p95 / native_p95 if native_p95 else None,
        },
    }, result


def _propagation_row(
    workload: propagation.Workload,
    current: ASSISTPropagator,
    *,
    repeats: int,
    warmups: int,
    domain: str,
) -> dict[str, Any]:
    kwargs = {
        "covariance": False,
        "chunk_size": workload.chunk_size,
        "max_processes": 1,
        "include_nongrav": domain == "nongrav",
    }
    timing, _ = _timing(
        current,
        lambda: current.propagate_orbits(workload.orbits, workload.times, **kwargs),
        repeats=repeats,
        warmups=warmups,
    )
    return {
        "domain": domain,
        "lane": workload.lane,
        "name": workload.name,
        "description": workload.description,
        "workload_shape": {
            "n_orbits": len(workload.orbits),
            "n_target_times": len(workload.times),
            "output_rows": len(workload.orbits) * len(workload.times),
            "unique_input_epochs": len(workload.orbits.coordinates.time.unique()),
        },
        "options": kwargs,
        "timing_seconds": timing,
    }


def _ephemeris_row(
    workload: ephemeris.Workload,
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
    timing, _ = _timing(
        current,
        lambda: current.generate_ephemeris(workload.orbits, observers, **kwargs),
        repeats=repeats,
        warmups=warmups,
    )
    return {
        "domain": "covariance" if workload.covariance else "ephemeris",
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
        "timing_seconds": timing,
    }


def _collision_rows(
    current: ASSISTPropagator,
    *,
    repeats: int,
    warmups: int,
    lane_names: set[str],
) -> list[dict[str, Any]]:
    lane_sizes = {"tiny": 10, "small": 50, "large": 200}
    rng = np.random.default_rng(20260703)
    conditions = impacts._conditions()
    rows = []
    for lane, size in lane_sizes.items():
        if lane not in lane_names:
            continue
        orbits = impacts._lane_orbits(
            size, rng, impactor_fraction=impacts.IMPACTOR_FRACTION
        )
        timing, result = _timing(
            current,
            lambda o=orbits: current.detect_collisions(
                o, impacts.NUM_DAYS, conditions, max_processes=1
            ),
            repeats=repeats,
            warmups=warmups,
        )
        rows.append(
            {
                "domain": "collisions",
                "lane": lane,
                "name": f"detect_collisions_{size}",
                "description": (
                    f"{size} same-epoch impactor/safe heliocentric orbits over "
                    f"{impacts.NUM_DAYS} days."
                ),
                "workload_shape": {
                    "n_orbits": size,
                    "num_days": impacts.NUM_DAYS,
                    "n_impacts": len(result[1]),
                },
                "options": {"max_processes": 1},
                "timing_seconds": timing,
            }
        )
    return rows


def _od_call(
    operation: str,
    current: ASSISTPropagator,
    initial: Any,
    observations: Any,
) -> Callable[[], Any]:
    if operation == "fit_least_squares":
        return lambda: current.fit_least_squares(initial, observations)
    if operation == "fit_least_squares_evaluated":
        return lambda: current.fit_least_squares_evaluated(initial, observations)
    if operation == "od_fit":
        return lambda: current.od_fit(
            initial,
            observations,
            rchi2_threshold=100.0,
            min_obs=5,
            min_arc_length=1.0,
            contamination_percentage=0.0,
            delta=1e-6,
            max_iter=20,
            method="central",
        )
    if operation == "vallado_least_squares":
        return lambda: current.vallado_least_squares(
            initial, observations, use_central_difference=True
        )
    raise ValueError(operation)


def _od_rows(
    current: ASSISTPropagator,
    *,
    repeats: int,
    warmups: int,
) -> list[dict[str, Any]]:
    observations, initial = od._problem(8)
    rows = []
    operations = (
        "fit_least_squares",
        "fit_least_squares_evaluated",
        "od_fit",
        "vallado_least_squares",
    )
    for operation in operations:
        timing, _ = _timing(
            current,
            _od_call(operation, current, initial, observations),
            repeats=repeats,
            warmups=warmups,
        )
        rows.append(
            {
                "domain": "od",
                "lane": "small",
                "name": operation,
                "description": (
                    f"Noise-free one-orbit {operation} with 8 X05 astrometric "
                    "observations over a 21-day arc."
                ),
                "workload_shape": {
                    "n_starting_orbits": 1,
                    "n_observations": len(observations),
                    "arc_days": 21.0,
                    "state_parameters": 6,
                },
                "options": {"max_processes": 1},
                "timing_seconds": timing,
            }
        )
    obs_ids = observations.id.to_pylist()

    def iod_call() -> Any:
        return current.initial_orbit_determination(
            observations,
            ["b", "a"],
            ["b"] * len(obs_ids) + ["a"] * len(obs_ids),
            obs_ids + obs_ids,
            min_obs=3,
            min_arc_length=1.0,
            rchi2_threshold=1e12,
            contamination_percentage=0.0,
            chunk_size=2,
        )

    timing, _ = _timing(current, iod_call, repeats=repeats, warmups=warmups)
    rows.append(
        {
            "domain": "od",
            "lane": "small",
            "name": "initial_orbit_determination",
            "description": (
                "Two duplicate 8-observation linkages run through complete "
                "Gauss-IOD orchestration in one native crossing."
            ),
            "workload_shape": {
                "n_linkages": 2,
                "n_observations": len(observations),
                "n_members": 2 * len(obs_ids),
                "chunk_size": 2,
            },
            "options": {"max_processes": 1},
            "timing_seconds": timing,
        }
    )
    return rows


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Current adam-assist benchmark suite",
        "",
        (
            f"{len(payload['workloads'])} current-only workloads. No frozen Python "
            "runtime is used; every workload uses `max_processes=1`."
        ),
        "",
        "| Domain | Workload | Shape | Current public p50/p95 | Native Rust p50/p95 | Public/native p50/p95 |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in payload["workloads"]:
        timing = row["timing_seconds"]
        current = timing["current_python"]
        native = timing["native_rust"]
        overhead = timing["public_over_native"]
        p50_overhead = overhead["p50"]
        p95_overhead = overhead["p95"]
        lines.append(
            f"| {row['domain']} | `{row['name']}` ({row['lane']}) "
            f"| `{json.dumps(row['workload_shape'], sort_keys=True)}` "
            f"| {_duration(current['p50'])} / {_duration(current['p95'])} "
            f"| {_duration(native.get('p50'))} / {_duration(native.get('p95'))} "
            f"| {f'{p50_overhead:.2f}×' if p50_overhead is not None else '—'} / "
            f"{f'{p95_overhead:.2f}×' if p95_overhead is not None else '—'} |"
        )
    return "\n".join(lines) + "\n"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--domains", nargs="+", choices=DOMAINS, default=list(DOMAINS))
    parser.add_argument("--lanes", nargs="+", choices=LANES, default=list(LANES))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--require-native",
        action="store_true",
        help="Fail if any selected workload lacks genuine Rust-owned timing.",
    )
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use three repeats and one warmup for a local smoke run.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    repeats = 3 if args.quick else args.repeats
    warmups = 1 if args.quick else args.warmups
    if repeats < 3:
        raise ValueError("--repeats must be at least 3 to report p50/p95")
    if warmups < 0:
        raise ValueError("--warmups must be non-negative")
    domains = set(args.domains)
    lanes = set(args.lanes)
    current = ASSISTPropagator()
    rows: list[dict[str, Any]] = []
    if "propagation" in domains:
        rows.extend(
            _propagation_row(
                workload,
                current,
                repeats=repeats,
                warmups=warmups,
                domain="propagation",
            )
            for workload in propagation._workloads()
            if workload.lane in lanes
        )
    if "nongrav" in domains:
        rows.extend(
            _propagation_row(
                workload,
                current,
                repeats=repeats,
                warmups=warmups,
                domain="nongrav",
            )
            for workload in nongrav._workloads()
            if workload.lane in lanes
        )
    if "ephemeris" in domains or "covariance" in domains:
        rows.extend(
            _ephemeris_row(workload, current, repeats=repeats, warmups=warmups)
            for workload in ephemeris._workloads()
            if workload.lane in lanes
            and (
                (workload.covariance and "covariance" in domains)
                or (not workload.covariance and "ephemeris" in domains)
            )
        )
    if "collisions" in domains:
        rows.extend(
            _collision_rows(current, repeats=repeats, warmups=warmups, lane_names=lanes)
        )
    if "od" in domains and "small" in lanes:
        rows.extend(_od_rows(current, repeats=repeats, warmups=warmups))

    payload = {
        "schema_version": 1,
        "benchmark_id": "adam-assist-current-only",
        "comparison_mode": "current_public_and_native_rust_no_legacy",
        "legacy_timing_included": False,
        "generated_at": datetime.now(UTC).isoformat(),
        "packages": {
            "adam-assist": _package_version("adam-assist"),
            "adam-core": _package_version("adam-core"),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "max_processes": 1,
        "repeats": repeats,
        "warmups": warmups,
        "domains": args.domains,
        "lanes": args.lanes,
        "native_unavailable_count": sum(
            row["timing_seconds"]["native_rust"].get("status") != "measured"
            for row in rows
        ),
        "workloads": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(_markdown(payload))
    print(args.output)
    print(args.markdown)
    return 1 if args.require_native and payload["native_unavailable_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
