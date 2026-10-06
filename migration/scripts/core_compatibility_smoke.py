#!/usr/bin/env python3
"""Exercise one installed adam-assist wheel across a supported Core 0.5 line."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import adam_core
import adam_core._rust_native
import numpy as np
import pyarrow as pa
from adam_core.time import Timestamp

import adam_assist
from adam_assist import ASSISTPropagator
from adam_assist import _native as adam_assist_native
from adam_assist.version import __version__ as runtime_assist_version

APOPHIS_EPOCH_MJD_TDB = 59215.0
APOPHIS_STATE = np.array(
    [
        -0.4098530841678254,
        0.9621648472038595,
        -0.06096604136465475,
        -0.015075524121655987,
        -0.004107955457204797,
        -0.00013931296759505984,
    ],
    dtype=np.float64,
)
TARGET_OFFSETS_DAYS = np.array([1.0, 7.0], dtype=np.float64)


def _orbit() -> Any:
    from adam_core.coordinates import CartesianCoordinates, Origin
    from adam_core.orbits import Orbits
    from adam_core.time import Timestamp

    values = APOPHIS_STATE
    return Orbits.from_kwargs(
        orbit_id=["compatibility-apophis"],
        object_id=["99942 Apophis (2004 MN4)"],
        coordinates=CartesianCoordinates.from_kwargs(
            x=[values[0]],
            y=[values[1]],
            z=[values[2]],
            vx=[values[3]],
            vy=[values[4]],
            vz=[values[5]],
            time=Timestamp.from_mjd([APOPHIS_EPOCH_MJD_TDB], scale="tdb"),
            origin=Origin.from_kwargs(code=["SUN"]),
            frame="ecliptic",
        ),
    )


def _installed_module(module: Any) -> str:
    path = Path(module.__file__).resolve()
    try:
        path.relative_to(Path(sys.prefix).resolve())
    except ValueError as error:
        raise AssertionError(
            f"module imported outside compatibility venv: {path}"
        ) from error
    return str(path)


def _schema_sha256(table: pa.Table) -> str:
    return hashlib.sha256(table.schema.serialize().to_pybytes()).hexdigest()


def _arrow_round_trip(orbits: Any) -> Any:
    from adam_core.orbits import Orbits

    table = orbits.table.combine_chunks()
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    restored_table = pa.ipc.open_stream(sink.getvalue()).read_all()
    restored = Orbits.from_pyarrow(restored_table)
    if restored.table.schema != table.schema:
        raise AssertionError("Orbits Arrow schema changed across IPC round trip")
    if restored.orbit_id.to_pylist() != orbits.orbit_id.to_pylist():
        raise AssertionError("Orbits Arrow round trip changed orbit IDs")
    np.testing.assert_array_equal(
        restored.coordinates.values, orbits.coordinates.values
    )
    return restored


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-core-version", required=True)
    parser.add_argument("--expected-assist-version", required=True)
    parser.add_argument("--assist-wheel", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    versions = {
        "adam-core": importlib.metadata.version("adam-core"),
        "adam-assist": importlib.metadata.version("adam-assist"),
    }
    expected = {
        "adam-core": args.expected_core_version,
        "adam-assist": args.expected_assist_version,
    }
    if versions != expected:
        raise AssertionError(f"distribution versions {versions} != {expected}")
    if adam_core.__version__ != args.expected_core_version:
        raise AssertionError("adam-core runtime/distribution versions differ")
    if runtime_assist_version != args.expected_assist_version:
        raise AssertionError("adam-assist runtime/distribution versions differ")

    wheel = args.assist_wheel.resolve()
    wheel_sha256 = hashlib.sha256(wheel.read_bytes()).hexdigest()
    orbits = _arrow_round_trip(_orbit())
    targets = Timestamp.from_mjd(
        APOPHIS_EPOCH_MJD_TDB + TARGET_OFFSETS_DAYS, scale="tdb"
    )
    propagated = ASSISTPropagator().propagate_orbits(
        orbits,
        targets,
        covariance=False,
        max_processes=1,
        chunk_size=1,
    )
    propagated = _arrow_round_trip(propagated)
    values = np.asarray(propagated.coordinates.values, dtype=np.float64)
    if values.shape != (len(TARGET_OFFSETS_DAYS), 6):
        raise AssertionError(f"unexpected propagated shape {values.shape}")
    if not np.isfinite(values).all():
        raise AssertionError("propagation returned non-finite values")
    if propagated.orbit_id.to_pylist() != ["compatibility-apophis"] * len(
        TARGET_OFFSETS_DAYS
    ):
        raise AssertionError("propagation changed orbit IDs")
    epochs = np.asarray(
        propagated.coordinates.time.mjd().to_numpy(zero_copy_only=False),
        dtype=np.float64,
    )
    np.testing.assert_allclose(
        epochs,
        APOPHIS_EPOCH_MJD_TDB + TARGET_OFFSETS_DAYS,
        rtol=0.0,
        atol=1.0e-12,
    )
    if not all(math.isfinite(value) for value in values.ravel()):
        raise AssertionError("propagation contains a non-finite scalar")

    report = {
        "schema_version": 1,
        "status": "passed",
        "versions": versions,
        "assist_wheel": str(wheel),
        "assist_wheel_sha256": wheel_sha256,
        "module_paths": {
            "adam_core": _installed_module(adam_core),
            "adam_core._rust_native": _installed_module(adam_core._rust_native),
            "adam_assist": _installed_module(adam_assist),
            "adam_assist._native": _installed_module(adam_assist_native),
        },
        "input_schema": str(orbits.table.schema),
        "input_schema_sha256": _schema_sha256(orbits.table),
        "output_schema": str(propagated.table.schema),
        "output_schema_sha256": _schema_sha256(propagated.table),
        "epochs_mjd_tdb": epochs.tolist(),
        "propagated_values": values.tolist(),
        "kernel_offline": os.environ.get("ADAM_CORE_KERNEL_OFFLINE"),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
