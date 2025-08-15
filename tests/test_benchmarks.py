import time
import numpy as np
import pytest

from adam_core.orbits.query.sbdb import query_sbdb
from adam_core.time import Timestamp
from adam_core.observers.observers import Observers
from adam_assist import ASSISTPropagator
from adam_core.dynamics.impacts import calculate_impacts, CollisionConditions
from adam_core.orbits import Orbits
from adam_core.orbits.variants import VariantOrbits


def build_time_grid(start_mjd: float, years: float, step_days: float) -> Timestamp:
    num_days = int(years * 365.25)
    mjds = np.arange(start_mjd, start_mjd + num_days + 1, step_days, dtype=float)
    return Timestamp.from_mjd(mjds, scale="tdb")


@pytest.mark.benchmark
def test_benchmark_propagation_vs_raw(benchmark):
    ids = [
        "2020 AV2",
        "A919 FB",
        "1993 SB",
        "433 Eros",
        "99942 Apophis",
        "202930 Ivezic",
        "3200 Phaethon",
        "101955 Bennu",
        "25143 Itokawa",
        "4179 Toutatis",
    ]
    orbits = query_sbdb(ids)
    times = build_time_grid(60000.0, 10.0, 1.0)

    prop = ASSISTPropagator()
    benchmark(prop.propagate_orbits, orbits, times)


@pytest.mark.benchmark
def test_benchmark_ephemeris_generation(benchmark):
    ids = ["99942 Apophis"]
    orbits = query_sbdb(ids)
    times = build_time_grid(60000.0, 1.0, 1.0)
    observers = Observers.from_code("X05", times)

    prop = ASSISTPropagator()
    benchmark(prop.generate_ephemeris, orbits, observers)


@pytest.mark.benchmark
def test_benchmark_impact_detection(benchmark):
    impactor = Orbits.from_parquet("tests/data/I00007_orbit.parquet")[0]
    prop = ASSISTPropagator()
    variants, impacts = benchmark(
        calculate_impacts,
        impactor,
        60,
        prop,
        200,
        1,
        42,
        CollisionConditions.default(),
    )
    assert len(variants) == 200


