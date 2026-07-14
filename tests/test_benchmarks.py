import numpy as np
import pytest
from adam_core.dynamics.impacts import CollisionConditions, calculate_impacts
from adam_core.observers.observers import Observers
from adam_core.orbits import Orbits
from adam_core.time import Timestamp

from adam_assist import ASSISTPropagator

BENCHMARK_ORBITS_PATH = "tests/data/benchmark_orbits.parquet"


def build_time_grid(start_mjd: float, years: float, step_days: float) -> Timestamp:
    num_days = int(years * 365.25)
    mjds = np.arange(start_mjd, start_mjd + num_days + 1, step_days, dtype=float)
    return Timestamp.from_mjd(mjds, scale="tdb")


@pytest.mark.benchmark
def test_benchmark_propagation_vs_raw(benchmark):
    # Keep provider availability outside the compute benchmark. This fixture is
    # a ten-row, uniquely identified composition of the reviewed orbit fixtures
    # already used by the deterministic propagation and impact tests.
    orbits = Orbits.from_parquet(BENCHMARK_ORBITS_PATH)
    times = build_time_grid(60000.0, 10.0, 1.0)

    prop = ASSISTPropagator()
    benchmark(prop.propagate_orbits, orbits, times)


@pytest.mark.benchmark
def test_benchmark_ephemeris_generation(benchmark):
    orbits = Orbits.from_parquet(BENCHMARK_ORBITS_PATH)[0]
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
