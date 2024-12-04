import pytest
from adam_core.dynamics.impacts import calculate_impacts
from adam_core.orbits import Orbits
from adam_core.orbits.query.horizons import query_horizons
from adam_core.time import Timestamp

from adam_assist import ASSISTPropagator

# Contains a likely impactor with ~60% chance of impact in 30 days
IMPACTOR_FILE_PATH_60 = "tests/data/I00007_orbit.parquet"
# Contains a likely impactor with 100% chance of impact in 30 days
IMPACTOR_FILE_PATH_100 = "tests/data/I00008_orbit.parquet"
# Contains a likely impactor with 0% chance of impact in 30 days
IMPACTOR_FILE_PATH_0 = "tests/data/I00009_orbit.parquet"


@pytest.mark.benchmark
@pytest.mark.parametrize("processes", [1, 2])
def test_calculate_impacts_benchmark(benchmark, processes):
    impactor = Orbits.from_parquet(IMPACTOR_FILE_PATH_60)[0]
    propagator = ASSISTPropagator()
    variants, impacts = benchmark(
        calculate_impacts,
        impactor,
        60,
        propagator,
        num_samples=200,
        processes=processes,
        seed=42  # This allows us to predict exact number of impactors empirically
    )
    assert len(variants) == 200, "Should have 200 variants"
    assert len(impacts) == 138, "Should have exactly 138 impactors"


@pytest.mark.benchmark
@pytest.mark.parametrize("processes", [1, 2])
def test_calculate_impacts_benchmark(benchmark, processes):
    
    impactor = Orbits.from_parquet(IMPACTOR_FILE_PATH_100)[0]
    propagator = ASSISTPropagator()
    variants, impacts = benchmark(
        calculate_impacts,
        impactor,
        60,
        propagator,
        num_samples=200,
        processes=processes,
        seed=42  # This allows us to predict exact number of impactors empirically
    )
    assert len(variants) == 200, "Should have 200 variants"
    assert len(impacts) == 200, "Should have exactly 200 impactors"


@pytest.mark.benchmark
@pytest.mark.parametrize("processes", [1, 2])
def test_calculate_impacts_benchmark(benchmark, processes):
    impactor = Orbits.from_parquet(IMPACTOR_FILE_PATH_0)[0]
    propagator = ASSISTPropagator()
    variants, impacts = benchmark(
        calculate_impacts,
        impactor,
        60,
        propagator,
        num_samples=200,
        processes=processes,
        seed=42  # This allows us to predict exact number of impactors empirically
    )
    assert len(variants) == 200, "Should have 200 variants"
    assert len(impacts) == 0, "Should have exactly 0 impactors"


def test_detect_impacts_time_direction():
    start_time = Timestamp.from_mjd([60000], scale="utc")
    orbit = query_horizons(["1980 PA"], start_time)
    
    propagator = ASSISTPropagator()

    results, impacts = propagator._detect_impacts(orbit, 60)
    assert results.coordinates.time.mjd().to_numpy()[0] >= orbit.coordinates.time.add_days(60).mjd().to_numpy()[0]

    results, impacts = propagator._detect_impacts(orbit, -60)
    assert results.coordinates.time.mjd().to_numpy()[0] <= orbit.coordinates.time.add_days(-60).mjd().to_numpy()[0]
