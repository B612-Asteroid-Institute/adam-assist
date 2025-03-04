import pytest
import pyarrow.compute as pc

from adam_core.constants import Constants as c
from adam_core.constants import KM_P_AU
from adam_core.coordinates import Origin
from adam_core.orbits import Orbits, VariantOrbits
from adam_core.dynamics.impacts import calculate_impacts, CollisionConditions, EARTH_RADIUS_KM
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

R_EARTH_KM = c.R_EARTH_EQUATORIAL * KM_P_AU


@pytest.mark.benchmark
@pytest.mark.parametrize("processes", [1, 2])
def test_calculate_impacts_benchmark_some_impacts(benchmark, processes):
    impactor = Orbits.from_parquet(IMPACTOR_FILE_PATH_60)[0]
    propagator = ASSISTPropagator()
    variants, impacts = benchmark(
        calculate_impacts,
        impactor,
        60,
        propagator,
        num_samples=200,
        processes=processes,
        seed=42,  # This allows us to predict exact number of impactors empirically
    )
    assert len(variants) == 200, "Should have 200 variants"
    assert len(impacts) == 138, "Should have exactly 138 impactors"

@pytest.mark.parametrize("processes", [1, 2])
def test_calculate_impacts(processes):
    impactor = Orbits.from_parquet(IMPACTOR_FILE_PATH_60)[0]
    propagator = ASSISTPropagator()

    variants_orbits = VariantOrbits.create(impactor, method="monte-carlo", num_samples=200, seed=42)
    variants, impacts = propagator.detect_collisions(
        variants_orbits,
        60,
        max_processes=processes,
    )
    assert len(variants) == 200, "Should have 200 variants"
    assert len(impacts) == 138, "Should have exactly 138 impactors"

    assert impacts.collision_coordinates.frame == "ecliptic"
    assert pc.all(pc.equal(impacts.collision_coordinates.origin.code, "EARTH")).as_py()
    assert pc.all(pc.less_equal(impacts.collision_coordinates.rho, R_EARTH_KM)).as_py()


@pytest.mark.benchmark
@pytest.mark.parametrize("processes", [1, 2])
def test_calculate_impacts_benchmark_impacts(benchmark, processes):

    impactor = Orbits.from_parquet(IMPACTOR_FILE_PATH_100)[0]
    propagator = ASSISTPropagator()
    variants, impacts = benchmark(
        calculate_impacts,
        impactor,
        60,
        propagator,
        num_samples=200,
        processes=processes,
        seed=42,  # This allows us to predict exact number of impactors empirically
        conditions=CollisionConditions.default(),
    )
    assert len(variants) == 200, "Should have 200 variants"
    assert len(impacts) == 200, "Should have exactly 200 impactors"


@pytest.mark.benchmark
@pytest.mark.parametrize("processes", [1, 2])
def test_calculate_impacts_benchmark_no_impacts(benchmark, processes):
    impactor = Orbits.from_parquet(IMPACTOR_FILE_PATH_0)[0]
    propagator = ASSISTPropagator()
    variants, impacts = benchmark(
        calculate_impacts,
        impactor,
        60,
        propagator,
        num_samples=200,
        processes=processes,
        seed=42,  # This allows us to predict exact number of impactors empirically
        conditions=CollisionConditions.default(),
    )
    assert len(variants) == 200, "Should have 200 variants"
    assert len(impacts) == 0, "Should have exactly 0 impactors"


def test_detect_collisions_time_direction():
    start_time = Timestamp.from_mjd([60000], scale="utc")
    orbit = query_horizons(["1980 PA"], start_time)

    conditions = CollisionConditions.default()

    propagator = ASSISTPropagator()

    results, impacts = propagator._detect_collisions(orbit, 60, conditions)
    assert (
        results.coordinates.time.mjd().to_numpy()[0]
        >= orbit.coordinates.time.add_days(60).mjd().to_numpy()[0]
    )

    results, impacts = propagator._detect_collisions(orbit, -60, conditions)
    assert (
        results.coordinates.time.mjd().to_numpy()[0]
        <= orbit.coordinates.time.add_days(-60).mjd().to_numpy()[0]
    )
