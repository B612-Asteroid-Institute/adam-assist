import pytest
from adam_core.dynamics.impacts import calculate_impacts
from adam_core.orbits import Orbits

from src.adam_core.propagator.adam_assist import (
    ASSISTPropagator,
    download_jpl_ephemeris_files,
)

# Contains a likely impactor with ~60% chance of impact in 30 days
IMPACTOR_FILE_PATH = "tests/data/I00007_orbit.parquet"


@pytest.mark.benchmark
@pytest.mark.parametrize("processes", [1, 2])
def test_calculate_impacts_benchmark(benchmark, processes):
    download_jpl_ephemeris_files()
    impactor = Orbits.from_parquet(IMPACTOR_FILE_PATH)[0]
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
