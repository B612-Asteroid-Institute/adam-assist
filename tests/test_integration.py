import time

import numpy as np
import pytest
from adam_core.dynamics.impacts import calculate_impacts
from adam_core.orbits import Orbits
from adam_core.orbits.query import query_sbdb
from adam_core.time import Timestamp

from adam_core.propagator.adam_assist import (
    ASSISTPropagator,
    download_jpl_ephemeris_files,
)

# Contains a likely impactor with ~60% chance of impact in 30 days
IMPACTOR_FILE_PATH = "tests/data/I00007_orbit.parquet"



def test_detect_impacts():
    impactors = Orbits.from_parquet(IMPACTOR_FILE_PATH)[0]
    propagator = ASSISTPropagator()
    variants, impacts = propagator.detect_impacts(impactors)
    return variants, impacts

@pytest.mark.benchmark
@pytest.mark.parametrize("processes", [1, None])
def test_calculate_impacts_benchmark(benchmark, processes):
    download_jpl_ephemeris_files()
    impactor = Orbits.from_parquet(IMPACTOR_FILE_PATH)[0]
    propagator = ASSISTPropagator()
    variants, impacts = benchmark(calculate_impacts, impactor, 60, propagator, num_samples=1000, processes=processes)
    assert len(variants) == 1000, "Should have 1000 variants"
    assert len(impacts) > 500, "Should have at least 500 impactors"
    assert len(impacts) < 700, "Should have less than 700 impactors"
