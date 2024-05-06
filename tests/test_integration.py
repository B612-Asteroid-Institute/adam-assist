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

IMPACTORS_FILE_PATH = "tests/data/impactors.parquet"

def test_assist_propagator():
    download_jpl_ephemeris_files()
    initial_time = Timestamp.from_mjd([60000.0], scale="tdb")
    times = initial_time.from_mjd(initial_time.mjd() + np.arange(0, 100))
    orbits = query_sbdb(["EDLU"])
    propagator = ASSISTPropagator()
    assist_propagated_orbits = propagator.propagate_orbits(orbits, times)
    return assist_propagated_orbits


def test_detect_impacts():
    impactors = Orbits.from_parquet(IMPACTORS_FILE_PATH)[0]
    propagator = ASSISTPropagator()
    variants, impacts = propagator.detect_impacts(impactors)
    return variants, impacts

@pytest.mark.benchmark
def test_calculate_impacts(benchmark):
    download_jpl_ephemeris_files()
    impactor = Orbits.from_parquet("/Users/aleck/Downloads/I00007_orbit.parquet")[0]
    propagator = ASSISTPropagator()
    variants, impacts = benchmark(calculate_impacts, impactor, 60, propagator, num_samples=1000, processes=1)
    return variants, impacts


if __name__ == "__main__":
    test_calculate_impacts()