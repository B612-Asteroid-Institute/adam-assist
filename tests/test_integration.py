import numpy as np
from adam_core.orbits.query import query_sbdb
from adam_core.time import Timestamp

from adam_core.propagators.assist import ASSISTPropagator, download_jpl_ephemeris_files


def test_assist_propagator():
    download_jpl_ephemeris_files()
    initial_time = Timestamp.from_mjd([60000.0], scale="tdb")
    times = initial_time.from_mjd(initial_time.mjd() + np.arange(0, 100))
    orbits = query_sbdb(["EDLU"])
    propagator = ASSISTPropagator()
    assist_propagated_orbits = propagator.propagate_orbits(orbits, times)
    return assist_propagated_orbits
