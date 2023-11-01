import numpy as np
from adam_assist import ASSISTPropagator
from adam_core.orbits.query import query_sbdb
from adam_core.time import Timestamp


def test_assist_propagator():
    initial_time = Timestamp.from_mjd([60000.0], scale="tdb")
    times = initial_time.from_mjd(initial_time.mjd() + np.arange(0, 100))
    orbits = query_sbdb(["2013 RR165"])
    propagator = ASSISTPropagator()
    propagated_orbits = propagator.propagate_orbits(orbits, times)
    return propagated_orbits
