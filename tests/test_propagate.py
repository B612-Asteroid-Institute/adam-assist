import pytest
from adam_core.orbits.query import query_sbdb
from numpy.testing import assert_allclose
from adam_core.coordinates.residuals import Residuals
from src.adam_core.propagator.adam_assist import (
    ASSISTPropagator,
    download_jpl_ephemeris_files,
)

def test_propagator_integration():
    """
    Propagate an orbit forward and backward in time to check for consistency
    """
    prop = ASSISTPropagator()
    edlu = query_sbdb(["Edlu"])
    # propagate forward 30 days
    thirty_days_forward = edlu.coordinates.time.add_days(30)
    forward = prop.propagate_orbits(edlu, thirty_days_forward, covariance=True)
    # propagate backward 30 days
    back_to_epoch = prop.propagate_orbits(forward, edlu.coordinates.time, covariance=True)
    # check that the original state and the propagated state are close
    assert_allclose(edlu.coordinates.values, back_to_epoch.coordinates.values, rtol=1e-10, atol=1e-10)
    residuals = Residuals.calculate(edlu.coordinates, back_to_epoch.coordinates)
    assert residuals.chi2[0].as_py() < 1e-10