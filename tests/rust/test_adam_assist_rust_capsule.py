"""Capsule conformance and adam-core-owned OD integration.

adam-assist supplies only ASSIST propagation and STM capabilities. The public
fit is owned and orchestrated by adam-core, which consumes the private native
propagator capsule without provider-specific OD work units.
"""

import numpy as np
import pytest
from adam_core.coordinates import (
    CartesianCoordinates,
    CoordinateCovariances,
    Origin,
    SphericalCoordinates,
)
from adam_core.observers import Observers
from adam_core.orbit_determination import fit_least_squares
from adam_core.orbit_determination.evaluate import OrbitDeterminationObservations
from adam_core.orbits import Orbits
from adam_core.time import Timestamp

from adam_assist import ASSISTPropagator as RustASSISTPropagator

EPOCH_MJD = 60000.0
TRUTH_STATE = np.array([1.2, 0.1, 0.05, -0.002, 0.016, 0.001])


def _make_orbit(state: np.ndarray, orbit_id: str) -> Orbits:
    return Orbits.from_kwargs(
        orbit_id=[orbit_id],
        object_id=[orbit_id],
        coordinates=CartesianCoordinates.from_kwargs(
            x=[state[0]],
            y=[state[1]],
            z=[state[2]],
            vx=[state[3]],
            vy=[state[4]],
            vz=[state[5]],
            time=Timestamp.from_mjd([EPOCH_MJD], scale="tdb"),
            origin=Origin.from_kwargs(code=["SUN"]),
            frame="ecliptic",
        ),
    )


@pytest.fixture(scope="module")
def od_problem(frozen_assist_regression):
    """Noise-free astrometry frozen from accepted legacy ASSIST ephemeris."""
    n = 8
    times = Timestamp.from_mjd(
        [EPOCH_MJD + 2.0 + i * 3.0 for i in range(n)], scale="utc"
    )
    observers = Observers.from_code("X05", times)
    values = frozen_assist_regression["od_observed_state"]
    arcsec = (1.0 / 3600.0) ** 2
    cov = np.tile(np.diag([1.0, arcsec, arcsec, 1.0, 1.0, 1.0]), (n, 1, 1))
    observed = SphericalCoordinates.from_kwargs(
        rho=values[:, 0],
        lon=values[:, 1],
        lat=values[:, 2],
        vrho=values[:, 3],
        vlon=values[:, 4],
        vlat=values[:, 5],
        time=Timestamp.from_mjd(
            frozen_assist_regression["od_observed_time_mjd"], scale="utc"
        ),
        origin=Origin.from_kwargs(
            code=frozen_assist_regression["od_observed_origin"].tolist()
        ),
        frame="equatorial",
        covariance=CoordinateCovariances.from_matrix(cov),
    )
    observations = OrbitDeterminationObservations.from_kwargs(
        id=[f"obs-{i}" for i in range(n)],
        coordinates=observed,
        observers=observers,
    )
    initial = _make_orbit(
        TRUTH_STATE + np.array([1e-3, -1e-3, 5e-4, 1e-5, -1e-5, 1e-5]), "fit"
    )
    return observations, initial



def test_assist_native_backend_is_capsule_only() -> None:
    propagator = RustASSISTPropagator()
    backend = propagator.native_backend

    assert backend is not None
    assert backend.name == "adam_assist"
    assert backend.work_units == frozenset({"propagator_capsule"})
    assert type(propagator.propagator_capsule()).__name__ == "PyCapsule"
    for method in (
        "fit_least_squares",
        "fit_least_squares_evaluated",
        "od_fit",
        "initial_orbit_determination",
        "vallado_least_squares",
    ):
        assert not hasattr(propagator, method)


def test_adam_core_fit_uses_assist_capsule(od_problem) -> None:
    observations, initial = od_problem

    fitted_orbit, fitted_members = fit_least_squares(
        initial, observations, RustASSISTPropagator()
    )

    assert fitted_orbit.success[0].as_py()
    assert len(fitted_members) == len(observations)
    assert fitted_members.solution.to_pylist().count(True) == len(observations)
    np.testing.assert_allclose(
        fitted_orbit.coordinates.values[0], TRUTH_STATE, rtol=0, atol=1e-6
    )
    covariance = fitted_orbit.coordinates.covariance.to_matrix()[0]
    assert np.isfinite(covariance).all()
    assert (np.diag(covariance) > 0).all()
