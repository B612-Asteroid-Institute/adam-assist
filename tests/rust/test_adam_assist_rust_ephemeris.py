from __future__ import annotations

import numpy as np
from adam_core.coordinates.cartesian import CartesianCoordinates
from adam_core.coordinates.origin import Origin
from adam_core.observers.observers import Observers
from adam_core.orbits.orbits import Orbits
from adam_core.time import Timestamp

from adam_assist import ASSISTPropagator as RustASSISTPropagator

# Frozen pinned-legacy parity: range/angles stay below 1.4e-10 while
# spherical velocity components have a stable 2.35e-8 implementation floor.
EPHEMERIS_STATE_ATOL = 3.0e-8


def _orbits() -> Orbits:
    coordinates = CartesianCoordinates.from_kwargs(
        x=[1.05, 1.35],
        y=[0.02, -0.08],
        z=[0.01, 0.03],
        vx=[-0.0005, 0.001],
        vy=[0.0165, 0.014],
        vz=[0.0002, -0.0001],
        time=Timestamp.from_mjd([60000.0, 60000.0], scale="tdb"),
        origin=Origin.from_kwargs(code=["SUN", "SUN"]),
        frame="ecliptic",
    )
    return Orbits.from_kwargs(
        orbit_id=["fixture-a", "fixture-b"],
        object_id=["fixture-a", "fixture-b"],
        coordinates=coordinates,
    )


def _sorted_values(ephemeris) -> np.ndarray:
    ordered = ephemeris.sort_by(
        [
            "orbit_id",
            "coordinates.time.days",
            "coordinates.time.nanos",
            "coordinates.origin.code",
        ]
    )
    return np.asarray(ordered.coordinates.values, dtype=np.float64)


def _sorted_covariance(ephemeris) -> np.ndarray:
    ordered = ephemeris.sort_by(
        [
            "orbit_id",
            "coordinates.time.days",
            "coordinates.time.nanos",
            "coordinates.origin.code",
        ]
    )
    return ordered.coordinates.covariance.to_matrix()


def _sorted_column(ephemeris, name: str) -> np.ndarray:
    ordered = ephemeris.sort_by(
        [
            "orbit_id",
            "coordinates.time.days",
            "coordinates.time.nanos",
            "coordinates.origin.code",
        ]
    )
    return ordered.column(name).to_numpy(zero_copy_only=False)


def test_generate_ephemeris_matches_frozen_adam_assist_regression(
    frozen_assist_regression,
) -> None:
    orbits = _orbits()
    times = Timestamp.from_mjd([60000.5, 60001.0], scale="utc")
    observers = Observers.from_code("X05", times)

    actual = RustASSISTPropagator().generate_ephemeris(
        orbits,
        observers,
        predict_magnitudes=False,
        predict_phase_angle=False,
        max_processes=1,
    )

    actual_values = _sorted_values(actual)
    expected_values = frozen_assist_regression["ephemeris_basic_state"]
    assert actual_values.shape == expected_values.shape
    np.testing.assert_allclose(
        actual_values, expected_values, atol=EPHEMERIS_STATE_ATOL, rtol=0
    )


def test_generate_ephemeris_mixed_observers_and_photometry_matches_frozen(
    frozen_assist_regression,
) -> None:
    """Mixed observer, light-time, phase-angle, and photometry regression."""
    import quivr as qv
    from adam_core.orbits.orbits import PhysicalParameters

    base = _orbits()
    orbits = Orbits.from_kwargs(
        orbit_id=base.orbit_id,
        object_id=base.object_id,
        coordinates=base.coordinates,
        physical_parameters=PhysicalParameters.from_kwargs(
            H_v=[18.2, 20.1],
            G=[0.15, 0.15],
        ),
    )
    times = Timestamp.from_mjd([60000.5, 60001.0], scale="utc")
    observers = qv.concatenate(
        [Observers.from_code("X05", times), Observers.from_code("500", times)]
    )

    actual = RustASSISTPropagator().generate_ephemeris(
        orbits,
        observers,
        predict_magnitudes=True,
        predict_phase_angle=True,
        max_processes=1,
    )

    np.testing.assert_allclose(
        _sorted_values(actual),
        frozen_assist_regression["ephemeris_photometry_state"],
        atol=EPHEMERIS_STATE_ATOL,
        rtol=0,
    )
    np.testing.assert_allclose(
        _sorted_column(actual, "light_time"),
        frozen_assist_regression["ephemeris_photometry_light_time"],
        atol=1.0e-12,
        rtol=0,
    )
    np.testing.assert_allclose(
        _sorted_column(actual, "predicted_magnitude_v"),
        frozen_assist_regression["ephemeris_photometry_predicted_magnitude_v"],
        atol=1.0e-9,
        rtol=0,
    )
    np.testing.assert_allclose(
        _sorted_column(actual, "alpha"),
        frozen_assist_regression["ephemeris_photometry_alpha"],
        atol=1.0e-9,
        rtol=0,
    )


def test_generate_ephemeris_covariance_matches_frozen_adam_assist(
    frozen_assist_regression,
) -> None:
    """Sigma-point covariance ephemeris remains within accepted parity floors."""
    from adam_core.coordinates.covariances import CoordinateCovariances

    base = _orbits()
    sigmas = np.tile(
        np.array([1.0e-8, 1.0e-8, 1.0e-8, 1.0e-10, 1.0e-10, 1.0e-10]),
        (len(base), 1),
    )
    coordinates = base.coordinates.set_column(
        "covariance", CoordinateCovariances.from_sigmas(sigmas)
    )
    orbits = Orbits.from_kwargs(
        orbit_id=base.orbit_id,
        object_id=base.object_id,
        coordinates=coordinates,
    )
    times = Timestamp.from_mjd([60000.5, 60001.0], scale="utc")
    observers = Observers.from_code("X05", times)

    actual = RustASSISTPropagator().generate_ephemeris(
        orbits,
        observers,
        covariance=True,
        covariance_method="sigma-point",
        max_processes=1,
    )

    np.testing.assert_allclose(
        _sorted_values(actual),
        frozen_assist_regression["ephemeris_covariance_state"],
        atol=EPHEMERIS_STATE_ATOL,
        rtol=0,
    )
    assert actual.coordinates.time.scale == "utc"

    actual_cov = _sorted_covariance(actual)
    expected_cov = frozen_assist_regression["ephemeris_covariance_covariance"]
    assert actual_cov.shape == expected_cov.shape
    assert not np.all(
        np.isnan(actual_cov)
    ), "rust covariance ephemeris produced an all-NaN covariance"
    np.testing.assert_allclose(actual_cov, expected_cov, atol=1.0e-16, rtol=1.0e-4)
