import numpy as np
from adam_core.coordinates.cartesian import CartesianCoordinates
from adam_core.coordinates.origin import Origin
from adam_core.orbits import Orbits
from adam_core.orbits.non_gravitational_parameters import NonGravitationalParameters
from adam_core.time import Timestamp

from adam_assist.propagator import (
    _configure_assist_non_gravitational_forces,
    _extract_assist_particle_params,
)


class FakeExtras:
    def __init__(self, forces):
        self.forces = list(forces)
        self.particle_params = None


def make_orbits_with_nongrav(nongrav: NonGravitationalParameters) -> Orbits:
    return Orbits.from_kwargs(
        orbit_id=["o1", "o2"],
        object_id=["o1", "o2"],
        non_gravitational_parameters=nongrav,
        coordinates=CartesianCoordinates.from_kwargs(
            x=[1.0, 1.2],
            y=[0.0, 0.1],
            z=[0.0, 0.0],
            vx=[0.0, 0.0],
            vy=[0.017, 0.015],
            vz=[0.0, 0.0],
            time=Timestamp.from_mjd([60000.0, 60000.0], scale="tdb"),
            origin=Origin.from_kwargs(code=["SUN", "SUN"]),
            frame="ecliptic",
        ),
    )


def test_extract_assist_particle_params_flattens_A1_A2_A3():
    orbits = make_orbits_with_nongrav(
        NonGravitationalParameters.from_kwargs(
            source=["SBDB", "NEOCC"],
            A1=[1.1e-13, None],
            A2=[-8.72e-14, -2.90e-14],
            A3=[None, 4.2e-15],
        )
    )

    particle_params = _extract_assist_particle_params(orbits)

    np.testing.assert_allclose(
        particle_params,
        np.array([1.1e-13, -8.72e-14, 0.0, 0.0, -2.90e-14, 4.2e-15]),
    )


def test_extract_assist_particle_params_treats_null_values_as_zero():
    orbits = make_orbits_with_nongrav(
        NonGravitationalParameters.from_kwargs(
            source=["SBDB", "NEOCC"],
            A1=[5.0e-13, None],
            A2=[-2.9e-14, -4.6e-14],
            A3=[None, None],
        )
    )

    particle_params = _extract_assist_particle_params(orbits)

    np.testing.assert_allclose(
        particle_params,
        np.array([5.0e-13, -2.9e-14, 0.0, 0.0, -4.6e-14, 0.0]),
    )


def test_extract_assist_particle_params_returns_none_without_values():
    # A source stamp without any A1/A2/A3 values means no non-grav solution:
    # no particle params and no NON_GRAVITATIONAL force should be configured.
    orbits = make_orbits_with_nongrav(
        NonGravitationalParameters.from_kwargs(
            source=["NEOCC", None],
            A1=[None, None],
            A2=[None, None],
            A3=[None, None],
        )
    )

    assert _extract_assist_particle_params(orbits) is None

    extras = FakeExtras(["SUN", "PLANETS"])
    _configure_assist_non_gravitational_forces(extras, orbits)
    assert extras.forces == ["SUN", "PLANETS"]
    assert extras.particle_params is None


def test_configure_assist_non_gravitational_forces_appends_force_and_params():
    orbits = make_orbits_with_nongrav(
        NonGravitationalParameters.from_kwargs(
            source=["SBDB", "SBDB"],
            A1=[None, None],
            A2=[-8.72e-14, -2.90e-14],
            A3=[None, None],
        )
    )
    extras = FakeExtras(["SUN", "PLANETS"])

    _configure_assist_non_gravitational_forces(extras, orbits)

    assert extras.forces == ["SUN", "PLANETS", "NON_GRAVITATIONAL"]
    np.testing.assert_allclose(
        extras.particle_params,
        [0.0, -8.72e-14, 0.0, 0.0, -2.90e-14, 0.0],
    )
