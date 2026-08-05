import numpy as np
import pytest
from adam_core.coordinates.cartesian import CartesianCoordinates
from adam_core.coordinates.origin import Origin
from adam_core.orbits import Orbits
from adam_core.orbits.non_gravitational_parameters import NonGravitationalParameters
from adam_core.time import Timestamp

from adam_assist.propagator import (
    _configure_assist_non_gravitational_forces,
    _extract_assist_particle_params,
    _partition_by_marsden_constants,
)

COMET_CONSTANTS = {
    "ALN": 0.1112620426,
    "NK": 4.6142,
    "NM": 2.15,
    "NN": 5.093,
    "R0": 2.808,
}


class FakeExtras:
    def __init__(self, forces):
        self.forces = list(forces)
        self.particle_params = None
        # assist.Extras ships with the asteroid-convention g(r) defaults.
        self.alpha = 1.0
        self.nk = 0.0
        self.nm = 2.0
        self.nn = 5.093
        self.r0 = 1.0


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
    # Null constants leave the asteroid-convention defaults in place.
    assert (extras.alpha, extras.nk, extras.nm, extras.nn, extras.r0) == (
        1.0,
        0.0,
        2.0,
        5.093,
        1.0,
    )


def test_configure_assist_non_gravitational_forces_sets_marsden_constants():
    orbits = make_orbits_with_nongrav(
        NonGravitationalParameters.from_kwargs(
            source=["SBDB", "SBDB"],
            A1=[1.07e-9, 8.9e-10],
            A2=[-3.6e-11, -2.1e-11],
            A3=[2.5e-11, None],
            **{name: [value, value] for name, value in COMET_CONSTANTS.items()},
        )
    )
    extras = FakeExtras(["SUN", "PLANETS"])

    _configure_assist_non_gravitational_forces(extras, orbits)

    assert extras.forces == ["SUN", "PLANETS", "NON_GRAVITATIONAL"]
    np.testing.assert_allclose(extras.alpha, COMET_CONSTANTS["ALN"])
    np.testing.assert_allclose(extras.nk, COMET_CONSTANTS["NK"])
    np.testing.assert_allclose(extras.nm, COMET_CONSTANTS["NM"])
    np.testing.assert_allclose(extras.nn, COMET_CONSTANTS["NN"])
    np.testing.assert_allclose(extras.r0, COMET_CONSTANTS["R0"])


def test_configure_assist_non_gravitational_forces_rejects_mixed_constants():
    orbits = make_orbits_with_nongrav(
        NonGravitationalParameters.from_kwargs(
            source=["SBDB", "SBDB"],
            A1=[1.07e-9, None],
            A2=[-3.6e-11, -2.9e-14],
            A3=[None, None],
            ALN=[COMET_CONSTANTS["ALN"], None],
            NK=[COMET_CONSTANTS["NK"], None],
            NM=[COMET_CONSTANTS["NM"], None],
            NN=[COMET_CONSTANTS["NN"], None],
            R0=[COMET_CONSTANTS["R0"], None],
        )
    )
    extras = FakeExtras(["SUN", "PLANETS"])

    with pytest.raises(ValueError, match="Marsden"):
        _configure_assist_non_gravitational_forces(extras, orbits)


def test_configure_ignores_constants_on_rows_without_accelerations():
    # A row carrying comet constants but no A-values exerts no force, so its
    # constants must not conflict with (or override) the active rows'.
    orbits = make_orbits_with_nongrav(
        NonGravitationalParameters.from_kwargs(
            source=["SBDB", "SBDB"],
            A1=[None, 5.0e-13],
            A2=[None, -2.9e-14],
            A3=[None, None],
            ALN=[COMET_CONSTANTS["ALN"], None],
            NK=[COMET_CONSTANTS["NK"], None],
            NM=[COMET_CONSTANTS["NM"], None],
            NN=[COMET_CONSTANTS["NN"], None],
            R0=[COMET_CONSTANTS["R0"], None],
        )
    )
    extras = FakeExtras(["SUN", "PLANETS"])

    _configure_assist_non_gravitational_forces(extras, orbits)

    assert (extras.alpha, extras.nk, extras.nm, extras.nn, extras.r0) == (
        1.0,
        0.0,
        2.0,
        5.093,
        1.0,
    )


def test_partition_by_marsden_constants_splits_mixed_tuples():
    orbits = make_orbits_with_nongrav(
        NonGravitationalParameters.from_kwargs(
            source=["SBDB", "SBDB"],
            A1=[1.07e-9, 5.0e-13],
            A2=[-3.6e-11, -2.9e-14],
            A3=[None, None],
            ALN=[COMET_CONSTANTS["ALN"], None],
            NK=[COMET_CONSTANTS["NK"], None],
            NM=[COMET_CONSTANTS["NM"], None],
            NN=[COMET_CONSTANTS["NN"], None],
            R0=[COMET_CONSTANTS["R0"], None],
        )
    )

    partitions = _partition_by_marsden_constants(orbits)

    assert len(partitions) == 2
    assert partitions[0].orbit_id.to_pylist() == ["o1"]
    assert partitions[1].orbit_id.to_pylist() == ["o2"]
    # Each partition now configures cleanly.
    for partition in partitions:
        _configure_assist_non_gravitational_forces(
            FakeExtras(["SUN", "PLANETS"]), partition
        )


def test_configure_rejects_non_finite_a_values():
    orbits = make_orbits_with_nongrav(
        NonGravitationalParameters.from_kwargs(
            source=["SBDB", "SBDB"],
            A1=[float("nan"), 5.0e-13],
            A2=[None, -2.9e-14],
            A3=[None, None],
        )
    )

    with pytest.raises(ValueError, match="Non-finite"):
        _configure_assist_non_gravitational_forces(FakeExtras(["SUN"]), orbits)


def test_configure_rejects_partial_marsden_tuples():
    orbits = make_orbits_with_nongrav(
        NonGravitationalParameters.from_kwargs(
            source=["SBDB", "SBDB"],
            A1=[1.07e-9, None],
            A2=[None, -2.9e-14],
            A3=[None, None],
            ALN=[COMET_CONSTANTS["ALN"], None],
            NK=[COMET_CONSTANTS["NK"], None],
            NM=[None, None],
            NN=[None, None],
            R0=[COMET_CONSTANTS["R0"], None],
        )
    )

    with pytest.raises(ValueError, match="Partially-specified"):
        _configure_assist_non_gravitational_forces(FakeExtras(["SUN"]), orbits)


@pytest.mark.parametrize(
    "overrides",
    [
        {"ALN": 0.0},
        {"ALN": -1.0},
        {"R0": 0.0},
        {"R0": -2.808},
        {"NM": float("nan")},
        {"NN": float("inf")},
    ],
)
def test_configure_rejects_invalid_marsden_constants(overrides):
    constants = {**COMET_CONSTANTS, **overrides}
    orbits = make_orbits_with_nongrav(
        NonGravitationalParameters.from_kwargs(
            source=["SBDB", "SBDB"],
            A1=[1.07e-9, 8.9e-10],
            A2=[-3.6e-11, -2.1e-11],
            A3=[None, None],
            **{name: [value, value] for name, value in constants.items()},
        )
    )

    with pytest.raises(ValueError, match="Invalid Marsden"):
        _configure_assist_non_gravitational_forces(FakeExtras(["SUN"]), orbits)


def test_partial_constants_allowed_on_force_free_rows():
    # A row without accelerations exerts no force: its (partial) constants
    # are never applied and must not be rejected.
    orbits = make_orbits_with_nongrav(
        NonGravitationalParameters.from_kwargs(
            source=["SBDB", "SBDB"],
            A1=[None, 5.0e-13],
            A2=[None, -2.9e-14],
            A3=[None, None],
            ALN=[COMET_CONSTANTS["ALN"], None],
            NK=[None, None],
            NM=[None, None],
            NN=[None, None],
            R0=[None, None],
        )
    )
    extras = FakeExtras(["SUN", "PLANETS"])

    _configure_assist_non_gravitational_forces(extras, orbits)

    assert extras.forces == ["SUN", "PLANETS", "NON_GRAVITATIONAL"]


def test_degenerate_nk_zero_law_is_canonicalized():
    # With NK = 0 the (1 + (r/R0)^NN)^-NK factor is identically 1, so an
    # explicit inverse-square tuple stored with NN = 0 (e.g. Phaethon's) is
    # the same force law as null constants and must share a simulation.
    orbits = make_orbits_with_nongrav(
        NonGravitationalParameters.from_kwargs(
            source=["SBDB", "SBDB"],
            A1=[None, 5.0e-13],
            A2=[-2.9e-14, -1.1e-14],
            A3=[None, None],
            ALN=[1.0, None],
            NK=[0.0, None],
            NM=[2.0, None],
            NN=[0.0, None],
            R0=[1.0, None],
        )
    )

    partitions = _partition_by_marsden_constants(orbits)
    assert len(partitions) == 1

    extras = FakeExtras(["SUN"])
    _configure_assist_non_gravitational_forces(extras, orbits)
    assert (extras.alpha, extras.nk, extras.nm, extras.nn, extras.r0) == (
        1.0,
        0.0,
        2.0,
        5.093,
        1.0,
    )


def test_partition_by_marsden_constants_keeps_uniform_batch_whole():
    # Explicit asteroid-convention constants match null constants and
    # force-free rows: numerically identical tuples must not split.
    orbits = make_orbits_with_nongrav(
        NonGravitationalParameters.from_kwargs(
            source=["SBDB", "SBDB"],
            A1=[5.0e-13, None],
            A2=[-2.9e-14, None],
            A3=[None, None],
            ALN=[1.0, None],
            NK=[0.0, None],
            NM=[2.0, None],
            NN=[5.093, None],
            R0=[1.0, None],
        )
    )

    partitions = _partition_by_marsden_constants(orbits)

    assert len(partitions) == 1
    assert len(partitions[0]) == 2
