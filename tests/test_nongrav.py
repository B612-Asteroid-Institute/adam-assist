import numpy as np
import pytest
import quivr as qv
from adam_core.coordinates.cartesian import CartesianCoordinates
from adam_core.coordinates.origin import Origin
from adam_core.dynamics.impacts import CollisionConditions
from adam_core.orbits import Orbits
from adam_core.orbits.non_gravitational_parameters import NonGravitationalParameters
from adam_core.time import Timestamp

from adam_assist import ASSISTPropagator

COMET_CONSTANTS = {
    "ALN": 0.1112620426,
    "NK": 4.6142,
    "NM": 2.15,
    "NN": 5.093,
    "R0": 2.808,
}


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


def _propagate(orbits: Orbits, mjd: float = 60010.0) -> Orbits:
    return ASSISTPropagator().propagate_orbits(
        orbits, Timestamp.from_mjd([mjd], scale="tdb")
    )


def test_include_nongrav_false_matches_explicitly_stripped_gravity_only() -> None:
    active = make_orbits_with_nongrav(
        NonGravitationalParameters.from_kwargs(
            source=["SBDB", "NEOCC"],
            A1=[5.0e-9, None],
            A2=[-2.9e-9, -4.6e-9],
            A3=[None, 4.2e-10],
        )
    )
    target = Timestamp.from_mjd([60010.0], scale="tdb")
    propagator = ASSISTPropagator()
    actual = propagator.propagate_orbits(active, target, include_nongrav=False)
    expected = propagator.propagate_orbits(
        active.without_non_gravitational_parameters(), target
    )

    np.testing.assert_array_equal(
        actual.coordinates.values, expected.coordinates.values
    )
    assert not actual.has_non_gravitational_parameters()
    assert not actual.coordinates.covariance.has_nongrav_block()


def test_non_gravitational_coefficients_change_trajectory_and_survive_output():
    active = make_orbits_with_nongrav(
        NonGravitationalParameters.from_kwargs(
            source=["SBDB", "NEOCC"],
            A1=[5.0e-9, None],
            A2=[-2.9e-9, -4.6e-9],
            A3=[None, 4.2e-10],
        )
    )
    gravity_only = active.without_non_gravitational_parameters()

    propagated = _propagate(active)
    control = _propagate(gravity_only)

    assert (
        np.max(np.abs(propagated.coordinates.values - control.coordinates.values))
        > 1e-8
    )
    assert propagated.non_gravitational_parameters.source.to_pylist() == [
        "SBDB",
        "NEOCC",
    ]
    assert propagated.non_gravitational_parameters.A2.to_pylist() == [
        -2.9e-9,
        -4.6e-9,
    ]


def test_null_and_zero_accelerations_are_force_free():
    nulls = make_orbits_with_nongrav(NonGravitationalParameters.nulls(2))
    zeros = make_orbits_with_nongrav(
        NonGravitationalParameters.from_kwargs(
            source=["SBDB", "NEOCC"],
            A1=[0.0, None],
            A2=[None, 0.0],
            A3=[0.0, 0.0],
        )
    )
    gravity_only = nulls.without_non_gravitational_parameters()

    expected = _propagate(gravity_only).coordinates.values
    np.testing.assert_array_equal(_propagate(nulls).coordinates.values, expected)
    np.testing.assert_array_equal(_propagate(zeros).coordinates.values, expected)


def test_explicit_marsden_constants_change_force_law():
    default = make_orbits_with_nongrav(
        NonGravitationalParameters.from_kwargs(
            source=["SBDB", "SBDB"],
            A1=[1.07e-7, 8.9e-8],
            A2=[-3.6e-9, -2.1e-9],
            A3=[2.5e-9, None],
        )
    )
    comet = default.set_column(
        "non_gravitational_parameters",
        NonGravitationalParameters.from_kwargs(
            source=["SBDB", "SBDB"],
            A1=[1.07e-7, 8.9e-8],
            A2=[-3.6e-9, -2.1e-9],
            A3=[2.5e-9, None],
            **{name: [value, value] for name, value in COMET_CONSTANTS.items()},
        ),
    )

    delta = (
        _propagate(comet).coordinates.values - _propagate(default).coordinates.values
    )
    assert np.max(np.abs(delta)) > 1e-8


def test_same_model_nongrav_batch_matches_solo_propagations():
    batched_input = make_orbits_with_nongrav(
        NonGravitationalParameters.from_kwargs(
            source=["SBDB", "NEOCC"],
            A1=[5.0e-9, None],
            A2=[-2.9e-9, -4.6e-9],
            A3=[None, 4.2e-10],
        )
    )
    batched = _propagate(batched_input, mjd=60150.0)
    solos = qv.concatenate(
        [_propagate(batched_input[i : i + 1], mjd=60150.0) for i in range(2)]
    ).sort_by(["orbit_id", "coordinates.time.days", "coordinates.time.nanos"])

    position_residual_km = (
        np.max(np.abs(batched.coordinates.r - solos.coordinates.r)) * 149597870.7
    )
    assert position_residual_km < 1e-4


def test_mixed_marsden_batch_matches_solo_propagations():
    mixed = make_orbits_with_nongrav(
        NonGravitationalParameters.from_kwargs(
            source=["SBDB", "SBDB"],
            A1=[1.07e-9, 5.0e-10],
            A2=[-3.6e-10, -2.9e-10],
            A3=[None, None],
            ALN=[COMET_CONSTANTS["ALN"], None],
            NK=[COMET_CONSTANTS["NK"], None],
            NM=[COMET_CONSTANTS["NM"], None],
            NN=[COMET_CONSTANTS["NN"], None],
            R0=[COMET_CONSTANTS["R0"], None],
        )
    )
    batched = _propagate(mixed)
    solos = qv.concatenate([_propagate(mixed[i : i + 1]) for i in range(len(mixed))])
    solos = solos.sort_by(
        ["orbit_id", "coordinates.time.days", "coordinates.time.nanos"]
    )
    np.testing.assert_allclose(
        batched.coordinates.values,
        solos.coordinates.values,
        rtol=0,
        atol=2e-15,
    )


def test_collision_detection_groups_mixed_marsden_models_and_restores_rows():
    mixed = make_orbits_with_nongrav(
        NonGravitationalParameters.from_kwargs(
            source=["SBDB", "SBDB"],
            A1=[1.07e-9, 5.0e-10],
            A2=[-3.6e-10, -2.9e-10],
            A3=[None, None],
            ALN=[COMET_CONSTANTS["ALN"], None],
            NK=[COMET_CONSTANTS["NK"], None],
            NM=[COMET_CONSTANTS["NM"], None],
            NN=[COMET_CONSTANTS["NN"], None],
            R0=[COMET_CONSTANTS["R0"], None],
        )
    )
    conditions = CollisionConditions.from_kwargs(
        condition_id=["Earth"],
        collision_object=Origin.from_kwargs(code=["EARTH"]),
        collision_distance=[1.0],
        stopping_condition=[False],
    )
    propagator = ASSISTPropagator()
    batched, impacts = propagator._detect_collisions(mixed, 2, conditions)
    solo_results = []
    for index in range(len(mixed)):
        result, solo_impacts = propagator._detect_collisions(
            mixed[index : index + 1], 2, conditions
        )
        assert len(solo_impacts) == 0
        solo_results.append(result)
    solos = qv.concatenate(solo_results).sort_by(["orbit_id"])
    batched = batched.sort_by(["orbit_id"])

    assert len(impacts) == 0
    np.testing.assert_allclose(
        batched.coordinates.values,
        solos.coordinates.values,
        rtol=0,
        atol=2e-15,
    )
    np.testing.assert_array_equal(
        batched.coordinates.time.mjd().to_numpy(),
        solos.coordinates.time.mjd().to_numpy(),
    )


def test_partial_constants_are_rejected_for_active_force():
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
    with pytest.raises(ValueError, match="Partially-specified Marsden"):
        _propagate(orbits)


def test_partial_constants_are_ignored_on_force_free_rows():
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
    assert len(_propagate(orbits)) == 2


def test_non_finite_acceleration_is_rejected():
    orbits = make_orbits_with_nongrav(
        NonGravitationalParameters.from_kwargs(
            source=["SBDB", "SBDB"],
            A1=[float("nan"), 5.0e-13],
            A2=[None, -2.9e-14],
            A3=[None, None],
        )
    )
    with pytest.raises(ValueError, match="Non-finite"):
        _propagate(orbits)


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
def test_invalid_marsden_constants_are_rejected(overrides):
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
        _propagate(orbits)


@pytest.mark.parametrize("bad_nn", [float("nan"), float("inf")])
def test_non_finite_nn_is_rejected_before_nk_zero_canonicalization(bad_nn):
    orbits = make_orbits_with_nongrav(
        NonGravitationalParameters.from_kwargs(
            source=["SBDB", "SBDB"],
            A1=[None, None],
            A2=[-2.9e-14, -1.1e-14],
            A3=[None, None],
            ALN=[1.0, 1.0],
            NK=[0.0, 0.0],
            NM=[2.0, 2.0],
            NN=[bad_nn, bad_nn],
            R0=[1.0, 1.0],
        )
    )
    with pytest.raises(ValueError, match="Invalid Marsden"):
        _propagate(orbits)
