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
            model=["nongrav", "yarkovsky"],
            solution_dimension=[7, 7],
            parameter_count=[2, 1],
            estimated_parameter_names=["A1,A2", "A2"],
            A1=[1.1e-13, None],
            A1_sigma=[None, None],
            A2=[-8.72e-14, -2.90e-14],
            A2_sigma=[None, None],
            A3=[None, 4.2e-15],
            A3_sigma=[None, None],
            DT=[None, None],
            DT_sigma=[None, None],
            R0=[None, None],
            R0_sigma=[None, None],
            ALN=[None, None],
            ALN_sigma=[None, None],
            NK=[None, None],
            NK_sigma=[None, None],
            NM=[None, None],
            NM_sigma=[None, None],
            NN=[None, None],
            NN_sigma=[None, None],
            AMRAT=[None, None],
            AMRAT_sigma=[None, None],
            RHO=[None, None],
            RHO_sigma=[None, None],
        )
    )

    particle_params = _extract_assist_particle_params(orbits)

    np.testing.assert_allclose(
        particle_params,
        np.array([1.1e-13, -8.72e-14, 0.0, 0.0, -2.90e-14, 4.2e-15]),
    )


def test_extract_assist_particle_params_rejects_unsupported_estimated_fields():
    orbits = make_orbits_with_nongrav(
        NonGravitationalParameters.from_kwargs(
            source=["SBDB", "SBDB"],
            model=["nongrav", "nongrav"],
            solution_dimension=[10, 7],
            parameter_count=[1, 1],
            estimated_parameter_names=["DT", "A2"],
            A1=[None, None],
            A1_sigma=[None, None],
            A2=[-8.72e-14, -2.90e-14],
            A2_sigma=[None, None],
            A3=[None, None],
            A3_sigma=[None, None],
            DT=[10.0, None],
            DT_sigma=[None, None],
            R0=[None, None],
            R0_sigma=[None, None],
            ALN=[None, None],
            ALN_sigma=[None, None],
            NK=[None, None],
            NK_sigma=[None, None],
            NM=[None, None],
            NM_sigma=[None, None],
            NN=[None, None],
            NN_sigma=[None, None],
            AMRAT=[None, None],
            AMRAT_sigma=[None, None],
            RHO=[None, None],
            RHO_sigma=[None, None],
        )
    )

    with pytest.raises(
        ValueError, match="Unsupported estimated parameters are present: DT"
    ):
        _extract_assist_particle_params(orbits)


def test_extract_assist_particle_params_allows_supported_values_with_fixed_metadata():
    orbits = make_orbits_with_nongrav(
        NonGravitationalParameters.from_kwargs(
            source=["SBDB", "NEOCC"],
            model=["nongrav", "yarkovsky"],
            solution_dimension=[8, 7],
            parameter_count=[2, 1],
            estimated_parameter_names=["A1,A2", "A2"],
            A1=[5.0e-13, None],
            A1_sigma=[None, None],
            A2=[-2.9e-14, -4.6e-14],
            A2_sigma=[None, None],
            A3=[None, None],
            A3_sigma=[None, None],
            DT=[None, None],
            DT_sigma=[None, None],
            R0=[1.0, None],
            R0_sigma=[None, None],
            ALN=[1.0, None],
            ALN_sigma=[None, None],
            NK=[0.0, None],
            NK_sigma=[None, None],
            NM=[2.0, None],
            NM_sigma=[None, None],
            NN=[None, None],
            NN_sigma=[None, None],
            AMRAT=[None, 0.0],
            AMRAT_sigma=[None, None],
            RHO=[None, None],
            RHO_sigma=[None, None],
        )
    )

    particle_params = _extract_assist_particle_params(orbits)

    np.testing.assert_allclose(
        particle_params,
        np.array([5.0e-13, -2.9e-14, 0.0, 0.0, -4.6e-14, 0.0]),
    )


def test_extract_assist_particle_params_rejects_metadata_without_A_values():
    orbits = make_orbits_with_nongrav(
        NonGravitationalParameters.from_kwargs(
            source=["NEOCC", "NEOCC"],
            model=["yarkovsky", "yarkovsky"],
            solution_dimension=[7, 7],
            parameter_count=[1, 1],
            estimated_parameter_names=["A2", "A2"],
            A1=[None, None],
            A1_sigma=[None, None],
            A2=[None, None],
            A2_sigma=[None, None],
            A3=[None, None],
            A3_sigma=[None, None],
            DT=[None, None],
            DT_sigma=[None, None],
            R0=[None, None],
            R0_sigma=[None, None],
            ALN=[None, None],
            ALN_sigma=[None, None],
            NK=[None, None],
            NK_sigma=[None, None],
            NM=[None, None],
            NM_sigma=[None, None],
            NN=[None, None],
            NN_sigma=[None, None],
            AMRAT=[None, None],
            AMRAT_sigma=[None, None],
            RHO=[None, None],
            RHO_sigma=[None, None],
        )
    )

    with pytest.raises(ValueError, match="metadata without usable A1/A2/A3"):
        _extract_assist_particle_params(orbits)


def test_configure_assist_non_gravitational_forces_appends_force_and_params():
    orbits = make_orbits_with_nongrav(
        NonGravitationalParameters.from_kwargs(
            source=["SBDB", "SBDB"],
            model=["nongrav", "nongrav"],
            solution_dimension=[7, 7],
            parameter_count=[1, 1],
            estimated_parameter_names=["A2", "A2"],
            A1=[None, None],
            A1_sigma=[None, None],
            A2=[-8.72e-14, -2.90e-14],
            A2_sigma=[None, None],
            A3=[None, None],
            A3_sigma=[None, None],
            DT=[None, None],
            DT_sigma=[None, None],
            R0=[None, None],
            R0_sigma=[None, None],
            ALN=[None, None],
            ALN_sigma=[None, None],
            NK=[None, None],
            NK_sigma=[None, None],
            NM=[None, None],
            NM_sigma=[None, None],
            NN=[None, None],
            NN_sigma=[None, None],
            AMRAT=[None, None],
            AMRAT_sigma=[None, None],
            RHO=[None, None],
            RHO_sigma=[None, None],
        )
    )
    extras = FakeExtras(["SUN", "PLANETS"])

    _configure_assist_non_gravitational_forces(extras, orbits)

    assert extras.forces == ["SUN", "PLANETS", "NON_GRAVITATIONAL"]
    np.testing.assert_allclose(
        extras.particle_params,
        [0.0, -8.72e-14, 0.0, 0.0, -2.90e-14, 0.0],
    )


def test_extract_assist_particle_params_ignores_source_metadata_without_estimates():
    # Catalogs commonly stamp source/dimension metadata on every row (NEOCC
    # always emits an LSP line; mpcq stamps its source). Rows without
    # estimated parameters and without A1/A2/A3 values are gravity-only and
    # must not raise.
    orbits = make_orbits_with_nongrav(
        NonGravitationalParameters.from_kwargs(
            source=["NEOCC", "MPCQ"],
            model=[None, None],
            solution_dimension=[6, None],
            parameter_count=[None, None],
            estimated_parameter_names=[None, None],
            A1=[None, None],
            A1_sigma=[None, None],
            A2=[None, None],
            A2_sigma=[None, None],
            A3=[None, None],
            A3_sigma=[None, None],
            DT=[None, None],
            DT_sigma=[None, None],
            R0=[None, None],
            R0_sigma=[None, None],
            ALN=[None, None],
            ALN_sigma=[None, None],
            NK=[None, None],
            NK_sigma=[None, None],
            NM=[None, None],
            NM_sigma=[None, None],
            NN=[None, None],
            NN_sigma=[None, None],
            AMRAT=[None, None],
            AMRAT_sigma=[None, None],
            RHO=[None, None],
            RHO_sigma=[None, None],
        )
    )

    assert _extract_assist_particle_params(orbits) is None


def test_extract_assist_particle_params_rejects_non_default_marsden_constants():
    # A cometary solution fit with the Marsden g(r) constants is not valid
    # under ASSIST's g(r) = (1 au / r)^2 force law.
    orbits = make_orbits_with_nongrav(
        NonGravitationalParameters.from_kwargs(
            source=["SBDB", "SBDB"],
            model=["nongrav", "cometary"],
            solution_dimension=[7, 9],
            parameter_count=[1, 3],
            estimated_parameter_names=["A2", "A1,A2,A3"],
            A1=[None, 1.04e-9],
            A1_sigma=[None, None],
            A2=[-2.9e-14, -6.7e-11],
            A2_sigma=[None, None],
            A3=[None, 2.9e-10],
            A3_sigma=[None, None],
            DT=[None, None],
            DT_sigma=[None, None],
            R0=[None, 2.808],
            R0_sigma=[None, None],
            ALN=[None, 0.1112620426],
            ALN_sigma=[None, None],
            NK=[None, 4.6142],
            NK_sigma=[None, None],
            NM=[None, 2.15],
            NM_sigma=[None, None],
            NN=[None, 5.093],
            NN_sigma=[None, None],
            AMRAT=[None, None],
            AMRAT_sigma=[None, None],
            RHO=[None, None],
            RHO_sigma=[None, None],
        )
    )

    with pytest.raises(ValueError, match="not valid under ASSIST"):
        _extract_assist_particle_params(orbits)
