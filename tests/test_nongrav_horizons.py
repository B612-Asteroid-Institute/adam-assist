"""
Frozen-fixture regressions for the non-gravitational force law against JPL
Horizons trajectories.

Each case starts from the Horizons state at a central epoch, attaches the
cited SBDB non-gravitational solution, propagates to Horizons reference
vectors at +/- [300, 150, 30, 0] days, and asserts both an absolute accuracy
bound and a large improvement over an identical gravity-only propagation --
so an accidentally disabled force path cannot pass.

The Horizons vectors are checked in (tests/data/nongrav_horizons_vectors.
parquet, built by tests/data/make_nongrav_horizons_fixtures.py, retrieved
2026-08-05) so the suite is deterministic; JPL solution updates require
refreshing the fixture and the parameters below together.

Covered force laws:
- 99942 Apophis (JPL soln 220): inverse-square asteroid convention.
- C/2022 E3 ZTF (JPL soln 82): standard comet Marsden law.
- 1I/'Oumuamua (JPL soln 16): custom (ALN, NK, NM, NN, R0) shape tuple
  (Micheli et al. 2018).

67P/81P-class comets whose JPL solutions estimate DT are deliberately not
accuracy gates here: this stack drops DT with a warning and retains km-level
model error through perihelion for such objects.
"""

import os

import numpy as np
import pyarrow as pa
import pytest
import quivr as qv
from adam_core.orbits import Orbits
from adam_core.orbits.non_gravitational_parameters import NonGravitationalParameters
from adam_core.time import Timestamp

from adam_assist import ASSISTPropagator

FIXTURE_PATH = os.path.join(
    os.path.dirname(__file__), "data", "nongrav_horizons_vectors.parquet"
)

KM_PER_AU = 149597870.7

# SBDB non-gravitational solutions frozen 2026-08-05 (au/d^2; g(r) constants
# dimensionless except R0 in au). These must correspond to the JPL orbit
# solutions the fixture's Horizons vectors were generated from.
CASES = {
    "99942 Apophis (2004 MN4)": {
        "central_mjd": 60000.0,
        "nongrav": {
            "A1": 5e-13,
            "A2": -2.901766637153165e-14,
            "A3": None,
            "ALN": 1.0,
            "NK": 0.0,
            "NM": 2.0,
            "NN": 5.093,
            "R0": 1.0,
        },
        # Conservative frozen-fixture bounds: reviewer-measured residuals
        # were ~16 m with a 3.74 km gravity-only control.
        "position_tolerance_km": 0.050,
        "gravity_control_minimum_km": 0.500,
    },
    "ZTF (C/2022 E3)": {
        "central_mjd": 59956.0,
        "nongrav": {
            "A1": None,
            "A2": -5.872322924187339e-10,
            "A3": -1.688049348767822e-09,
            "ALN": 0.1112620426,
            "NK": 4.6142,
            "NM": 2.15,
            "NN": 5.093,
            "R0": 2.808,
        },
        # Measured ~9 m with a ~2,749 km gravity-only control.
        "position_tolerance_km": 0.050,
        "gravity_control_minimum_km": 0.500,
    },
    "1I/'Oumuamua (A/2017 U1)": {
        "central_mjd": 58005.0,
        "nongrav": {
            "A1": 2.790193364334668e-07,
            "A2": 1.441264159911234e-08,
            "A3": 1.57392261588653e-08,
            "ALN": 0.0408373333128795,
            "NK": 2.6,
            "NM": 2.0,
            "NN": 3.0,
            "R0": 5.0,
        },
        # Measured ~188 m with a ~2.78 million km gravity-only control.
        "position_tolerance_km": 0.500,
        "gravity_control_minimum_km": 5.000,
    },
}


def _load_case(object_id: str, central_mjd: float) -> tuple[Orbits, Orbits]:
    fixture = Orbits.from_parquet(FIXTURE_PATH)
    rows = fixture.apply_mask(
        np.array([oid == object_id for oid in fixture.object_id.to_pylist()])
    )
    assert len(rows) == 7, f"fixture is missing rows for {object_id}"
    mjds = rows.coordinates.time.mjd().to_numpy(zero_copy_only=False)
    start = rows.apply_mask(np.isclose(mjds, central_mjd))
    assert len(start) == 1
    return start, rows


def _attach_nongrav(orbits: Orbits, params: dict) -> Orbits:
    return orbits.set_column(
        "non_gravitational_parameters",
        NonGravitationalParameters.from_kwargs(
            source=["SBDB"], **{name: [value] for name, value in params.items()}
        ),
    )


def _max_position_residual_km(propagated: Orbits, reference: Orbits) -> float:
    # Align on time: propagate_orbits returns rows for every requested epoch.
    prop_mjd = propagated.coordinates.time.mjd().to_numpy(zero_copy_only=False)
    ref_mjd = reference.coordinates.time.mjd().to_numpy(zero_copy_only=False)
    order = np.argsort(prop_mjd)
    ref_order = np.argsort(ref_mjd)
    np.testing.assert_allclose(prop_mjd[order], ref_mjd[ref_order], atol=1.2e-8)
    delta = (
        propagated.coordinates.r[order] - reference.coordinates.r[ref_order]
    ) * KM_PER_AU
    return float(np.linalg.norm(delta, axis=1).max())


@pytest.mark.parametrize("object_id", list(CASES))
def test_nongrav_force_law_matches_horizons(object_id):
    case = CASES[object_id]
    start, reference = _load_case(object_id, case["central_mjd"])
    times = reference.coordinates.time

    prop = ASSISTPropagator()

    nongrav_start = _attach_nongrav(start, case["nongrav"])
    residual_km = _max_position_residual_km(
        prop.propagate_orbits(nongrav_start, times), reference
    )
    assert residual_km < case["position_tolerance_km"], (
        f"{object_id}: non-grav propagation off Horizons by {residual_km:.4f} km "
        f"(tolerance {case['position_tolerance_km']} km)"
    )

    # The gravity-only control from the identical state must be much worse,
    # so a silently disabled force cannot satisfy the bound above.
    control_km = _max_position_residual_km(
        prop.propagate_orbits(start, times), reference
    )
    assert control_km > case["gravity_control_minimum_km"], (
        f"{object_id}: gravity-only control unexpectedly close to Horizons "
        f"({control_km:.4f} km) -- the non-grav bound is not discriminating"
    )
    assert residual_km < control_km / 10.0


def test_nongrav_mixed_batch_matches_solo_propagations():
    """
    A single same-epoch batch carrying all three force laws (inverse-square,
    standard comet, custom tuple) plus a force-free row must partition into
    per-law simulations whose results are identical to propagating each
    orbit alone. The fixture states have different epochs, so each is first
    brought to a common epoch with its own solo propagation.
    """
    prop = ASSISTPropagator()
    common_epoch = Timestamp.from_mjd([60000.0], scale="tdb")

    starts = []
    for object_id, case in CASES.items():
        start, _ = _load_case(object_id, case["central_mjd"])
        start = _attach_nongrav(start, case["nongrav"])
        at_epoch = prop.propagate_orbits(start, common_epoch)
        at_epoch = at_epoch.set_column(
            "non_gravitational_parameters",
            start.non_gravitational_parameters,
        )
        # The fixture queries each assigned the same default orbit_id;
        # rows need unique ids to be matched back out of the batch.
        starts.append(
            at_epoch.set_column(
                "orbit_id", pa.array([object_id], type=pa.large_string())
            )
        )
    # A force-free control row: same state as the first orbit, no non-grav
    # solution, grouped freely by the partitioner.
    force_free = starts[0].set_column(
        "non_gravitational_parameters", NonGravitationalParameters.nulls(1)
    )
    force_free = force_free.set_column(
        "orbit_id", pa.array(["force-free"], type=pa.large_string())
    )
    starts.append(force_free)

    times = Timestamp.from_mjd([60030.0, 60150.0], scale="tdb")
    solos = [prop.propagate_orbits(orbit, times) for orbit in starts]
    batched = prop.propagate_orbits(qv.concatenate(starts), times)

    assert len(batched) == len(starts) * len(times)
    for solo in solos:
        orbit_id = solo.orbit_id[0].as_py()
        batched_rows = batched.apply_mask(
            np.array([oid == orbit_id for oid in batched.orbit_id.to_pylist()])
        )
        residual_km = _max_position_residual_km(batched_rows, solo)
        # Rows whose (canonicalized) law matches the defaults share a
        # simulation with the force-free row, so IAS15's adaptive stepper
        # sees a different particle set than the solo run: mm-level step
        # noise is expected. Anything near the force scales (km) is a
        # routing bug.
        assert residual_km < 1e-4, (
            f"{orbit_id}: mixed-batch propagation differs from solo "
            f"propagation by {residual_km} km"
        )


def test_nongrav_covariance_propagates_through_variants():
    """
    A real 9x9 covariance (coordinates + A1/A2/A3) must propagate through
    ASSIST: variants jointly sampled from the A2 dimension carry different
    accelerations, so the propagated cloud must spread far beyond an
    identical cloud whose A-parameters are held fixed.
    """
    from adam_core.coordinates.covariances import CoordinateCovariances
    from adam_core.orbits.variants import VariantOrbits

    case = CASES["99942 Apophis (2004 MN4)"]
    start, _ = _load_case("99942 Apophis (2004 MN4)", case["central_mjd"])
    start = _attach_nongrav(start, case["nongrav"])

    # Tight coordinate covariance, dominant A2 variance: the propagated
    # spread should be driven by the non-grav dimension.
    full = np.zeros((9, 9))
    np.fill_diagonal(full[:3, :3], (1e-11) ** 2)  # ~1.5 m position sigma
    np.fill_diagonal(full[3:6, 3:6], (1e-13) ** 2)  # ~0.17 mm/s velocity sigma
    sigma_a2 = 5e-13  # au/d^2, much larger than the fitted A2 itself
    full[7, 7] = sigma_a2**2
    orbit = start.set_column(
        "coordinates.covariance",
        CoordinateCovariances.from_matrix(full[np.newaxis, ...]),
    )

    fixed = np.array(full)
    fixed[6:, :] = 0.0
    fixed[:, 6:] = 0.0
    control_orbit = start.set_column(
        "coordinates.covariance",
        CoordinateCovariances.from_matrix(fixed[np.newaxis, ...]),
    )

    prop = ASSISTPropagator()
    times = Timestamp.from_mjd([case["central_mjd"] + 150.0], scale="tdb")

    def _spread_km(orbits_in: Orbits) -> float:
        variants = VariantOrbits.create(
            orbits_in, method="monte-carlo", num_samples=64, seed=42
        )
        propagated = prop.propagate_orbits(variants, times)
        positions = propagated.coordinates.r * KM_PER_AU
        return float(np.linalg.norm(positions - positions.mean(axis=0), axis=1).std())

    nongrav_spread = _spread_km(orbit)
    control_spread = _spread_km(control_orbit)
    # sigma_A2 = 5e-13 au/d^2 over 150 days integrates to a ~1.7 km
    # dispersion (0.5 * a * t^2 = 0.5 * 5e-13 * 150^2 au); the fixed-A cloud
    # only carries the metre-scale coordinate uncertainty.
    assert nongrav_spread > 1.0, (
        f"A2 variance did not propagate into position spread "
        f"({nongrav_spread:.3f} km)"
    )
    assert nongrav_spread > 50.0 * control_spread, (
        f"non-grav spread {nongrav_spread:.3f} km vs fixed-A control "
        f"{control_spread:.3f} km is not discriminating"
    )


def test_public_covariance_propagation_preserves_9d_solution() -> None:
    from adam_core.coordinates.covariances import CoordinateCovariances

    case = CASES["99942 Apophis (2004 MN4)"]
    start, _ = _load_case("99942 Apophis (2004 MN4)", case["central_mjd"])
    start = _attach_nongrav(start, case["nongrav"])
    covariance = np.zeros((9, 9))
    np.fill_diagonal(covariance[:3, :3], (1e-11) ** 2)
    np.fill_diagonal(covariance[3:6, 3:6], (1e-13) ** 2)
    covariance[7, 7] = (5e-13) ** 2
    start = start.set_column(
        "coordinates.covariance",
        CoordinateCovariances.from_matrix(covariance[np.newaxis, ...]),
    )

    result = ASSISTPropagator().propagate_orbits(
        start,
        Timestamp.from_mjd([case["central_mjd"] + 30.0], scale="tdb"),
        covariance=True,
        covariance_method="sigma-point",
        num_samples=19,
        max_processes=1,
    )

    assert result.coordinates.covariance.nongrav_block_mask().tolist() == [True]
    assert np.isfinite(result.coordinates.covariance.to_full_matrix()).all()
    assert result.non_gravitational_parameters.A2.to_pylist() == [case["nongrav"]["A2"]]
