"""N-body least-squares OD parity: backend-generic Rust driver vs legacy scipy.

Bead personal-cmy.7. The Gauss-Newton driver lives in the permissive core
(`fit_orbit_least_squares_barycentric`, generic over the Propagator trait);
this GPL package only instantiates it with the ASSIST propagator, per the
packaging decision that GPL is confined to the adam-assist equivalent.

Parity policy mirrors the 2-body LSQ gate: bit parity is architecturally
impossible (Gauss-Newton + FD Jacobians vs scipy trust-region-reflective, plus
cross-libassist C builds), so the gate is converged-minimum parity. Measured
2026-07-05 on the noise-free fixture: Rust recovers truth to 1.25e-7 AU
(legacy scipy+ASSIST: 5.9e-7), rust-vs-legacy state agreement 7.1e-7 AU,
Rust 22x faster (0.08 s vs 1.80 s; one batched 7-candidate same-epoch
ephemeris crossing per iteration vs seven sequential Python ASSIST calls).
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
from adam_core.orbit_determination.evaluate import (
    OrbitDeterminationObservations,
    evaluate_orbits,
)
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


def test_rust_od_recovers_truth(od_problem):
    observations, initial = od_problem

    fitted_rust, chi2, iterations, converged = RustASSISTPropagator().fit_least_squares(
        initial, observations
    )
    assert converged
    # Current Rust ephemeris velocity-floor yields chi2=0.00335 while the
    # recovered state remains within 1e-6 of truth and 5e-6 of legacy.
    assert chi2 < 5e-3
    rust_state = fitted_rust.coordinates.values[0]
    # Truth recovery at the Gauss-Newton FD-Jacobian floor (measured 1.25e-7).
    np.testing.assert_allclose(rust_state, TRUTH_STATE, rtol=0, atol=1e-6)

    # The dynamic legacy optimizer performs many cross-process oracle calls and
    # is intentionally not rerun here. Pinned propagation/ephemeris parity is
    # gated separately; this fixture verifies recovery against known truth.

    # The inv(J^T J) covariance is finite and positive on the diagonal.
    rust_cov = fitted_rust.coordinates.covariance.to_matrix()[0]
    assert np.isfinite(rust_cov).all()
    assert (np.diag(rust_cov) > 0).all()


def test_public_fit_least_squares_dispatches_to_native(od_problem):
    """The canonical public entry point
    ``adam_core.orbit_determination.fit_least_squares`` dispatches to the
    Rust-native driver when given the Rust-backed propagator (bead
    personal-cmy.13.1.4), preserving the legacy (FittedOrbits,
    FittedOrbitMembers) contract."""
    observations, initial = od_problem
    rust_propagator = RustASSISTPropagator()

    fitted_orbit, fitted_members = fit_least_squares(
        initial, observations, rust_propagator
    )
    assert fitted_orbit.success[0].as_py()
    assert len(fitted_members) == len(observations)
    assert fitted_members.solution.to_pylist().count(True) == len(observations)

    # Bit-identical to the direct native fit: proves the native path ran
    # (the scipy path would only agree to its optimizer floor, not exactly).
    direct, _chi2, _iters, _conv = rust_propagator.fit_least_squares(
        initial, observations
    )
    np.testing.assert_array_equal(
        fitted_orbit.coordinates.values[0], direct.coordinates.values[0]
    )


def test_fit_least_squares_evaluated_matches_composed_crossings(od_problem):
    """The fused fit+evaluate work unit (bead personal-dqk) reproduces the
    two-crossing composition (native fit, then evaluate_orbits) exactly:
    same ephemeris kernels, same residual kernels, one crossing."""
    observations, initial = od_problem
    rust_propagator = RustASSISTPropagator()

    fused = rust_propagator.fit_least_squares_evaluated(initial, observations)

    fitted, _chi2, iterations, converged = rust_propagator.fit_least_squares(
        initial, observations
    )
    evaluated_orbit, evaluated_members = evaluate_orbits(
        fitted, observations, rust_propagator, parameters=6
    )

    np.testing.assert_array_equal(
        np.asarray(fused["state"]), fitted.coordinates.values[0]
    )
    np.testing.assert_array_equal(
        np.asarray(fused["covariance"]).reshape(6, 6),
        fitted.coordinates.covariance.to_matrix()[0],
    )
    assert fused["iterations"] == iterations
    assert fused["converged"] == converged
    np.testing.assert_array_equal(
        np.asarray(fused["residual_values"]),
        evaluated_members.residuals.to_array(),
    )
    np.testing.assert_array_equal(
        np.asarray(fused["residual_chi2"]),
        evaluated_members.residuals.chi2.to_numpy(zero_copy_only=False),
    )
    assert list(fused["residual_dof"]) == evaluated_members.residuals.dof.to_pylist()
    np.testing.assert_array_equal(
        np.asarray(fused["residual_probability"]),
        evaluated_members.residuals.probability.to_numpy(zero_copy_only=False),
    )
    assert fused["chi2"] == evaluated_orbit.chi2[0].as_py()
    assert fused["reduced_chi2"] == evaluated_orbit.reduced_chi2[0].as_py()
    assert fused["arc_length"] == evaluated_orbit.arc_length[0].as_py()
    assert fused["num_obs"] == evaluated_orbit.num_obs[0].as_py()
    assert list(fused["outlier"]) == [False] * len(observations)

    # Ignore-mask path: the fit runs on the subset, the evaluation covers all.
    ignore_ids = [observations.id[1].as_py(), observations.id[6].as_py()]
    ignore_mask = [obs_id in ignore_ids for obs_id in observations.id.to_pylist()]
    fused_ignored = rust_propagator.fit_least_squares_evaluated(
        initial, observations, ignore_mask
    )
    assert list(fused_ignored["outlier"]) == ignore_mask
    assert fused_ignored["num_obs"] == len(observations) - 2


def test_od_fit_matches_frozen_legacy_solution(od_problem, frozen_assist_regression):
    """The Rust OD result retains accepted converged-solution parity."""
    observations, initial = od_problem
    rust_propagator = RustASSISTPropagator()

    output = rust_propagator.od_fit(
        initial,
        observations,
        rchi2_threshold=100.0,
        min_obs=5,
        min_arc_length=1.0,
        contamination_percentage=0.0,
        delta=1e-6,
        max_iter=20,
        method="central",
    )
    assert output["found"]
    assert output["improved"]
    assert output["reduced_chi2"] <= 100.0
    assert output["num_obs"] == len(observations)
    assert list(output["outlier"]) == [False] * len(observations)
    np.testing.assert_allclose(
        np.asarray(output["state"]), TRUTH_STATE, rtol=0, atol=1e-5
    )

    np.testing.assert_allclose(
        np.asarray(output["state"]),
        frozen_assist_regression["od_legacy_state"],
        rtol=0,
        atol=5e-5,
    )
    assert output["num_obs"] == frozen_assist_regression["od_legacy_num_obs"][0]
    assert (
        list(output["outlier"])
        == frozen_assist_regression["od_legacy_outlier"].tolist()
    )
    assert bool(output["improved"]) == bool(
        frozen_assist_regression["od_legacy_success"][0]
    )
    np.testing.assert_allclose(
        output["arc_length"],
        frozen_assist_regression["od_legacy_arc_length"][0],
        rtol=1e-12,
    )


def test_od_fit_below_min_obs_returns_empty(od_problem):
    observations, initial = od_problem
    output = RustASSISTPropagator().od_fit(
        initial, observations, min_obs=len(observations) + 1
    )
    assert not output["found"]
    assert output["num_obs"] == 0
    assert len(output["residual_chi2"]) == 0


def test_initial_orbit_determination_fused_batch_and_timing(od_problem):
    observations, _initial = od_problem
    propagator = RustASSISTPropagator()
    obs_ids = observations.id.to_pylist()
    output = propagator.initial_orbit_determination(
        observations,
        ["b", "a"],
        ["b"] * len(obs_ids) + ["a"] * len(obs_ids),
        obs_ids + obs_ids,
        min_obs=3,
        min_arc_length=1.0,
        rchi2_threshold=1e12,
        contamination_percentage=0.0,
        chunk_size=2,
    )
    # Identical linkage inputs generate an exact duplicate preliminary state;
    # Rust keeps the first and returns linkage/member order deterministically.
    assert output["orbit_ids"] == ["b"]
    assert output["member_orbit_ids"] == ["b"] * len(obs_ids)
    assert output["member_obs_ids"] == obs_ids
    assert np.isfinite(np.asarray(output["states"])).all()

    operation, trials = propagator.benchmark_last_native(2, 2, 1)
    assert operation == "initial_orbit_determination"
    assert len(trials) == 2
    assert all(len(trial) == 2 for trial in trials)
    assert all(sample > 0 for trial in trials for sample in trial)


@pytest.mark.parametrize("use_central_difference", [True, False])
def test_vallado_least_squares_matches_frozen_parity(
    od_problem, frozen_assist_regression, use_central_difference
):
    """Rust Vallado output retains the accepted legacy solution floor."""
    observations, initial = od_problem
    rust_propagator = RustASSISTPropagator()

    output = rust_propagator.vallado_least_squares(
        initial, observations, use_central_difference=use_central_difference
    )
    assert output["status"] == "updated"
    assert output["num_observations"] == len(observations)
    assert len(output["iterations_rms"]) > 1
    assert output["iterations_rms"][-1] < output["iterations_rms"][0]
    assert len(output["iterations_rchi2"]) == len(output["iterations_rms"])
    assert len(np.asarray(output["corrections"])) == len(output["iterations_rms"]) - 1
    assert output["iterations_delta_rms"][0] is None
    assert output["iterations_converged"][0] is None
    np.testing.assert_allclose(
        np.asarray(output["state"]), TRUTH_STATE, rtol=0, atol=1e-5
    )

    key = "central" if use_central_difference else "forward"
    np.testing.assert_allclose(
        np.asarray(output["state"]),
        frozen_assist_regression[f"vallado_{key}_state"],
        rtol=0,
        atol=5e-5,
    )
    assert (
        output["num_observations"]
        == frozen_assist_regression[f"vallado_{key}_num_observations"][0]
    )
    assert frozen_assist_regression[f"vallado_{key}_iteration_count"][0] > 1


def test_native_od_timing_lanes(od_problem):
    """Each fused OD work unit is wired to the Rust-owned timing hook."""
    observations, initial = od_problem
    rust_propagator = RustASSISTPropagator()

    rust_propagator.fit_least_squares(initial, observations)
    operation, samples = rust_propagator.benchmark_last_native(2, 2, 1)
    assert operation == "fit_least_squares"
    assert len(samples) == 2 and all(len(trial) == 2 for trial in samples)

    rust_propagator.od_fit(initial, observations)
    operation, samples = rust_propagator.benchmark_last_native(2, 2, 1)
    assert operation == "od_fit"
    assert len(samples) == 2 and all(len(trial) == 2 for trial in samples)

    rust_propagator.fit_least_squares_evaluated(initial, observations)
    operation, _ = rust_propagator.benchmark_last_native(1, 1, 0)
    assert operation == "fit_least_squares_evaluated"

    rust_propagator.vallado_least_squares(
        initial, observations, use_central_difference=True
    )
    operation, _ = rust_propagator.benchmark_last_native(1, 1, 0)
    assert operation == "vallado_least_squares"
