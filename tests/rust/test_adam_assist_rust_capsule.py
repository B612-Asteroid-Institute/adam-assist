"""Capsule conformance and adam-core-owned OD integration.

adam-assist supplies only ASSIST propagation and STM capabilities. The public
fit is owned and orchestrated by adam-core, which consumes the private native
propagator capsule without provider-specific OD work units.
"""

import ast
import ctypes
import gc
import importlib
from pathlib import Path

import numpy as np
import pytest
from adam_core import _rust_native
from adam_core.coordinates import (
    CartesianCoordinates,
    CoordinateCovariances,
    Origin,
    SphericalCoordinates,
)
from adam_core.observers import Observers
from adam_core.orbit_determination import refine, run_iod, run_od
from adam_core.orbit_determination.evaluate import OrbitDeterminationObservations
from adam_core.orbit_determination.native_orbit_fitter import NativeOrbitFitter
from adam_core.orbit_determination.provenance import backend_identity
from adam_core.orbits import Orbits
from adam_core.orbits.non_gravitational_parameters import (
    NonGravitationalParameters,
)
from adam_core.orbits.physical_parameters import PhysicalParameters
from adam_core.propagator.propagator import NativeBackend, Propagator
from adam_core.time import Timestamp

from adam_assist import ASSISTPropagator as RustASSISTPropagator

CARGO_VERSION = importlib.import_module("adam_assist._native").CARGO_VERSION
EPOCH_MJD = 60000.0
TRUTH_STATE = np.array([1.2, 0.1, 0.05, -0.002, 0.016, 0.001])
CAPSULE_NAME = b"adam_core.propagator.v2"
CAP_THREAD_SAFE = 1 << 0
CAP_STM = 1 << 1
CAP_NONGRAV = 1 << 2
FAILURE_NONE = 0
FAILURE_NON_FINITE_INPUT_STATE = 1
CUSTOM_NONGRAV = {
    "source": ["custom"],
    "A1": [5.0e-9],
    "A2": [-2.9e-9],
    "A3": [4.2e-10],
    "ALN": [0.1112620426],
    "NK": [4.6142],
    "NM": [2.15],
    "NN": [5.093],
    "R0": [2.808],
}


class _ExternalPropagatorV2(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("struct_size", ctypes.c_size_t),
        ("capabilities", ctypes.c_uint64),
        ("integration_time_scale", ctypes.c_char_p),
        ("two_body_dynamics", ctypes.c_uint8),
        ("name", ctypes.c_char_p),
        ("version", ctypes.c_char_p),
        ("context", ctypes.c_void_p),
        ("propagate", ctypes.c_void_p),
        ("propagate_stm", ctypes.c_void_p),
        ("reserved", ctypes.c_size_t * 4),
    ]


class _PropagateRequestV2(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("struct_size", ctypes.c_size_t),
        ("n_orbits", ctypes.c_size_t),
        ("states", ctypes.POINTER(ctypes.c_double)),
        ("physical_parameters", ctypes.POINTER(ctypes.c_double)),
        ("non_gravitational_parameters", ctypes.POINTER(ctypes.c_double)),
        ("epoch_days", ctypes.POINTER(ctypes.c_int64)),
        ("epoch_nanos", ctypes.POINTER(ctypes.c_int64)),
        ("orbit_time_scale", ctypes.c_char_p),
        ("frame", ctypes.c_char_p),
        ("origin", ctypes.c_char_p),
        ("n_times", ctypes.c_size_t),
        ("time_days", ctypes.POINTER(ctypes.c_int64)),
        ("time_nanos", ctypes.POINTER(ctypes.c_int64)),
        ("time_scale", ctypes.c_char_p),
        ("pairwise", ctypes.c_uint8),
        ("out_states", ctypes.POINTER(ctypes.c_double)),
        ("out_valid", ctypes.POINTER(ctypes.c_uint8)),
        ("out_failure_codes", ctypes.POINTER(ctypes.c_uint8)),
        ("error", ctypes.POINTER(ctypes.c_char)),
        ("error_len", ctypes.c_size_t),
        ("reserved", ctypes.c_size_t * 4),
    ]


class _PropagateStmRequestV2(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("struct_size", ctypes.c_size_t),
        ("state", ctypes.POINTER(ctypes.c_double)),
        ("non_gravitational_parameters", ctypes.POINTER(ctypes.c_double)),
        ("epoch_days", ctypes.c_int64),
        ("epoch_nanos", ctypes.c_int64),
        ("n_times", ctypes.c_size_t),
        ("time_days", ctypes.POINTER(ctypes.c_int64)),
        ("time_nanos", ctypes.POINTER(ctypes.c_int64)),
        ("out_states", ctypes.POINTER(ctypes.c_double)),
        ("out_stm", ctypes.POINTER(ctypes.c_double)),
        ("out_valid", ctypes.POINTER(ctypes.c_uint8)),
        ("error", ctypes.POINTER(ctypes.c_char)),
        ("error_len", ctypes.c_size_t),
        ("reserved", ctypes.c_size_t * 4),
    ]


def _capsule_table(capsule) -> _ExternalPropagatorV2:
    get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
    get_pointer.argtypes = (ctypes.py_object, ctypes.c_char_p)
    get_pointer.restype = ctypes.c_void_p
    pointer = get_pointer(capsule, CAPSULE_NAME)
    assert pointer
    return ctypes.cast(pointer, ctypes.POINTER(_ExternalPropagatorV2)).contents


def _double_buffer(values: np.ndarray):
    flat = np.ascontiguousarray(values, dtype=np.float64).reshape(-1)
    return (ctypes.c_double * len(flat))(*flat)


def _nongrav_buffer(values: list[float] | None):
    if values is None:
        return None
    assert len(values) % 8 == 0
    return (ctypes.c_double * len(values))(*values)


def _call_capsule_propagate(
    capsule,
    states: np.ndarray,
    target_days: list[int],
    nongrav: list[float] | None = None,
):
    table = _capsule_table(capsule)
    states = np.asarray(states, dtype=np.float64).reshape(-1, 6)
    n_orbits = len(states)
    n_times = len(target_days)
    rows = n_orbits * n_times
    states_buffer = _double_buffer(states)
    nongrav_buffer = _nongrav_buffer(nongrav)
    epoch_days = (ctypes.c_int64 * n_orbits)(*([int(EPOCH_MJD)] * n_orbits))
    epoch_nanos = (ctypes.c_int64 * n_orbits)(*([0] * n_orbits))
    times = (ctypes.c_int64 * n_times)(*target_days)
    nanos = (ctypes.c_int64 * n_times)(*([0] * n_times))
    output = (ctypes.c_double * (rows * 6))(*([np.nan] * (rows * 6)))
    valid = (ctypes.c_uint8 * rows)(*([0] * rows))
    failures = (ctypes.c_uint8 * rows)(*([255] * rows))
    error = ctypes.create_string_buffer(1024)
    request = _PropagateRequestV2(
        abi_version=2,
        struct_size=ctypes.sizeof(_PropagateRequestV2),
        n_orbits=n_orbits,
        states=states_buffer,
        physical_parameters=None,
        non_gravitational_parameters=nongrav_buffer,
        epoch_days=epoch_days,
        epoch_nanos=epoch_nanos,
        orbit_time_scale=b"tdb",
        frame=b"ecliptic",
        origin=b"SUN",
        n_times=n_times,
        time_days=times,
        time_nanos=nanos,
        time_scale=b"tdb",
        pairwise=0,
        out_states=output,
        out_valid=valid,
        out_failure_codes=failures,
        error=ctypes.cast(error, ctypes.POINTER(ctypes.c_char)),
        error_len=len(error),
        reserved=(ctypes.c_size_t * 4)(0, 0, 0, 0),
    )
    callback_type = ctypes.CFUNCTYPE(
        ctypes.c_int32, ctypes.c_void_p, ctypes.POINTER(_PropagateRequestV2)
    )
    status = callback_type(table.propagate)(table.context, ctypes.byref(request))
    return (
        status,
        np.ctypeslib.as_array(output).copy().reshape(rows, 6),
        list(valid),
        list(failures),
        error.value.decode(),
    )


def _call_capsule_stm(
    capsule,
    state: np.ndarray,
    target_days: list[int],
    nongrav: list[float] | None = None,
):
    table = _capsule_table(capsule)
    state_buffer = _double_buffer(state)
    nongrav_buffer = _nongrav_buffer(nongrav)
    n_times = len(target_days)
    times = (ctypes.c_int64 * n_times)(*target_days)
    nanos = (ctypes.c_int64 * n_times)(*([0] * n_times))
    output = (ctypes.c_double * (n_times * 6))(*([np.nan] * (n_times * 6)))
    stms = (ctypes.c_double * (n_times * 36))(*([np.nan] * (n_times * 36)))
    valid = (ctypes.c_uint8 * n_times)(*([0] * n_times))
    error = ctypes.create_string_buffer(1024)
    request = _PropagateStmRequestV2(
        abi_version=2,
        struct_size=ctypes.sizeof(_PropagateStmRequestV2),
        state=state_buffer,
        non_gravitational_parameters=nongrav_buffer,
        epoch_days=int(EPOCH_MJD),
        epoch_nanos=0,
        n_times=n_times,
        time_days=times,
        time_nanos=nanos,
        out_states=output,
        out_stm=stms,
        out_valid=valid,
        error=ctypes.cast(error, ctypes.POINTER(ctypes.c_char)),
        error_len=len(error),
        reserved=(ctypes.c_size_t * 4)(0, 0, 0, 0),
    )
    callback_type = ctypes.CFUNCTYPE(
        ctypes.c_int32, ctypes.c_void_p, ctypes.POINTER(_PropagateStmRequestV2)
    )
    status = callback_type(table.propagate_stm)(table.context, ctypes.byref(request))
    return (
        status,
        np.ctypeslib.as_array(output).copy().reshape(n_times, 6),
        np.ctypeslib.as_array(stms).copy().reshape(n_times, 6, 6),
        list(valid),
        error.value.decode(),
    )


def _make_orbit(
    state: np.ndarray,
    orbit_id: str,
    *,
    nongrav: NonGravitationalParameters | None = None,
    physical: PhysicalParameters | None = None,
) -> Orbits:
    return Orbits.from_kwargs(
        orbit_id=[orbit_id],
        object_id=[orbit_id],
        non_gravitational_parameters=nongrav,
        physical_parameters=physical,
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
    cov = np.full((n, 6, 6), np.nan)
    cov[:, 1, 1] = arcsec
    cov[:, 2, 2] = arcsec
    observed = SphericalCoordinates.from_kwargs(
        rho=np.full(n, np.nan),
        lon=values[:, 1],
        lat=values[:, 2],
        vrho=np.full(n, np.nan),
        vlon=np.full(n, np.nan),
        vlat=np.full(n, np.nan),
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
    assert backend.version == CARGO_VERSION
    assert backend.work_units == frozenset({"propagator_capsule"})
    assert propagator.two_body_dynamics is False
    # This is the provider identity Core records in run_od provenance.
    assert backend_identity(propagator) == f"adam_assist {CARGO_VERSION}"

    # Keep the Python declaration tied to the immutable native table identity,
    # even when Python and Cargo versions happen to have the same spelling.
    source_path = Path(__file__).parents[2] / "src/adam_assist/propagator.py"
    module = ast.parse(source_path.read_text())
    assert any(
        isinstance(node, ast.ImportFrom)
        and node.module == "_native"
        and any(
            alias.name == "CARGO_VERSION" and alias.asname == "_NATIVE_CARGO_VERSION"
            for alias in node.names
        )
        for node in ast.walk(module)
    )
    backend_assignment = next(
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "native_backend"
            for target in node.targets
        )
    )
    version = next(
        keyword.value
        for keyword in backend_assignment.value.keywords
        if keyword.arg == "version"
    )
    assert isinstance(version, ast.Name)
    assert version.id == "_NATIVE_CARGO_VERSION"

    capsule = propagator.propagator_capsule()
    get_name = ctypes.pythonapi.PyCapsule_GetName
    get_name.argtypes = (ctypes.py_object,)
    get_name.restype = ctypes.c_char_p
    assert type(capsule).__name__ == "PyCapsule"
    assert get_name(capsule) == CAPSULE_NAME
    table = _capsule_table(capsule)
    assert table.abi_version == 2
    assert table.struct_size == ctypes.sizeof(_ExternalPropagatorV2)
    assert table.capabilities & (CAP_THREAD_SAFE | CAP_STM | CAP_NONGRAV) == (
        CAP_THREAD_SAFE | CAP_STM | CAP_NONGRAV
    )
    assert table.integration_time_scale == b"tdb"
    assert table.two_body_dynamics == 0
    assert table.name == b"adam_assist"
    assert table.version.decode() == CARGO_VERSION
    assert table.context
    assert table.propagate
    assert table.propagate_stm
    assert list(table.reserved) == [0, 0, 0, 0]
    for method in (
        "fit_least_squares",
        "fit_least_squares_evaluated",
        "od_fit",
        "initial_orbit_determination",
        "vallado_least_squares",
    ):
        assert not hasattr(propagator, method)


def test_adam_core_refine_uses_assist_capsule_stm(od_problem) -> None:
    observations, initial = od_problem

    fitted_orbit, fitted_members = refine(
        initial,
        observations,
        RustASSISTPropagator(),
        rejection="none",
        fit_settings={"jacobian": "propagator"},
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


def test_capsule_propagation_preserves_order_nongrav_and_row_failures() -> None:
    capsule = RustASSISTPropagator().propagator_capsule()
    custom = [
        CUSTOM_NONGRAV["A1"][0],
        CUSTOM_NONGRAV["A2"][0],
        CUSTOM_NONGRAV["A3"][0],
        CUSTOM_NONGRAV["ALN"][0],
        CUSTOM_NONGRAV["NK"][0],
        CUSTOM_NONGRAV["NM"][0],
        CUSTOM_NONGRAV["NN"][0],
        CUSTOM_NONGRAV["R0"][0],
    ]
    status, custom_states, valid, failures, error = _call_capsule_propagate(
        capsule, TRUTH_STATE, [60020, 60010], custom
    )
    assert (status, valid, failures, error) == (
        0,
        [1, 1],
        [FAILURE_NONE, FAILURE_NONE],
        "",
    )
    status, default_states, valid, failures, error = _call_capsule_propagate(
        capsule,
        TRUTH_STATE,
        [60020, 60010],
        custom[:3] + [np.nan] * 5,
    )
    assert (status, valid, failures, error) == (
        0,
        [1, 1],
        [FAILURE_NONE, FAILURE_NONE],
        "",
    )
    status, gravity_states, valid, failures, error = _call_capsule_propagate(
        capsule, TRUTH_STATE, [60020, 60010]
    )
    assert (status, valid, failures, error) == (
        0,
        [1, 1],
        [FAILURE_NONE, FAILURE_NONE],
        "",
    )
    assert np.max(np.abs(custom_states - gravity_states)) > 1e-10
    assert np.max(np.abs(custom_states - default_states)) > 1e-12
    # The provider returns caller target order; no callback-local sorting leaks.
    assert not np.array_equal(custom_states[0], custom_states[1])

    inactive = [0.0, 0.0, 0.0] + custom[3:]
    status, inactive_states, valid, failures, error = _call_capsule_propagate(
        capsule, TRUTH_STATE, [60020, 60010], inactive
    )
    assert (status, valid, failures, error) == (
        0,
        [1, 1],
        [FAILURE_NONE, FAILURE_NONE],
        "",
    )
    np.testing.assert_array_equal(inactive_states, gravity_states)

    mixed = np.vstack([TRUTH_STATE, TRUTH_STATE.copy()])
    mixed[1, 0] = np.nan
    status, output, valid, failures, error = _call_capsule_propagate(
        capsule, mixed, [60010]
    )
    assert status == 0
    assert valid == [1, 0]
    assert failures == [FAILURE_NONE, FAILURE_NON_FINITE_INPUT_STATE]
    assert error == ""
    assert np.isfinite(output[0]).all()
    assert np.isnan(output[1]).all()


def test_capsule_partial_marsden_constants_match_filled_defaults() -> None:
    capsule = RustASSISTPropagator().propagator_capsule()
    accelerations = [5.0e-9, -2.9e-9, 4.2e-10]
    partial = accelerations + [0.25, np.nan, np.nan, 4.0, np.nan]
    filled = accelerations + [0.25, 0.0, 2.0, 4.0, 1.0]

    partial_result = _call_capsule_propagate(
        capsule, TRUTH_STATE, [60020, 60010], partial
    )
    filled_result = _call_capsule_propagate(
        capsule, TRUTH_STATE, [60020, 60010], filled
    )
    assert partial_result[0] == filled_result[0] == 0
    assert partial_result[2:] == filled_result[2:]
    np.testing.assert_array_equal(partial_result[1], filled_result[1])

    partial_stm = _call_capsule_stm(capsule, TRUTH_STATE, [60020, 60010], partial)
    filled_stm = _call_capsule_stm(capsule, TRUTH_STATE, [60020, 60010], filled)
    assert partial_stm[0] == filled_stm[0] == 0
    assert partial_stm[3:] == filled_stm[3:]
    np.testing.assert_array_equal(partial_stm[1], filled_stm[1])
    np.testing.assert_array_equal(partial_stm[2], filled_stm[2])


def test_capsule_stm_honors_custom_and_inactive_nongrav() -> None:
    capsule = RustASSISTPropagator().propagator_capsule()
    custom = [
        CUSTOM_NONGRAV["A1"][0],
        CUSTOM_NONGRAV["A2"][0],
        CUSTOM_NONGRAV["A3"][0],
        CUSTOM_NONGRAV["ALN"][0],
        CUSTOM_NONGRAV["NK"][0],
        CUSTOM_NONGRAV["NM"][0],
        CUSTOM_NONGRAV["NN"][0],
        CUSTOM_NONGRAV["R0"][0],
    ]
    status, stm_states, stms, valid, error = _call_capsule_stm(
        capsule, TRUTH_STATE, [60020, 60010], custom
    )
    assert (status, valid, error) == (0, [1, 1], "")
    assert np.isfinite(stms).all()
    status, propagated, valid, failures, error = _call_capsule_propagate(
        capsule, TRUTH_STATE, [60020, 60010], custom
    )
    assert (status, valid, failures, error) == (0, [1, 1], [0, 0], "")
    np.testing.assert_allclose(stm_states, propagated, rtol=0, atol=5e-15)

    inactive = [0.0, 0.0, 0.0] + custom[3:]
    status, inactive_states, inactive_stms, valid, error = _call_capsule_stm(
        capsule, TRUTH_STATE, [60020, 60010], inactive
    )
    assert (status, valid, error) == (0, [1, 1], "")
    status, gravity_states, gravity_stms, valid, error = _call_capsule_stm(
        capsule, TRUTH_STATE, [60020, 60010]
    )
    assert (status, valid, error) == (0, [1, 1], "")
    np.testing.assert_array_equal(inactive_states, gravity_states)
    np.testing.assert_array_equal(inactive_stms, gravity_stms)


def test_capsule_owns_provider_lifetime(od_problem) -> None:
    provider = RustASSISTPropagator()
    capsule = provider.propagator_capsule()
    del provider
    gc.collect()

    class RetainedCapsule(Propagator):
        native_backend = NativeBackend(
            "adam_assist", CARGO_VERSION, frozenset({"propagator_capsule"})
        )

        def propagator_capsule(self):
            return capsule

        def propagate_orbits(self, *args, **kwargs):
            raise AssertionError("Core must consume the retained native capsule")

        def generate_ephemeris(self, *args, **kwargs):
            raise AssertionError("Core must consume the retained native capsule")

    observations, initial = od_problem
    fitted, _ = refine(
        initial,
        observations,
        RetainedCapsule(),
        rejection="none",
        fit_settings={"jacobian": "propagator"},
    )
    assert fitted.success[0].as_py()


def test_python_only_propagator_override_is_rejected(od_problem) -> None:
    class PythonOnlyAssist(RustASSISTPropagator):
        def propagate_orbits(self, *args, **kwargs):
            return super().propagate_orbits(*args, **kwargs)

    observations, initial = od_problem
    with pytest.raises(TypeError, match="no native orbit-determination backend"):
        refine(initial, observations, PythonOnlyAssist(), rejection="none")


def _observations_from_truth(truth: Orbits) -> OrbitDeterminationObservations:
    n = 8
    times = Timestamp.from_mjd(
        [EPOCH_MJD + 2.0 + i * 3.0 for i in range(n)], scale="utc"
    )
    observers = Observers.from_code("X05", times)
    predicted = (
        RustASSISTPropagator()
        .generate_ephemeris(
            truth,
            observers,
            covariance=False,
            max_processes=1,
            predict_magnitudes=False,
            predict_phase_angle=False,
        )
        .coordinates
    )
    covariance = np.full((n, 6, 6), np.nan)
    variance = (1.0 / 3600.0) ** 2
    covariance[:, 1, 1] = variance
    covariance[:, 2, 2] = variance
    coordinates = SphericalCoordinates.from_kwargs(
        rho=np.full(n, np.nan),
        lon=predicted.lon,
        lat=predicted.lat,
        vrho=np.full(n, np.nan),
        vlon=np.full(n, np.nan),
        vlat=np.full(n, np.nan),
        time=predicted.time,
        origin=predicted.origin,
        frame=predicted.frame,
        covariance=CoordinateCovariances.from_matrix(covariance),
    )
    return OrbitDeterminationObservations.from_kwargs(
        id=[f"custom-{i}" for i in range(n)],
        coordinates=coordinates,
        observers=observers,
    )


def test_core_refine_preserves_physical_and_custom_nongrav_metadata() -> None:
    nongrav = NonGravitationalParameters.from_kwargs(**CUSTOM_NONGRAV)
    physical = PhysicalParameters.from_kwargs(
        H_v=[17.2],
        H_v_sigma=[0.3],
        G=[0.21],
        G_sigma=[0.04],
        sigma_eff=[0.12],
        chi2_red=[1.4],
    )
    truth = _make_orbit(TRUTH_STATE, "custom-truth", nongrav=nongrav, physical=physical)
    observations = _observations_from_truth(truth)
    initial = _make_orbit(
        TRUTH_STATE + np.array([1e-3, -1e-3, 5e-4, 1e-5, -1e-5, 1e-5]),
        "custom-fit",
        nongrav=nongrav,
        physical=physical,
    )

    fitted, members = refine(
        initial,
        observations,
        RustASSISTPropagator(),
        rejection="none",
        fit_settings={"jacobian": "propagator"},
    )

    assert fitted.success[0].as_py()
    assert members.obs_id.to_pylist() == observations.id.to_pylist()
    np.testing.assert_allclose(
        fitted.coordinates.values[0], TRUTH_STATE, rtol=0, atol=1e-6
    )
    assert fitted.coordinates.frame == initial.coordinates.frame
    assert (
        fitted.coordinates.origin.code.to_pylist()
        == initial.coordinates.origin.code.to_pylist()
    )
    assert fitted.coordinates.time.equals(initial.coordinates.time)
    assert fitted.non_gravitational_parameters.source.to_pylist() == ["custom"]
    for field, expected in CUSTOM_NONGRAV.items():
        if field == "source":
            continue
        assert (
            getattr(fitted.non_gravitational_parameters, field).to_pylist() == expected
        )
    for field, expected in {
        "H_v": [17.2],
        "H_v_sigma": [0.3],
        "G": [0.21],
        "G_sigma": [0.04],
        "sigma_eff": [0.12],
        "chi2_red": [1.4],
    }.items():
        assert getattr(fitted.physical_parameters, field).to_pylist() == expected


def test_core_refine_preserves_but_does_not_activate_source_only_nongrav(
    od_problem,
) -> None:
    observations, _ = od_problem
    inactive = NonGravitationalParameters.from_kwargs(
        source=["metadata-only"],
        A1=[0.0],
        A2=[None],
        A3=[0.0],
        ALN=[0.1112620426],
        NK=[4.6142],
        NM=[2.15],
        NN=[5.093],
        R0=[2.808],
    )
    initial = _make_orbit(
        TRUTH_STATE + np.array([1e-3, -1e-3, 5e-4, 1e-5, -1e-5, 1e-5]),
        "inactive-fit",
        nongrav=inactive,
    )
    fitted, _ = refine(
        initial,
        observations,
        RustASSISTPropagator(),
        rejection="none",
        fit_settings={"jacobian": "propagator"},
    )
    assert fitted.success[0].as_py()
    assert fitted.non_gravitational_parameters.source.to_pylist() == ["metadata-only"]
    assert fitted.non_gravitational_parameters.A1.to_pylist() == [0.0]
    assert fitted.non_gravitational_parameters.A2.to_pylist() == [None]
    np.testing.assert_allclose(
        fitted.coordinates.values[0], TRUTH_STATE, rtol=0, atol=1e-6
    )


def test_adam_core_run_iod_uses_assist_capsule_once(od_problem) -> None:
    class CountingAssist(RustASSISTPropagator):
        native_backend = NativeBackend(
            "adam_assist", CARGO_VERSION, frozenset({"propagator_capsule"})
        )

        def __init__(self):
            super().__init__()
            self.capsule_acquisitions = 0

        def propagator_capsule(self):
            self.capsule_acquisitions += 1
            return super().propagator_capsule()

        def propagate_orbits(self, *args, **kwargs):
            raise AssertionError("Core IOD must use the ASSIST native capsule")

        def generate_ephemeris(self, *args, **kwargs):
            raise AssertionError("Core IOD must use the ASSIST native capsule")

    observations, _ = od_problem
    propagator = CountingAssist()
    fitted, members, report = run_iod(
        observations,
        propagator,
        min_obs=6,
        min_arc_length=1.0,
        rchi2_threshold=1000.0,
        observation_selection_method="combinations",
        velocity_method="gibbs",
        acceptance="prefit",
    )

    assert report.found
    assert len(fitted) == 1
    assert len(members) == len(observations)
    assert propagator.capsule_acquisitions == 1


def test_distinct_core_iod_and_assist_refinement_capsules_cross_python_once(
    od_problem,
) -> None:
    class CountingCoreTwoBody(Propagator):
        two_body_dynamics = True
        native_backend = NativeBackend(
            "adam_core.two_body.external",
            _rust_native.two_body_propagator_semver(),
            frozenset({"propagator_capsule"}),
        )

        def __init__(self):
            self.capsule_acquisitions = 0

        def propagator_capsule(self):
            self.capsule_acquisitions += 1
            return _rust_native.two_body_propagator_capsule()

        def propagate_orbits(self, *args, **kwargs):
            raise AssertionError("Core IOD must use the native capsule")

        def generate_ephemeris(self, *args, **kwargs):
            raise AssertionError("Core IOD must use the native capsule")

    class CountingAssist(RustASSISTPropagator):
        native_backend = NativeBackend(
            "adam_assist", CARGO_VERSION, frozenset({"propagator_capsule"})
        )

        def __init__(self):
            super().__init__()
            self.capsule_acquisitions = 0

        def propagator_capsule(self):
            self.capsule_acquisitions += 1
            return super().propagator_capsule()

        def propagate_orbits(self, *args, **kwargs):
            raise AssertionError("Core refinement must use the ASSIST native capsule")

        def generate_ephemeris(self, *args, **kwargs):
            raise AssertionError("Core refinement must use the ASSIST native capsule")

    observations, _ = od_problem
    iod = CountingCoreTwoBody()
    refinement = CountingAssist()
    fitter = NativeOrbitFitter(
        iod_propagator=iod,
        min_obs=6,
        min_arc_length=1.0,
        iod_rchi2_threshold=1000.0,
        observation_selection_method="combinations",
        velocity_method="gibbs",
        iod_acceptance="prefit",
        outlier_rejection="worst_residual",
        rchi2_threshold=100.0,
        rejection_kwargs={"jacobian": "propagator"},
    )

    fitted, members = run_od(
        observations,
        fitter=fitter,
        propagator=refinement,
        models=None,
        object_id="distinct-stage",
    )

    assert len(fitted) == 1
    assert fitted.success[0].as_py()
    assert len(members) == len(observations)
    assert iod.capsule_acquisitions == 1
    assert refinement.capsule_acquisitions == 1
    assert (
        fitted.provenance.iod_backend[0]
        .as_py()
        .startswith("adam_core.two_body.external ")
    )
    assert fitted.provenance.refinement_backend[0].as_py() == (
        f"adam_assist {CARGO_VERSION}"
    )
    truth_at_fit_epoch = RustASSISTPropagator().propagate_orbits(
        _make_orbit(TRUTH_STATE, "distinct-truth"), fitted.coordinates.time
    )
    np.testing.assert_allclose(
        fitted.coordinates.values[0],
        truth_at_fit_epoch.coordinates.values[0],
        rtol=0,
        atol=1e-9,
    )
    assert fitted.reduced_chi2[0].as_py() < 1e-10
