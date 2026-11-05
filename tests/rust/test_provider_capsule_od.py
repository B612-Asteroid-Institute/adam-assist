"""Independent probes of the ASSIST ABI-v2 provider with Core OD.

The imported ``od_problem`` fixture intentionally shares its pytest argument
name throughout this module.
"""

# ruff: noqa: F811

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
from adam_core.coordinates import CartesianCoordinates, Origin
from adam_core.observations import Associations, SourceCatalog
from adam_core.orbit_determination import evaluate_orbits, refine, run_iod, run_od
from adam_core.orbits import Orbits
from adam_core.propagator.propagator import NativeBackend, Propagator
from adam_core.time import Timestamp
from test_adam_assist_rust_capsule import (  # noqa: F401
    CAPSULE_NAME,
    CARGO_VERSION,
    CUSTOM_NONGRAV,
    TRUTH_STATE,
    _call_capsule_stm,
    _canonical_inputs,
    _capsule_table,
    _ExternalPropagatorV2,
    _od_config,
    _PropagateRequestV2,
    od_problem,
)

from adam_assist import ASSISTPropagator

CONFIG_FP = "a41a7bec3265e1dded511415b0991eb838f207230b1fd0a50cfcc4ac743aa70f"
DATA_FP = "f4ac12257d77c7e8dbafa0f24a0f029001b7144f0577bebab27a79c9aca10e69"
RESULTS: dict[str, object] = {}


def _capsule_fingerprint(provider: ASSISTPropagator) -> str:
    """Read a capsule-owned table while retaining the capsule owner."""
    capsule = provider.propagator_capsule()
    return bytes(_capsule_table(capsule).data_fingerprint).decode()


def _one_orbit() -> Orbits:
    return Orbits.from_kwargs(
        orbit_id=["kernel-binding"],
        coordinates=CartesianCoordinates.from_kwargs(
            x=[TRUTH_STATE[0]],
            y=[TRUTH_STATE[1]],
            z=[TRUTH_STATE[2]],
            vx=[TRUTH_STATE[3]],
            vy=[TRUTH_STATE[4]],
            vz=[TRUTH_STATE[5]],
            time=Timestamp.from_mjd([60000], scale="tdb"),
            origin=Origin.from_kwargs(code=["SUN"]),
            frame="ecliptic",
        ),
    )


_PyCapsule_New = ctypes.pythonapi.PyCapsule_New
_PyCapsule_New.restype = ctypes.py_object
_PyCapsule_New.argtypes = (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p)


def _ipc(table) -> bytes:
    arrow = table.table.combine_chunks()
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, arrow.schema) as writer:
        writer.write_table(arrow)
    return sink.getvalue().to_pybytes()


def _assert_exact(left, right) -> None:
    for a, b in zip(left, right, strict=True):
        assert _ipc(a) == _ipc(b)


class _Tampered:
    """A byte copy of the real ASSIST table, mutated, in a new capsule.

    The original capsule stays alive so the copied context remains valid.
    """

    def __init__(self, mutate) -> None:
        self.base_provider = ASSISTPropagator()
        self.base = self.base_provider.propagator_capsule()
        source = _capsule_table(self.base)
        self.table = _ExternalPropagatorV2.from_buffer_copy(source)
        mutate(self.table)
        self.capsule = _PyCapsule_New(ctypes.addressof(self.table), CAPSULE_NAME, None)


def _provider(capsule, name: str = "adam_assist", version: str = CARGO_VERSION):
    class Fake(Propagator):
        native_backend = NativeBackend(name, version, frozenset({"propagator_capsule"}))

        def propagator_capsule(self):
            return capsule

        def propagate_orbits(self, *args, **kwargs):
            raise AssertionError("Core must use the capsule")

        def generate_ephemeris(self, *args, **kwargs):
            raise AssertionError("Core must use the capsule")

    return Fake()


def _set(field, value):
    def mutate(table):
        setattr(table, field, value)

    return mutate


def _reserved(table):
    table.reserved[0] = 1


def _no_thread_safe(table):
    table.capabilities = table.capabilities & ~1


def _unknown_cap(table):
    table.capabilities = table.capabilities | (1 << 10)


def _stm_cap_without_callback(table):
    table.propagate_stm = None


TAMPER_CASES = {
    "uppercase_configuration_fingerprint": _set(
        "configuration_fingerprint", CONFIG_FP.upper().encode()
    ),
    "short_data_fingerprint": _set("data_fingerprint", DATA_FP[:63].encode()),
    "long_data_fingerprint": _set("data_fingerprint", (DATA_FP + "0").encode()),
    "nonhex_configuration_fingerprint": _set(
        "configuration_fingerprint", ("g" * 64).encode()
    ),
    "null_data_fingerprint": _set("data_fingerprint", None),
    "null_configuration_fingerprint": _set("configuration_fingerprint", None),
    "legacy_v2_size_without_suffix": _set("struct_size", 112),
    "partial_suffix_size": _set("struct_size", 120),
    "abi_version_1": _set("abi_version", 1),
    "abi_version_3": _set("abi_version", 3),
    "nonzero_reserved_word": _reserved,
    "missing_thread_safe_capability": _no_thread_safe,
    "unknown_capability_bit": _unknown_cap,
    "stm_capability_without_callback": _stm_cap_without_callback,
    "table_name_mismatch": _set("name", b"adam_assist_forged"),
    "table_version_mismatch": _set("version", b"0.5.0"),
    "two_body_flag_forged": _set("two_body_dynamics", 1),
    "two_body_flag_invalid": _set("two_body_dynamics", 7),
}


def test_control_copy_of_real_table_is_accepted(od_problem) -> None:
    tampered = _Tampered(lambda table: None)
    source, associations = _canonical_inputs(od_problem)
    fitted, members, failures = run_iod(
        source, associations, _provider(tampered.capsule), _od_config().for_iod()
    )
    assert len(failures) == 0 and len(fitted) == 1
    direct = run_iod(source, associations, ASSISTPropagator(), _od_config().for_iod())
    _assert_exact((fitted, members, failures), direct)


@pytest.mark.parametrize("case", sorted(TAMPER_CASES))
def test_tampered_table_is_rejected_before_work(od_problem, case) -> None:
    tampered = _Tampered(TAMPER_CASES[case])
    source, associations = _canonical_inputs(od_problem)
    with pytest.raises(Exception) as info:
        run_iod(
            source,
            associations,
            _provider(tampered.capsule),
            _od_config().for_iod(),
        )
    RESULTS[case] = f"{type(info.value).__name__}: {info.value}"
    print(f"\nTAMPER {case}: {type(info.value).__name__}: {info.value}")


def test_well_formed_forged_fingerprint_is_recorded_not_detected(od_problem) -> None:
    forged = hashlib.sha256(b"forged").hexdigest()
    tampered = _Tampered(_set("configuration_fingerprint", forged.encode()))
    source, associations = _canonical_inputs(od_problem)
    fitted, _, failures = run_iod(
        source, associations, _provider(tampered.capsule), _od_config().for_iod()
    )
    assert len(failures) == 0
    provider = fitted.provenance.iod_provider[0].as_py()
    assert forged in provider and CONFIG_FP not in provider


def test_declared_identity_mismatch_is_rejected(od_problem) -> None:
    capsule = ASSISTPropagator().propagator_capsule()
    source, associations = _canonical_inputs(od_problem)
    for name, version in (("adam_assist", "9.9.9"), ("other", CARGO_VERSION)):
        with pytest.raises(Exception) as info:
            run_iod(
                source,
                associations,
                _provider(capsule, name, version),
                _od_config().for_iod(),
            )
        print(f"\nDECLARED {name} {version}: {info.value}")


@pytest.mark.parametrize("propagator_jacobian", [False, True])
def test_exact_composition_evaluation_and_determinism(
    od_problem, propagator_jacobian
) -> None:
    source, associations = _canonical_inputs(od_problem)
    config = _od_config(propagator_jacobian=propagator_jacobian)
    propagator = ASSISTPropagator()
    iod = run_iod(source, associations, propagator, config.for_iod())
    assert len(iod[0]) == 1 and len(iod[2]) == 0
    composed = refine(iod[0], iod[1], source, propagator, config.for_refinement())
    fused = run_od(source, associations, propagator, config)
    _assert_exact(fused, composed)
    repeated = run_od(source, associations, ASSISTPropagator(), config)
    _assert_exact(fused, repeated)
    evaluated_fused = evaluate_orbits(
        fused[0], fused[1], source, propagator, config.for_evaluation()
    )
    evaluated_composed = evaluate_orbits(
        composed[0], composed[1], source, propagator, config.for_evaluation()
    )
    _assert_exact(evaluated_fused, evaluated_composed)
    provenance = fused[0].provenance
    for field in ("iod_provider", "refinement_provider"):
        provider = json.loads(getattr(provenance, field)[0].as_py())
        assert provider["name"] == "adam_assist"
        assert provider["version"] == CARGO_VERSION
        assert provider["abi_version"] == 2
        assert provider["capabilities"] == 7
        assert provider["known_capabilities"] == 7
        assert provider["identity_validated_by_capsule"] is True
        assert provider["integration_time_scale"] == "tdb"
        assert provider["two_body_dynamics"] is False
        assert provider["configuration_fingerprint"] == CONFIG_FP
        assert provider["data_fingerprint"] == DATA_FP
    # No ASSIST-owned profile/config crosses the boundary: the persisted
    # configuration is Core's and carries no provider-specific settings.
    configuration = provenance.configuration[0].as_py()
    assert "assist" not in configuration.lower()


def _two_group_inputs(od_problem):
    source, _ = _canonical_inputs(od_problem)
    bad_rows = source.table.slice(0, 2)
    bad_ids = pa.array([f"bad-{i}" for i in range(2)], type=pa.large_string())
    index = bad_rows.schema.get_field_index("id")
    bad = bad_rows.set_column(index, bad_rows.schema.field(index), bad_ids)
    combined = SourceCatalog.from_pyarrow(
        pa.concat_tables([source.table, bad]).combine_chunks()
    )
    associations = Associations.from_kwargs(
        detection_id=combined.id,
        object_id=["good"] * len(source) + ["bad"] * 2,
    )
    return source, combined, associations


def test_typed_failures_and_group_isolation(od_problem) -> None:
    source, combined, associations = _two_group_inputs(od_problem)
    config = _od_config(propagator_jacobian=True)
    propagator = ASSISTPropagator()
    fitted, _members, failures = run_od(combined, associations, propagator, config)
    assert len(fitted) == 1
    assert failures.group_id.to_pylist() == ["bad"]
    RESULTS["typed_failure"] = {
        "stage": failures.stage.to_pylist(),
        "code": failures.code.to_pylist(),
        "message": failures.message.to_pylist(),
    }
    print("\nTYPED FAILURE", RESULTS["typed_failure"])
    alone = run_od(
        source,
        Associations.from_kwargs(
            detection_id=source.id, object_id=["good"] * len(source)
        ),
        propagator,
        config,
    )
    np.testing.assert_array_equal(
        fitted.coordinates.values.view(np.uint64),
        alone[0].coordinates.values.view(np.uint64),
    )
    # Reordered association rows must produce the same canonical products.
    order = list(range(len(associations)))[::-1]
    reordered = associations.take(pa.array(order, type=pa.int64()))
    shuffled = run_od(combined, reordered, propagator, config)
    np.testing.assert_array_equal(
        shuffled[0].coordinates.values.view(np.uint64),
        fitted.coordinates.values.view(np.uint64),
    )
    assert shuffled[2].code.to_pylist() == failures.code.to_pylist()


def _call(capsule, states, epochs, target_days, nongrav=None, pairwise=False):
    table = _capsule_table(capsule)
    states = np.asarray(states, dtype=np.float64).reshape(-1, 6)
    n_orbits = len(states)
    n_times = len(target_days)
    rows = n_orbits if pairwise else n_orbits * n_times
    flat = np.ascontiguousarray(states).reshape(-1)
    states_buffer = (ctypes.c_double * len(flat))(*flat)
    ng = None
    if nongrav is not None:
        ng_flat = list(np.asarray(nongrav, dtype=np.float64).reshape(-1))
        ng = (ctypes.c_double * len(ng_flat))(*ng_flat)
    epoch_days = (ctypes.c_int64 * n_orbits)(*[int(e) for e in epochs])
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
        non_gravitational_parameters=ng,
        epoch_days=epoch_days,
        epoch_nanos=epoch_nanos,
        orbit_time_scale=b"tdb",
        frame=b"ecliptic",
        origin=b"SUN",
        n_times=n_times,
        time_days=times,
        time_nanos=nanos,
        time_scale=b"tdb",
        pairwise=1 if pairwise else 0,
        out_states=output,
        out_valid=valid,
        out_failure_codes=failures,
        error=ctypes.cast(error, ctypes.POINTER(ctypes.c_char)),
        error_len=len(error),
        reserved=(ctypes.c_size_t * 4)(0, 0, 0, 0),
    )
    callback = ctypes.CFUNCTYPE(
        ctypes.c_int32, ctypes.c_void_p, ctypes.POINTER(_PropagateRequestV2)
    )(table.propagate)
    status = callback(table.context, ctypes.byref(request))
    return (
        status,
        np.ctypeslib.as_array(output).copy().reshape(rows, 6),
        list(valid),
        list(failures),
        error.value.decode(),
    )


CUSTOM = [
    CUSTOM_NONGRAV[key][0] for key in ("A1", "A2", "A3", "ALN", "NK", "NM", "NN", "R0")
]


def test_capsule_batching_matches_singletons_mixed_nongrav_and_epochs() -> None:
    capsule = ASSISTPropagator().propagator_capsule()
    states = np.vstack(
        [
            TRUTH_STATE,
            TRUTH_STATE * [1.1, 1, 1, 1, 0.95, 1],
            TRUTH_STATE * [0.9, 1.2, 1, 1, 1.05, 1],
        ]
    )
    epochs = [60000, 60003, 59990]
    nongrav = [CUSTOM, [np.nan] * 8, [0.0, 0.0, 0.0] + CUSTOM[3:]]
    targets = [60030, 59995, 60010, 60001]
    status, batch, valid, codes, error = _call(
        capsule, states, epochs, targets, nongrav
    )
    assert (status, error) == (0, "") and all(valid) and not any(codes)
    assert np.isfinite(batch).all()
    for i in range(3):
        s, single, v, _c, e = _call(
            capsule,
            states[i : i + 1],
            epochs[i : i + 1],
            targets,
            [nongrav[i]],
        )
        assert (s, e) == (0, "")
        assert np.isfinite(single).all()
        np.testing.assert_array_equal(
            batch[i * len(targets) : (i + 1) * len(targets)].view(np.uint64),
            single.view(np.uint64),
        )
    # NaN-row and zero-acceleration rows equal gravity-only (null pointer).
    s, gravity, _, _, _ = _call(capsule, states[1:3], epochs[1:3], targets)
    assert s == 0 and np.isfinite(gravity).all()
    np.testing.assert_array_equal(batch[4:].view(np.uint64), gravity.view(np.uint64))
    # Pairwise mode: orbit i to target i.
    s, pairwise, v, _c, e = _call(
        capsule, states, epochs, targets[:3], nongrav, pairwise=True
    )
    assert (s, e) == (0, "") and all(v)
    assert np.isfinite(pairwise).all()
    # Pairwise integrates each orbit to one target only, so IAS15 step
    # sequences differ from the multi-target cross product: compare closely.
    pair_residuals = [
        np.abs(pairwise[i] - batch[i * len(targets) + i]) for i in range(3)
    ]
    assert all(np.isfinite(residual).all() for residual in pair_residuals)
    pair_diff = max(float(np.max(residual)) for residual in pair_residuals)
    assert np.isfinite(pair_diff)
    RESULTS["pairwise_vs_cross_product_max_abs"] = pair_diff
    print("\nPAIRWISE vs CROSS-PRODUCT max abs", pair_diff)
    assert pair_diff < 1e-12


def test_capsule_matches_public_propagate_orbits() -> None:
    propagator = ASSISTPropagator()
    capsule = propagator.propagator_capsule()
    targets = [60010, 60020]
    _status, states, _valid, _, _ = _call(capsule, [TRUTH_STATE], [60000], targets)
    orbit = Orbits.from_kwargs(
        orbit_id=["o"],
        coordinates=CartesianCoordinates.from_kwargs(
            x=[TRUTH_STATE[0]],
            y=[TRUTH_STATE[1]],
            z=[TRUTH_STATE[2]],
            vx=[TRUTH_STATE[3]],
            vy=[TRUTH_STATE[4]],
            vz=[TRUTH_STATE[5]],
            time=Timestamp.from_mjd([60000], scale="tdb"),
            origin=Origin.from_kwargs(code=["SUN"]),
            frame="ecliptic",
        ),
    )
    public = propagator.propagate_orbits(
        orbit, Timestamp.from_mjd(targets, scale="tdb")
    )
    values = public.coordinates.values
    assert np.isfinite(values).all() and np.isfinite(states).all()
    diff = float(np.max(np.abs(values - states)))
    RESULTS["capsule_vs_public_max_abs"] = diff
    print(
        "\nCAPSULE vs PUBLIC max abs", diff, "bitwise", np.array_equal(values, states)
    )
    assert diff < 1e-12


def test_stm_matches_finite_difference() -> None:
    capsule = ASSISTPropagator().propagator_capsule()
    status, _states, stms, valid, error = _call_capsule_stm(
        capsule, TRUTH_STATE, [60020], CUSTOM
    )
    assert (status, valid, error) == (0, [1], "")
    assert np.isfinite(stms).all()
    h = np.array([1e-7, 1e-7, 1e-7, 1e-9, 1e-9, 1e-9])
    fd = np.zeros((6, 6))
    for j in range(6):
        plus = TRUTH_STATE.copy()
        plus[j] += h[j]
        minus = TRUTH_STATE.copy()
        minus[j] -= h[j]
        _, sp, _, _, _ = _call(capsule, [plus], [60000], [60020], [CUSTOM])
        _, sm, _, _, _ = _call(capsule, [minus], [60000], [60020], [CUSTOM])
        assert np.isfinite(sp).all() and np.isfinite(sm).all()
        fd[:, j] = (sp[0] - sm[0]) / (2 * h[j])
    assert np.isfinite(fd).all()
    rel = np.max(np.abs(fd - stms[0]) / (np.abs(stms[0]) + 1e-12))
    assert np.isfinite(rel)
    RESULTS["stm_fd_rel"] = float(rel)
    print("\nSTM vs central FD max rel", rel)
    assert rel < 1e-4


def test_benign_hardlink_change_before_first_use_establishes_current_bytes() -> None:
    base = ASSISTPropagator()
    # Keep the temporary directory beside the installed kernels so this remains
    # a real hard-link test when /tmp is mounted on another filesystem.
    with tempfile.TemporaryDirectory(dir=os.path.dirname(base.planets_path)) as tmp:
        planets = os.path.join(tmp, "planets.bsp")
        asteroids = os.path.join(tmp, "asteroids.bsp")
        os.link(base.planets_path, planets)
        os.link(base.asteroids_path, asteroids)
        direct = ASSISTPropagator(planets_path=planets, asteroids_path=asteroids)
        capsule_first = ASSISTPropagator(planets_path=planets, asteroids_path=asteroids)

        # Reproduce uv's hardlink install after provider construction but
        # before either provider establishes content trust.
        os.link(planets, os.path.join(tmp, "planets-extra.bsp"))
        os.link(asteroids, os.path.join(tmp, "asteroids-extra.bsp"))

        orbit = Orbits.from_kwargs(
            orbit_id=["pretrust-hardlink"],
            coordinates=CartesianCoordinates.from_kwargs(
                x=[TRUTH_STATE[0]],
                y=[TRUTH_STATE[1]],
                z=[TRUTH_STATE[2]],
                vx=[TRUTH_STATE[3]],
                vy=[TRUTH_STATE[4]],
                vz=[TRUTH_STATE[5]],
                time=Timestamp.from_mjd([60000], scale="tdb"),
                origin=Origin.from_kwargs(code=["SUN"]),
                frame="ecliptic",
            ),
        )
        propagated = direct.propagate_orbits(
            orbit, Timestamp.from_mjd([60010], scale="tdb")
        )
        assert np.isfinite(propagated.coordinates.values).all()
        capsule_first_owner = capsule_first.propagator_capsule()
        direct_owner = direct.propagator_capsule()
        assert _capsule_table(capsule_first_owner).data_fingerprint == DATA_FP.encode()
        assert _capsule_table(direct_owner).data_fingerprint == DATA_FP.encode()


def _replace_byte(path: Path, offset: int, value: int) -> int:
    metadata = path.stat()
    with path.open("r+b") as handle:
        handle.seek(offset)
        original = handle.read(1)[0]
        handle.seek(offset)
        handle.write(bytes([value]))
        handle.flush()
        os.fsync(handle.fileno())
    os.utime(path, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
    return original


def _gm5_offset(path: Path) -> int:
    contents = path.read_bytes()
    start = contents.index(b"Initial conditions and constants")
    return contents.index(b"GM5           2.8253458252257917D-07", start) + 14


def test_pretrust_constant_binding_rejects_same_fingerprint_different_science() -> None:
    base = ASSISTPropagator()
    with tempfile.TemporaryDirectory() as tmp:
        planets = Path(tmp) / "planets.bsp"
        shutil.copyfile(base.planets_path, planets)
        loaded_original = ASSISTPropagator(
            planets_path=planets, asteroids_path=base.asteroids_path
        )
        offset = _gm5_offset(planets)
        original = _replace_byte(planets, offset, ord("3"))
        loaded_tampered = ASSISTPropagator(
            planets_path=planets, asteroids_path=base.asteroids_path
        )

        with pytest.raises(RuntimeError, match="bytes ASSIST loaded"):
            loaded_original.propagator_capsule()
        tampered_fingerprint = _capsule_fingerprint(loaded_tampered)
        assert tampered_fingerprint != DATA_FP

        _replace_byte(planets, offset, original)
        assert _capsule_fingerprint(loaded_original) == DATA_FP
        propagated = loaded_original.propagate_orbits(
            _one_orbit(), Timestamp.from_mjd([60400], scale="tdb")
        )
        assert np.isfinite(propagated.coordinates.values).all()
        with pytest.raises(
            RuntimeError,
            match="eager content changed after provenance was established",
        ):
            loaded_tampered.propagator_capsule()
        _replace_byte(planets, offset, ord("3"))
        assert _capsule_fingerprint(loaded_tampered) == tampered_fingerprint
        _replace_byte(planets, offset, original)


def test_pretrust_inode_replacement_rejects_then_original_inode_recovers() -> None:
    base = ASSISTPropagator()
    with tempfile.TemporaryDirectory() as tmp:
        planets = Path(tmp) / "planets.bsp"
        loaded = Path(tmp) / "planets.loaded.bsp"
        replacement = Path(tmp) / "planets.replacement.bsp"
        shutil.copyfile(base.planets_path, planets)
        provider = ASSISTPropagator(
            planets_path=planets, asteroids_path=base.asteroids_path
        )
        planets.rename(loaded)
        shutil.copyfile(loaded, planets)
        with pytest.raises(RuntimeError, match="canonical path was replaced"):
            provider.propagator_capsule()

        planets.rename(replacement)
        loaded.rename(planets)
        assert _capsule_fingerprint(provider) == DATA_FP
        replacement.unlink()

        posttrust_loaded = Path(tmp) / "planets.posttrust-loaded.bsp"
        posttrust_replacement = Path(tmp) / "planets.posttrust-replacement.bsp"
        planets.rename(posttrust_loaded)
        shutil.copyfile(posttrust_loaded, planets)
        with pytest.raises(
            RuntimeError,
            match="mapping identity changed after provenance was established.*reconstruct",
        ):
            provider.propagator_capsule()
        planets.rename(posttrust_replacement)
        posttrust_loaded.rename(planets)
        assert _capsule_fingerprint(provider) == DATA_FP


def test_pretrust_truncation_fails_before_propagation_without_sigbus() -> None:
    base = ASSISTPropagator()
    with tempfile.TemporaryDirectory() as tmp:
        planets = Path(tmp) / "planets.bsp"
        shutil.copyfile(base.planets_path, planets)
        code = (
            "from pathlib import Path;import os;"
            "from adam_assist import ASSISTPropagator;"
            "from adam_core.time import Timestamp;"
            "from test_provider_capsule_od import _one_orbit;"
            f"p=ASSISTPropagator(planets_path=Path({str(planets)!r}),"
            f"asteroids_path=Path({str(base.asteroids_path)!r}));"
            f"os.truncate({str(planets)!r},os.path.getsize({str(planets)!r})//2);"
            "errors=[];"
            "\nfor call in (lambda:p.propagator_capsule(),"
            "lambda:p.propagate_orbits(_one_orbit(),Timestamp.from_mjd([60030],scale='tdb'))):"
            "\n try: call()"
            "\n except Exception as error: errors.append(str(error))"
            "\nprint(errors);assert len(errors)==2;"
            "assert all('bytes ASSIST loaded' in error or 'DAF record' in error for error in errors)"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        assert result.returncode == 0, (result.stdout, result.stderr, result.returncode)


@pytest.mark.skipif(sys.platform != "linux", reason="requires Linux ctime semantics")
def test_oscillating_writer_has_no_provenance_science_breach() -> None:
    attempts = 900
    base = ASSISTPropagator()
    with tempfile.TemporaryDirectory() as tmp:
        planets = Path(tmp) / "planets.bsp"
        stop = Path(tmp) / "stop"
        shutil.copyfile(base.planets_path, planets)
        offset = _gm5_offset(planets)
        with planets.open("rb") as handle:
            handle.seek(offset)
            assert handle.read(1) == b"2"
        reference_provider = ASSISTPropagator(
            planets_path=planets, asteroids_path=base.asteroids_path
        )
        reference = reference_provider.propagate_orbits(
            _one_orbit(), Timestamp.from_mjd([60400], scale="tdb")
        ).coordinates.values

        mutator_code = (
            "import os,pathlib,sys;"
            "p=sys.argv[1];off=int(sys.argv[2]);stop=pathlib.Path(sys.argv[3]);"
            "fd=os.open(p,os.O_WRONLY);i=0;"
            "\nwhile True:"
            "\n os.pwrite(fd,b'3' if i%2==0 else b'2',off);i+=1"
            "\n if i%4096==0 and stop.exists(): break"
            "\nos.pwrite(fd,b'2',off);os.fsync(fd);os.close(fd);print(i)"
        )
        mutator = subprocess.Popen(
            [sys.executable, "-c", mutator_code, str(planets), str(offset), str(stop)],
            stdout=subprocess.PIPE,
            text=True,
        )
        time.sleep(0.1)
        providers: list[ASSISTPropagator] = []
        errors = 0
        try:
            for _ in range(attempts):
                try:
                    providers.append(
                        ASSISTPropagator(
                            planets_path=planets,
                            asteroids_path=base.asteroids_path,
                        )
                    )
                except RuntimeError as error:
                    message = str(error)
                    assert (
                        "metadata changed concurrently during ephemeris construction"
                        in message
                        or "eagerly parsed content changed during ephemeris construction"
                        in message
                    )
                    errors += 1
        finally:
            stop.touch()
            toggles = int(mutator.communicate(timeout=30)[0])
        with planets.open("rb") as handle:
            handle.seek(offset)
            assert handle.read(1) == b"2"

        canonical = rejected = breaches = 0
        for provider in providers:
            try:
                fingerprint = _capsule_fingerprint(provider)
            except RuntimeError as error:
                assert "eager content" in str(error)
                rejected += 1
                continue
            assert fingerprint == DATA_FP
            canonical += 1
            propagated = provider.propagate_orbits(
                _one_orbit(), Timestamp.from_mjd([60400], scale="tdb")
            ).coordinates.values
            assert np.isfinite(propagated).all()
            if not np.array_equal(propagated, reference):
                breaches += 1

        assert errors + len(providers) == attempts
        assert breaches == 0
        print(
            "oscillating construction race: "
            f"attempts={attempts} errors={errors} successes={len(providers)} "
            f"canonical={canonical} rejected={rejected} breaches={breaches} "
            f"toggles={toggles}"
        )
        retry = ASSISTPropagator(
            planets_path=planets, asteroids_path=base.asteroids_path
        )
        assert _capsule_fingerprint(retry) == DATA_FP


def test_data_fingerprint_is_content_identity_lazy_and_cached() -> None:
    base = ASSISTPropagator()
    with tempfile.TemporaryDirectory() as tmp:
        planets = os.path.join(tmp, "planets.bsp")
        asteroids = os.path.join(tmp, "asteroids.bsp")
        os.symlink(base.planets_path, planets)
        os.symlink(base.asteroids_path, asteroids)
        linked = ASSISTPropagator(planets_path=planets, asteroids_path=asteroids)
        base_capsule = base.propagator_capsule()
        linked_capsule = linked.propagator_capsule()
        a = _capsule_table(base_capsule)
        b = _capsule_table(linked_capsule)
        assert a.data_fingerprint == b.data_fingerprint == DATA_FP.encode()
        base_capsule_again = base.propagator_capsule()
        assert _capsule_table(base_capsule_again).data_fingerprint == DATA_FP.encode()

    # Construction must only load ASSIST/SPICE. In this fresh process the first
    # actual use is a capsule request, which performs the intentionally
    # expensive content hash; the next request reuses it.
    code = (
        "import json,time;from adam_assist import ASSISTPropagator;"
        "t=time.perf_counter();p=ASSISTPropagator();t1=time.perf_counter();"
        "q=ASSISTPropagator();t2=time.perf_counter();"
        "p.propagator_capsule();t3=time.perf_counter();"
        "p.propagator_capsule();t4=time.perf_counter();"
        "print(json.dumps([t1-t,t2-t1,t3-t2,t4-t3]))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    first_ctor, second_ctor, first_metadata, cached_metadata = json.loads(out.stdout)
    RESULTS["construction_and_metadata_seconds"] = {
        "first_constructor": first_ctor,
        "second_constructor": second_ctor,
        "first_metadata": first_metadata,
        "cached_metadata": cached_metadata,
    }
    assert first_ctor < 1.0
    assert second_ctor < 1.0
    assert cached_metadata < 0.25
