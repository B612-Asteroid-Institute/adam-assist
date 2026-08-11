import numpy as np
import pyarrow as pa
import quivr as qv
from adam_core.constants import KM_P_AU
from adam_core.coordinates import Origin
from adam_core.orbits import Orbits
from adam_core.orbits.non_gravitational_parameters import NonGravitationalParameters

from src.adam_assist.propagator import ASSISTPropagator, CollisionConditions

IMPACTOR_FILE_PATH_60 = "tests/data/I00007_orbit.parquet"
# Contains a likely impactor with 100% chance of impact in 30 days
IMPACTOR_FILE_PATH_100 = "tests/data/I00008_orbit.parquet"
# Contains a likely impactor with 0% chance of impact in 30 days
IMPACTOR_FILE_PATH_0 = "tests/data/I00009_orbit.parquet"


def test_detect_collisions():
    orbits = Orbits.from_parquet(IMPACTOR_FILE_PATH_100)[0]
    propagator = ASSISTPropagator()

    collision_conditions = CollisionConditions.from_kwargs(
        condition_id=["Default - Earth"],
        collision_object=Origin.from_kwargs(code=["EARTH"]),
        collision_distance=[7000],
        stopping_condition=[True],
    )
    _results, collisions = propagator._detect_collisions(
        orbits, 60, collision_conditions
    )

    assert len(collisions) == 1
    assert collisions.collision_coordinates.rho.to_numpy()[0] <= 7000 / KM_P_AU

    collision_conditions = CollisionConditions.from_kwargs(
        condition_id=["Default - Earth", "Default - Earth"],
        collision_object=Origin.from_kwargs(code=["EARTH", "EARTH"]),
        collision_distance=[10000, 7000],
        stopping_condition=[False, True],
    )
    _results, collisions = propagator._detect_collisions(
        orbits, 60, collision_conditions
    )

    assert len(collisions) > 1


def test_detect_collisions_mixed_marsden_laws_match_partitioned_runs():
    """
    A single-epoch batch mixing an impacting comet-law row, a custom-tuple
    row, an inverse-square row, and a force-free row must partition into
    per-law simulations whose results AND collision events exactly match
    running each partition's rows directly (identical simulation contents
    are deterministic, so the comparison is bitwise).
    """
    impactor = Orbits.from_parquet(IMPACTOR_FILE_PATH_100)[0]
    epoch = impactor.coordinates.time

    def _with_nongrav(
        base: Orbits, orbit_id: str, offset_au: float, **nongrav
    ) -> Orbits:
        coords = base.coordinates
        shifted = coords.set_column(
            "x", pa.array([coords.x[0].as_py() + offset_au], type=pa.float64())
        )
        columns = {
            name: [nongrav.get(name)]
            for name in ("A1", "A2", "A3", "ALN", "NK", "NM", "NN", "R0")
        }
        out = base.set_column("coordinates", shifted)
        out = out.set_column("orbit_id", pa.array([orbit_id], type=pa.large_string()))
        return out.set_column(
            "non_gravitational_parameters",
            NonGravitationalParameters.from_kwargs(
                source=["TEST"] if nongrav else [None], **columns
            ),
        )

    # The impacting trajectory carries the comet-standard law (with a
    # negligible A1 so the impact is preserved), putting it ALONE in its
    # partition: its stopping condition then cannot truncate other rows.
    # The inverse-square and force-free rows share the defaults partition
    # and ride offset copies that never encounter the Earth.
    rows = [
        _with_nongrav(
            impactor,
            "impactor-comet-standard",
            0.0,
            A1=1.0e-13,
            ALN=0.1112620426,
            NK=4.6142,
            NM=2.15,
            NN=5.093,
            R0=2.808,
        ),
        _with_nongrav(
            impactor,
            "custom-tuple",
            -0.35,
            A1=2.8e-10,
            ALN=0.0408373333128795,
            NK=2.6,
            NM=2.0,
            NN=3.0,
            R0=5.0,
        ),
        _with_nongrav(impactor, "inverse-square", 0.35, A2=-1e-13),
        _with_nongrav(impactor, "force-free", 0.55),
    ]
    assert all(orbit.coordinates.time.equals(epoch) for orbit in rows)

    propagator = ASSISTPropagator()
    conditions = CollisionConditions.from_kwargs(
        condition_id=["Default - Earth"],
        collision_object=Origin.from_kwargs(code=["EARTH"]),
        collision_distance=[7000],
        stopping_condition=[True],
    )

    # The batch must partition into [impactor], [custom-tuple],
    # [inverse-square + force-free] (the latter two share the canonical
    # defaults law). Each partition is integrated as one simulation, so
    # running the same groups directly must reproduce the batch EXACTLY:
    # adaptive-step end times are trajectory-dependent, which makes
    # identical-simulation comparison the only bitwise-stable control.
    groups = [rows[0], rows[1], qv.concatenate(rows[2:])]
    expected = {}
    for group in groups:
        results, events = propagator._detect_collisions(group, 60, conditions)
        for orbit_id in group.orbit_id.to_pylist():
            row_mask = np.array(
                [oid == orbit_id for oid in results.orbit_id.to_pylist()], dtype=bool
            )
            event_mask = np.array(
                [oid == orbit_id for oid in events.orbit_id.to_pylist()], dtype=bool
            )
            expected[orbit_id] = (
                results.apply_mask(row_mask),
                events.apply_mask(event_mask),
            )

    batch_results, batch_events = propagator._detect_collisions(
        qv.concatenate(rows), 60, conditions
    )

    assert len(batch_results) == len(rows)
    # The impactor must actually produce an event so event concatenation
    # across partitions is exercised, and the pure-gravity rows must not.
    assert len(expected["impactor-comet-standard"][1]) >= 1
    assert len(expected["force-free"][1]) == 0

    for orbit_id, (group_results, group_events) in expected.items():
        mask = np.array([oid == orbit_id for oid in batch_results.orbit_id.to_pylist()])
        batch_row = batch_results.apply_mask(mask)
        assert len(batch_row) == 1
        np.testing.assert_array_equal(
            batch_row.coordinates.time.mjd().to_numpy(zero_copy_only=False),
            group_results.coordinates.time.mjd().to_numpy(zero_copy_only=False),
            err_msg=f"{orbit_id}: batch and per-partition end times differ",
        )
        np.testing.assert_array_equal(
            batch_row.coordinates.r,
            group_results.coordinates.r,
            err_msg=f"{orbit_id}: batch and per-partition positions differ",
        )

        event_mask = np.array(
            [oid == orbit_id for oid in batch_events.orbit_id.to_pylist()], dtype=bool
        )
        batch_orbit_events = batch_events.apply_mask(event_mask)
        assert len(batch_orbit_events) == len(group_events), (
            f"{orbit_id}: batch recorded {len(batch_orbit_events)} events, "
            f"per-partition run recorded {len(group_events)}"
        )
        if len(group_events) > 0:
            np.testing.assert_array_equal(
                batch_orbit_events.coordinates.time.mjd().to_numpy(
                    zero_copy_only=False
                ),
                group_events.coordinates.time.mjd().to_numpy(zero_copy_only=False),
                err_msg=f"{orbit_id}: batch and per-partition event times differ",
            )
