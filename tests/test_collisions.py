from adam_core.orbits import Orbits
from src.adam_core.propagator.adam_assist import ASSISTPropagator, CollisionConditions

IMPACTOR_FILE_PATH_60 = "tests/data/I00007_orbit.parquet"
# Contains a likely impactor with 100% chance of impact in 30 days
IMPACTOR_FILE_PATH_100 = "tests/data/I00008_orbit.parquet"
# Contains a likely impactor with 0% chance of impact in 30 days
IMPACTOR_FILE_PATH_0 = "tests/data/I00009_orbit.parquet"

def test_detect_collisions():
    orbits = Orbits.from_parquet(IMPACTOR_FILE_PATH_100)[0]
    propagator = ASSISTPropagator()

    collision_conditions = CollisionConditions.from_kwargs(
        collision_object_name=["Earth"],
        collision_distance=[7000],
        stopping_condition=[True],
    )
    results, collisions = propagator._detect_collisions(orbits, 60, collision_conditions)

    assert len(collisions) == 1
    assert collisions.distance.to_numpy()[0] <= 7000

    collision_conditions = CollisionConditions.from_kwargs(
        collision_object_name=["Earth", "Earth"],
        collision_distance=[10000, 7000],
        stopping_condition=[False, True],
    )
    results, collisions = propagator._detect_collisions(orbits, 60, collision_conditions)

    assert len(collisions) > 1