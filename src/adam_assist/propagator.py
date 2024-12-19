import hashlib
from ctypes import c_uint32
from typing import Dict, List, Tuple, Union

import assist
import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv
import rebound
from adam_core.constants import KM_P_AU
from adam_core.constants import Constants as c
from adam_core.coordinates import CartesianCoordinates, Origin, transform_coordinates
from adam_core.coordinates.origin import OriginCodes
from adam_core.dynamics.impacts import EarthImpacts, ImpactMixin
from adam_core.orbits import Orbits
from adam_core.orbits.variants import VariantOrbits
from adam_core.propagator.propagator import OrbitType, Propagator, TimestampType
from adam_core.time import Timestamp
from jpl_small_bodies_de441_n16 import de441_n16
from naif_de440 import de440
from quivr.concat import concatenate

C = c.C

try:
    from adam_assist.version import __version__
except ImportError:
    __version__ = "0.0.0"

# Use the Earth's equatorial radius as used in DE4XX ephemerides
# adam_core defines it in au but we need it in km
EARTH_RADIUS_KM = c.R_EARTH_EQUATORIAL * KM_P_AU


def uint32_hash(s: str) -> c_uint32:
    sha256_result = hashlib.sha256(s.encode()).digest()
    # Get the first 4 bytes of the SHA256 hash to obtain a uint32 value.
    return c_uint32(int.from_bytes(sha256_result[:4], byteorder="big"))


def hash_orbit_ids_to_uint32(
    # orbit_ids: np.ndarray[Tuple[np.dtype[np.int_]], np.dtype[np.str_]],
    orbit_ids: npt.NDArray[np.str_],
) -> Tuple[Dict[int, str], List[c_uint32]]:
    """
    Derive uint32 hashes from orbit id strigns

    Rebound uses uint32 to track individual particles, but we use orbit id strings.
    Here we attempt to generate uint32 hashes for each and return the mapping as well.
    """
    hashes = [uint32_hash(o) for o in orbit_ids]
    # Because uint32 is an unhashable type,
    # we use a dict mapping from uint32 to orbit id string
    mapping = {hashes[i].value: orbit_ids[i] for i in range(len(orbit_ids))}

    return mapping, hashes


class ASSISTPropagator(Propagator, ImpactMixin):  # type: ignore

    def __init__(
        self,
        *args: object,  # Generic type for arbitrary positional arguments
        min_dt: float = 1e-15,
        initial_dt: float = 0.001,
        adaptive_mode: int = 2,
        epsilon: float = 1e-9,
        **kwargs: object,  # Generic type for arbitrary keyword arguments
    ) -> None:
        super().__init__(*args, **kwargs)
        if min_dt <= 0:
            raise ValueError("min_dt must be positive")
        if initial_dt <= 0:
            raise ValueError("initial_dt must be positive")
        if min_dt > initial_dt:
            raise ValueError("min_dt must be smaller than initial_dt")
        self.min_dt = min_dt
        self.initial_dt = initial_dt
        self.adaptive_mode = adaptive_mode
        self.epsilon = epsilon

    def _propagate_orbits(self, orbits: OrbitType, times: TimestampType) -> OrbitType:
        """
        Propagate the orbits to the specified times.
        """
        # The coordinate frame is the equatorial International Celestial Reference Frame (ICRF).
        # This is also the native coordinate system for the JPL binary files.
        # For units we use solar masses, astronomical units, and days.
        # The time coordinate is Barycentric Dynamical Time (TDB) in Julian days.
        # Convert coordinates to ICRF using TDB time
        transformed_coords = transform_coordinates(
            orbits.coordinates,
            origin_out=OriginCodes.SOLAR_SYSTEM_BARYCENTER,
            frame_out="equatorial",
        )
        transformed_input_orbit_times = transformed_coords.time.rescale("tdb")
        transformed_coords = transformed_coords.set_column(
            "time", transformed_input_orbit_times
        )
        transformed_orbits = orbits.set_column("coordinates", transformed_coords)

        # Group orbits by unique time, then propagate them
        results = None
        unique_times = transformed_orbits.coordinates.time.unique()
        for epoch in unique_times:
            mask = transformed_orbits.coordinates.time.equals(epoch)
            epoch_orbits = transformed_orbits.apply_mask(mask)
            propagated_orbits = self._propagate_orbits_inner(epoch_orbits, times)
            if results is None:
                results = propagated_orbits
            else:
                results = concatenate([results, propagated_orbits])

        # Sanity check that the results are of the correct type
        assert isinstance(results, OrbitType)

        return results

    def _propagate_orbits_inner(
        self, orbits: OrbitType, times: TimestampType
    ) -> OrbitType:
        """
        Propagates one or more orbits with the same epoch to the specified times.
        """
        ephem = assist.Ephem(
            planets_path=de440,
            asteroids_path=de441_n16,
        )
        sim = None
        sim = rebound.Simulation()

        # Set the simulation time, relative to the jd_ref
        start_tdb_time = orbits.coordinates.time.jd().to_numpy()[0]
        start_tdb_time = start_tdb_time - ephem.jd_ref
        sim.t = start_tdb_time

        particle_ids = orbits.orbit_id.to_numpy(zero_copy_only=False)

        # Serialize the variantorbit
        if isinstance(orbits, VariantOrbits):
            orbit_ids = orbits.orbit_id.to_numpy(zero_copy_only=False).astype(str)
            variant_ids = orbits.variant_id.to_numpy(zero_copy_only=False).astype(str)
            # Use numpy string operations to concatenate the orbit_id and variant_id
            particle_ids = np.char.add(
                np.char.add(orbit_ids, np.repeat("-", len(orbit_ids))), variant_ids
            )
            particle_ids = np.array(particle_ids, dtype="object")

        orbit_id_mapping, uint_orbit_ids = hash_orbit_ids_to_uint32(particle_ids)

        # Add the orbits as particles to the simulation
        coords_df = orbits.coordinates.to_dataframe()

        assist.Extras(sim, ephem)

        for i in range(len(coords_df)):
            sim.add(
                x=coords_df.x[i],
                y=coords_df.y[i],
                z=coords_df.z[i],
                vx=coords_df.vx[i],
                vy=coords_df.vy[i],
                vz=coords_df.vz[i],
                hash=uint_orbit_ids[i],
            )

        # Set the integrator parameters
        sim.dt = self.initial_dt
        sim.ri_ias15.min_dt = self.min_dt
        sim.ri_ias15.adaptive_mode = self.adaptive_mode
        sim.ri_ias15.epsilon = self.epsilon

        # Prepare the times as jd - jd_ref
        integrator_times = times.rescale("tdb").jd()
        integrator_times = pc.subtract(integrator_times, ephem.jd_ref)
        integrator_times = integrator_times.to_numpy()

        results = None

        # Step through each time, move the simulation forward and
        # collect the results.
        for i in range(len(integrator_times)):
            sim.integrate(integrator_times[i])

            # Get serialized particle data as numpy arrays
            orbit_id_hashes = np.zeros(sim.N, dtype="uint32")
            step_xyzvxvyvz = np.zeros((sim.N, 6), dtype="float64")

            sim.serialize_particle_data(xyzvxvyvz=step_xyzvxvyvz, hash=orbit_id_hashes)

            if isinstance(orbits, Orbits):
                # Retrieve original orbit id from hash
                orbit_ids = [orbit_id_mapping[h] for h in orbit_id_hashes]
                time_step_results = Orbits.from_kwargs(
                    coordinates=CartesianCoordinates.from_kwargs(
                        x=step_xyzvxvyvz[:, 0],
                        y=step_xyzvxvyvz[:, 1],
                        z=step_xyzvxvyvz[:, 2],
                        vx=step_xyzvxvyvz[:, 3],
                        vy=step_xyzvxvyvz[:, 4],
                        vz=step_xyzvxvyvz[:, 5],
                        time=Timestamp.from_jd(
                            pa.repeat(sim.t + ephem.jd_ref, sim.N), scale="tdb"
                        ),
                        origin=Origin.from_kwargs(
                            code=pa.repeat(
                                "SOLAR_SYSTEM_BARYCENTER",
                                sim.N,
                            )
                        ),
                        frame="equatorial",
                    ),
                    orbit_id=orbit_ids,
                    object_id=orbits.object_id,
                )
            elif isinstance(orbits, VariantOrbits):
                # Retrieve the orbit id and weights from hash
                particle_ids = [orbit_id_mapping[h] for h in orbit_id_hashes]
                orbit_ids, variant_ids = zip(
                    *[particle_id.split("-") for particle_id in particle_ids]
                )

                time_step_results = VariantOrbits.from_kwargs(
                    orbit_id=orbit_ids,
                    variant_id=variant_ids,
                    object_id=orbits.object_id,
                    weights=orbits.weights,
                    weights_cov=orbits.weights_cov,
                    coordinates=CartesianCoordinates.from_kwargs(
                        x=step_xyzvxvyvz[:, 0],
                        y=step_xyzvxvyvz[:, 1],
                        z=step_xyzvxvyvz[:, 2],
                        vx=step_xyzvxvyvz[:, 3],
                        vy=step_xyzvxvyvz[:, 4],
                        vz=step_xyzvxvyvz[:, 5],
                        time=Timestamp.from_jd(
                            pa.repeat(sim.t + ephem.jd_ref, sim.N), scale="tdb"
                        ),
                        origin=Origin.from_kwargs(
                            code=pa.repeat(
                                "SOLAR_SYSTEM_BARYCENTER",
                                sim.N,
                            )
                        ),
                        frame="equatorial",
                    ),
                )

            if results is None:
                results = time_step_results
            else:
                results = concatenate([results, time_step_results])

        # Store the last simulation in a private variable for reference
        self._last_simulation = sim
        return results

    def _detect_impacts(
        self, orbits: OrbitType, num_days: int
    ) -> Tuple[VariantOrbits, EarthImpacts]:
        # Assert that the time for each orbit definition is the same for the simulator to work
        assert len(pc.unique(orbits.coordinates.time.mjd())) == 1

        # The coordinate frame is the equatorial International Celestial Reference Frame (ICRF).
        # This is also the native coordinate system for the JPL binary files.
        # For units we use solar masses, astronomical units, and days.
        # The time coordinate is Barycentric Dynamical Time (TDB) in Julian days.

        # Convert coordinates to ICRF using TDB time
        coords = transform_coordinates(
            orbits.coordinates,
            origin_out=OriginCodes.SOLAR_SYSTEM_BARYCENTER,
            frame_out="equatorial",
        )
        input_orbit_times = coords.time.rescale("tdb")
        coords = coords.set_column("time", input_orbit_times)
        orbits = orbits.set_column("coordinates", coords)

        ephem = assist.Ephem(
            planets_path=de440,
            asteroids_path=de441_n16,
        )
        sim = None
        sim = rebound.Simulation()

        backward_propagation = num_days < 0

        # Set the simulation time, relative to the jd_ref
        start_tdb_time = orbits.coordinates.time.jd().to_numpy()[0]
        start_tdb_time = start_tdb_time - ephem.jd_ref
        sim.t = start_tdb_time

        particle_ids = orbits.orbit_id.to_numpy(zero_copy_only=False)

        # Serialize the variantorbit
        if isinstance(orbits, VariantOrbits):
            orbit_ids = orbits.orbit_id.to_numpy(zero_copy_only=False).astype(str)
            variant_ids = orbits.variant_id.to_numpy(zero_copy_only=False).astype(str)
            # Use numpy string operations to concatenate the orbit_id and variant_id
            particle_ids = np.char.add(
                np.char.add(orbit_ids, np.repeat("-", len(orbit_ids))), variant_ids
            )
            particle_ids = np.array(particle_ids, dtype="object")

        orbit_id_mapping, uint_orbit_ids = hash_orbit_ids_to_uint32(particle_ids)

        # Add the orbits as particles to the simulation
        coords_df = orbits.coordinates.to_dataframe()

        # ASSIST _must_ be initialized before adding particles
        assist.Extras(sim, ephem)

        for i in range(len(coords_df)):
            sim.add(
                x=coords_df.x[i],
                y=coords_df.y[i],
                z=coords_df.z[i],
                vx=coords_df.vx[i],
                vy=coords_df.vy[i],
                vz=coords_df.vz[i],
                hash=uint_orbit_ids[i],
            )

        # Prepare the times as jd - jd_ref
        final_integrator_time = (
            orbits.coordinates.time.add_days(num_days).jd().to_numpy()[0]
        )
        final_integrator_time = final_integrator_time - ephem.jd_ref

        # Results stores the final positions of the objects
        # If an object is an impactor, this represents its position at impact time
        results = None
        earth_impacts = None
        past_integrator_time = False
        time_step_results: Union[None, OrbitType] = None

        # Set the integrator parameters
        sim.dt = self.initial_dt
        sim.ri_ias15.min_dt = self.min_dt
        sim.ri_ias15.adaptive_mode = self.adaptive_mode
        sim.ri_ias15.epsilon = self.epsilon

        if backward_propagation:
            sim.dt = sim.dt * -1

        # Step through each time, move the simulation forward and
        # collect the results. End if all orbits are removed from
        # the simulation or the final integrator time is reached.
        while past_integrator_time is False and len(orbits) > 0:
            sim.steps(1)
            if (sim.t >= final_integrator_time and not backward_propagation) or (
                backward_propagation and sim.t <= final_integrator_time
            ):
                past_integrator_time = True

            # Get serialized particle data as numpy arrays
            orbit_id_hashes = np.zeros(sim.N, dtype="uint32")
            step_xyzvxvyvz = np.zeros((sim.N, 6), dtype="float64")

            sim.serialize_particle_data(xyzvxvyvz=step_xyzvxvyvz, hash=orbit_id_hashes)

            if isinstance(orbits, Orbits):
                # Retrieve original orbit id from hash
                orbit_ids = [orbit_id_mapping[h] for h in orbit_id_hashes]
                time_step_results = Orbits.from_kwargs(
                    coordinates=CartesianCoordinates.from_kwargs(
                        x=step_xyzvxvyvz[:, 0],
                        y=step_xyzvxvyvz[:, 1],
                        z=step_xyzvxvyvz[:, 2],
                        vx=step_xyzvxvyvz[:, 3],
                        vy=step_xyzvxvyvz[:, 4],
                        vz=step_xyzvxvyvz[:, 5],
                        time=Timestamp.from_jd(
                            pa.repeat(sim.t + ephem.jd_ref, sim.N), scale="tdb"
                        ),
                        origin=Origin.from_kwargs(
                            code=pa.repeat(
                                "SOLAR_SYSTEM_BARYCENTER",
                                sim.N,
                            )
                        ),
                        frame="equatorial",
                    ),
                    orbit_id=orbit_ids,
                    object_id=orbits.object_id,
                )
            elif isinstance(orbits, VariantOrbits):
                # Retrieve the orbit id and weights from hash
                particle_ids = [orbit_id_mapping[h] for h in orbit_id_hashes]
                orbit_ids, variant_ids = zip(
                    *[particle_id.split("-") for particle_id in particle_ids]
                )

                # Historically we've done a check here to make sure the orbit of the orbits
                # and serialized particles is consistent
                # np.testing.assert_array_equal(orbits.orbit_id.to_numpy(zero_copy_only=False).astype(str), orbit_ids)
                # np.testing.assert_array_equal(orbits.variant_id.to_numpy(zero_copy_only=False).astype(str), variant_ids)

                time_step_results = VariantOrbits.from_kwargs(
                    orbit_id=orbit_ids,
                    variant_id=variant_ids,
                    object_id=orbits.object_id,
                    weights=orbits.weights,
                    weights_cov=orbits.weights_cov,
                    coordinates=CartesianCoordinates.from_kwargs(
                        x=step_xyzvxvyvz[:, 0],
                        y=step_xyzvxvyvz[:, 1],
                        z=step_xyzvxvyvz[:, 2],
                        vx=step_xyzvxvyvz[:, 3],
                        vy=step_xyzvxvyvz[:, 4],
                        vz=step_xyzvxvyvz[:, 5],
                        time=Timestamp.from_jd(
                            pa.repeat(sim.t + ephem.jd_ref, sim.N), scale="tdb"
                        ),
                        origin=Origin.from_kwargs(
                            code=pa.repeat(
                                "SOLAR_SYSTEM_BARYCENTER",
                                sim.N,
                            )
                        ),
                        frame="equatorial",
                    ),
                )

            assert isinstance(time_step_results, OrbitType)
            time_step_results = time_step_results.set_column(
                "coordinates",
                transform_coordinates(
                    time_step_results.coordinates,
                    origin_out=OriginCodes.SUN,
                    frame_out="ecliptic",
                ),
            )

            # Get the Earth's position at the current time
            # earth_geo = get_perturber_state(OriginCodes.EARTH, results.coordinates.time[0], origin=OriginCodes.SUN)
            # diff = time_step_results.coordinates.values - earth_geo.coordinates.values
            earth_geo = ephem.get_particle("Earth", sim.t)
            earth_geo = CartesianCoordinates.from_kwargs(
                x=[earth_geo.x],
                y=[earth_geo.y],
                z=[earth_geo.z],
                vx=[earth_geo.vx],
                vy=[earth_geo.vy],
                vz=[earth_geo.vz],
                time=Timestamp.from_jd([sim.t + ephem.jd_ref], scale="tdb"),
                origin=Origin.from_kwargs(
                    code=["SOLAR_SYSTEM_BARYCENTER"],
                ),
                frame="equatorial",
            )
            earth_geo = transform_coordinates(
                earth_geo,
                origin_out=OriginCodes.SUN,
                frame_out="ecliptic",
            )
            diff = time_step_results.coordinates.values - earth_geo.values

            # Calculate the distance in KM
            # We use the IAU definition of the astronomical unit (149_597_870.7 km)
            normalized_distance = np.linalg.norm(diff[:, :3], axis=1) * KM_P_AU

            # Calculate which particles are within an Earth radius
            within_radius = normalized_distance < EARTH_RADIUS_KM

            # If any are within our earth radius, we record the impact
            # and do bookkeeping to remove the particle from the simulation
            if np.any(within_radius):
                distances = normalized_distance[within_radius]
                impacting_orbits = time_step_results.apply_mask(within_radius)

                if isinstance(orbits, VariantOrbits):
                    new_impacts = EarthImpacts.from_kwargs(
                        orbit_id=impacting_orbits.orbit_id,
                        distance=distances,
                        coordinates=impacting_orbits.coordinates,
                        variant_id=impacting_orbits.variant_id,
                    )
                elif isinstance(orbits, Orbits):
                    new_impacts = EarthImpacts.from_kwargs(
                        orbit_id=impacting_orbits.orbit_id,
                        distance=distances,
                        coordinates=impacting_orbits.coordinates,
                    )
                if earth_impacts is None:
                    earth_impacts = new_impacts
                else:
                    earth_impacts = qv.concatenate([earth_impacts, new_impacts])

                # Remove the particle from the simulation, orbits, and store in results
                for hash_id in orbit_id_hashes[within_radius]:
                    sim.remove(hash=c_uint32(hash_id))
                    # For some reason, it fails if we let rebound convert the hash to c_uint32

                # Remove the particle from the input / running orbits
                # This allows us to carry through object_id, weights, and weights_cov
                orbits = orbits.apply_mask(~within_radius)
                # Put the orbits / variants of the impactors into the results set
                if results is None:
                    results = impacting_orbits
                else:
                    results = qv.concatenate([results, impacting_orbits])

        # Add the final positions of the particles that are not already in the results
        if time_step_results is not None:
            if results is None:
                results = time_step_results
            else:
                if isinstance(orbits, Orbits):
                    still_in_simulation = pc.invert(
                        pc.is_in(time_step_results.orbit_id, results.orbit_id)
                    )
                elif isinstance(orbits, VariantOrbits):
                    still_in_simulation = pc.invert(
                        pc.is_in(time_step_results.variant_id, results.variant_id)
                    )
                results = qv.concatenate(
                    [results, time_step_results.apply_mask(still_in_simulation)]
                )

        if earth_impacts is None:
            earth_impacts = EarthImpacts.from_kwargs(
                orbit_id=[],
                distance=[],
                coordinates=CartesianCoordinates.from_kwargs(
                    x=[],
                    y=[],
                    z=[],
                    vx=[],
                    vy=[],
                    vz=[],
                    time=Timestamp.from_jd([], scale="tdb"),
                    origin=Origin.from_kwargs(
                        code=[],
                    ),
                    frame="ecliptic",
                ),
                variant_id=[],
            )

        # Store the last simulation in a private variable for reference
        self._last_simulation = sim
        return results, earth_impacts
