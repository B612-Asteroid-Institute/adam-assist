import gc
import hashlib
import os
import pathlib
from ctypes import c_uint32
from typing import Dict, Tuple

import assist
import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv
import rebound
import urllib3
import jax.numpy as jnp
import numpy as np
from jax import jit, lax, vmap
from adam_core.coordinates import CartesianCoordinates, Origin, transform_coordinates, SphericalCoordinates, CoordinateCovariances, OriginCodes
from adam_core.coordinates.covariances import transform_covariances_jacobian, CoordinateCovariances
from adam_core.coordinates.transform import _cartesian_to_spherical, transform_coordinates
from adam_core.coordinates.origin import OriginCodes
from adam_core.dynamics.impacts import EarthImpacts, ImpactMixin
from adam_core.orbits import Orbits
from adam_core.orbits.variants import VariantOrbits
from adam_core.time import Timestamp
from adam_core.utils import get_perturber_state
from quivr.concat import concatenate

from adam_core.propagator.propagator import (
    EphemerisMixin,
    EphemerisType,
    ObserverType,
    OrbitType,
    Propagator,
    TimestampType,
)

from adam_core.observers.observers import Observers
from adam_core.orbits.ephemeris import Ephemeris
from adam_core.dynamics.aberrations import add_stellar_aberration

DATA_DIR = os.getenv("ASSIST_DATA_DIR", "~/.adam_assist_data")

EARTH_RADIUS_KM = 6371.0


def download_jpl_ephemeris_files(data_dir: str = DATA_DIR):
    ephemeris_urls = (
        "https://ssd.jpl.nasa.gov/ftp/eph/small_bodies/asteroids_de441/sb441-n16.bsp",
        "https://ssd.jpl.nasa.gov/ftp/eph/planets/Linux/de440/linux_p1550p2650.440",
    )
    data_dir = pathlib.Path(data_dir).expanduser()
    data_dir.mkdir(parents=True, exist_ok=True)
    for url in ephemeris_urls:
        file_name = pathlib.Path(url).name
        file_path = data_dir.joinpath(file_name)
        if not file_path.exists():
            # use urllib3
            http = urllib3.PoolManager()
            with http.request("GET", url, preload_content=False) as r, open(
                file_path, "wb"
            ) as out_file:
                if r.status != 200:
                    raise RuntimeError(f"Failed to download {url}")
                while True:
                    data = r.read(1024)
                    if not data:
                        break
                    out_file.write(data)
            r.release_conn()


def uint32_hash(s) -> c_uint32:
    sha256_result = hashlib.sha256(s.encode()).digest()
    # Get the first 4 bytes of the SHA256 hash to obtain a uint32 value.
    return c_uint32(int.from_bytes(sha256_result[:4], byteorder="big"))


def hash_orbit_ids_to_uint32(
    orbit_ids: np.ndarray[str],
) -> Tuple[Dict[int, str], np.ndarray[np.uint32]]:
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




class ASSISTPropagator(Propagator, ImpactMixin, EphemerisMixin):

    @staticmethod
    def _propagate_orbits(orbits: OrbitType, times: TimestampType) -> OrbitType:
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

        root_dir = pathlib.Path(DATA_DIR).expanduser()
        ephem = assist.Ephem(
            root_dir.joinpath("linux_p1550p2650.440"),
            root_dir.joinpath("sb441-n16.bsp"),
        )
        sim=None
        gc.collect()
        sim = rebound.Simulation()
        sim.ri_ias15.min_dt = 1e-15
        sim.ri_ias15.adaptive_mode = 2

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

        ax = assist.Extras(sim, ephem)

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

        sim.ri_ias15.min_dt = 1e-15
        sim.ri_ias15.adaptive_mode = 2

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
                )
            elif isinstance(orbits, VariantOrbits):
                # Retrieve the orbit id and weights from hash
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

        results = results.set_column(
            "coordinates",
            transform_coordinates(
                results.coordinates,
                origin_out=OriginCodes.SUN,
                frame_out="ecliptic",
            ),
        )

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

        root_dir = pathlib.Path(DATA_DIR).expanduser()
        ephem_paths = [
            root_dir.joinpath("linux_p1550p2650.440"),
            root_dir.joinpath("sb441-n16.bsp"),
        ]
        ephem = assist.Ephem(*ephem_paths)
        sim = None
        gc.collect()
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
        
        # ASSIST _must_ be initialized before adding particles
        ax = assist.Extras(sim, ephem)

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

        
        # sim.integrator = "ias15"
        sim.ri_ias15.min_dt = 1e-15
        # sim.dt = 1e-9
        # sim.force_is_velocity_dependent = 0
        sim.ri_ias15.adaptive_mode = 2

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
        time_step_results = None

        # Step through each time, move the simulation forward and
        # collect the results.
        while past_integrator_time is False:
            sim.steps(1)
            # print(sim.dt_last_done)
            if sim.t >= final_integrator_time:
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
            normalized_distance = np.linalg.norm(diff[:, :3], axis=1) * 149597870.691

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

        # Add the final positions of the particles to the results
        if results is None:
            results = time_step_results
        else:
            results = qv.concatenate([results, time_step_results])

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
        return results, earth_impacts

    def _add_light_time(
        self,
        orbits: jnp.ndarray,
        t0: jnp.ndarray,
        observer_positions: jnp.ndarray,
        lt_tol: float = 1e-10,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        orbits_aberrated, lts = self._add_light_time_vmap(
            orbits, t0, observer_positions, lt_tol
        )
        return orbits_aberrated, lts

    def _add_light_time_single(
        self,
        orbit: jnp.ndarray,
        t0: float,
        observer_position: jnp.ndarray,
        lt_tol: float = 1e-10,
    ) -> Tuple[jnp.ndarray, jnp.float64]:
        dlt = 1e30
        lt = 1e30

        C = 299792.458  # Speed of light in km/s

        @jit
        def _iterate_light_time(p):
            orbit_i = p[0]
            t0 = p[1]
            lt0 = p[2]
            dlt = p[3]

            # Calculate topocentric distance
            rho = jnp.linalg.norm(orbit_i[:3] - observer_position)

            # Calculate initial guess of light time
            lt = rho / C

            # Calculate difference between previous light time correction
            # and current guess
            dlt = jnp.abs(lt - lt0)

            # Propagate backwards to new epoch
            t1 = t0 - lt
            orbit_propagated = self.propagate_orbit(orbit, t1)

            return [orbit_propagated, t1, lt, dlt]

        @jit
        def _while_condition(p):
            dlt = p[-1]
            return dlt > lt_tol

        p = [orbit, t0, lt, dlt]
        p = lax.while_loop(_while_condition, _iterate_light_time, p)

        orbit_aberrated = p[0]
        t0_aberrated = p[1]
        lt = p[2]
        return orbit_aberrated, lt

    _add_light_time_vmap = jit(
        vmap(
            _add_light_time_single,
            in_axes=(0, 0, 0, None),
            out_axes=(0, 0)
        )
    )

    @jit
    def _compute_ephemeris(
        self,
        propagated_orbit: jnp.ndarray,
        observation_time: float,
        observer_coordinates: jnp.ndarray,
        lt_tol: float = 1e-10,
        stellar_aberration: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.float64]:
        
        propagated_orbits_aberrated, light_time = self._add_light_time(
            propagated_orbit,
            observation_time,
            observer_coordinates[0:3],
            lt_tol=lt_tol,
        )

        topocentric_coordinates = propagated_orbits_aberrated - observer_coordinates

        topocentric_coordinates = lax.cond(
            stellar_aberration,
            lambda topocentric_coords: topocentric_coords.at[0:3].set(
                add_stellar_aberration(
                    propagated_orbits_aberrated.reshape(1, -1),
                    observer_coordinates.reshape(1, -1),
                )[0],
            ),
            lambda topocentric_coords: topocentric_coords,
            topocentric_coordinates,
        )

        ephemeris_spherical = _cartesian_to_spherical(topocentric_coordinates)

        return ephemeris_spherical, light_time

    _compute_ephemeris_vmap = jit(
        vmap(
            lambda self, propagated_orbit, observation_time, observer_coordinates, lt_tol, stellar_aberration: self._compute_ephemeris(
                propagated_orbit, observation_time, observer_coordinates, lt_tol, stellar_aberration
            ),
            in_axes=(None, 0, 0, 0, None, None),
            out_axes=(0, 0),
        )
    )

    def _generate_ephemeris(
        self,
        propagated_orbits: Orbits,
        observers: Observers,
        lt_tol: float = 1e-10,
        stellar_aberration: bool = False,
    ) -> Ephemeris:
        
        propagated_orbits_barycentric = propagated_orbits.set_column(
            "coordinates",
            transform_coordinates(
                propagated_orbits.coordinates,
                CartesianCoordinates,
                frame_out="ecliptic",
                origin_out=OriginCodes.SOLAR_SYSTEM_BARYCENTER,
            ),
        )
        observers_barycentric = observers.set_column(
            "coordinates",
            transform_coordinates(
                observers.coordinates,
                CartesianCoordinates,
                frame_out="ecliptic",
                origin_out=OriginCodes.SOLAR_SYSTEM_BARYCENTER,
            ),
        )

        # Stack the observer coordinates and codes for each orbit in the propagated orbits
        num_orbits = len(propagated_orbits_barycentric.orbit_id.unique())
        observer_coordinates = np.tile(
            observers_barycentric.coordinates.values, (num_orbits, 1)
        )
        observer_codes = np.tile(observers.code.to_numpy(zero_copy_only=False), num_orbits)

        times = propagated_orbits.coordinates.time.to_astropy()
        ephemeris_spherical, light_time = self._compute_ephemeris_vmap(
            propagated_orbits_barycentric.coordinates.values,
            times.mjd,
            observer_coordinates,
            lt_tol,
            stellar_aberration,
        )
        ephemeris_spherical = np.array(ephemeris_spherical)
        light_time = np.array(light_time)

        if not propagated_orbits.coordinates.covariance.is_all_nan():

            cartesian_covariances = propagated_orbits.coordinates.covariance.to_matrix()
            covariances_spherical = transform_covariances_jacobian(
                propagated_orbits.coordinates.values,
                cartesian_covariances,
                self._compute_ephemeris,
                in_axes=(None, 0, 0, 0, None, None),
                out_axes=(0, 0),
                observation_times=times.utc.mjd,
                observer_coordinates=observer_coordinates,
                lt_tol=lt_tol,
                stellar_aberration=stellar_aberration,
            )
            covariances_spherical = CoordinateCovariances.from_matrix(
                np.array(covariances_spherical)
            )

        else:
            covariances_spherical = None

        spherical_coordinates = SphericalCoordinates.from_kwargs(
            time=propagated_orbits.coordinates.time,
            rho=ephemeris_spherical[:, 0],
            lon=ephemeris_spherical[:, 1],
            lat=ephemeris_spherical[:, 2],
            vrho=ephemeris_spherical[:, 3],
            vlon=ephemeris_spherical[:, 4],
            vlat=ephemeris_spherical[:, 5],
            covariance=covariances_spherical,
            origin=Origin.from_kwargs(code=observer_codes),
            frame="ecliptic",
        )

        # Rotate the spherical coordinates from the ecliptic frame
        # to the equatorial frame
        spherical_coordinates = transform_coordinates(
            spherical_coordinates, SphericalCoordinates, frame_out="equatorial"
        )

        return Ephemeris.from_kwargs(
            orbit_id=propagated_orbits_barycentric.orbit_id,
            object_id=propagated_orbits_barycentric.object_id,
            coordinates=spherical_coordinates,
            light_time=light_time,
        )

