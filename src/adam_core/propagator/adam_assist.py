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
from adam_core.coordinates import CartesianCoordinates, Origin, transform_coordinates
from adam_core.coordinates.origin import OriginCodes
from adam_core.dynamics.impacts import EarthImpacts, ImpactMixin
from adam_core.orbits import Orbits
from adam_core.orbits.variants import VariantOrbits
from adam_core.time import Timestamp
from adam_core.utils import get_perturber_state
from quivr.concat import concatenate

from adam_core.propagator.propagator import (
    EphemerisType,
    ObserverType,
    OrbitType,
    Propagator,
    TimestampType,
)

DATA_DIR = os.getenv("ASSIST_DATA_DIR", "~/.adam_assist_data")


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


class ASSISTPropagator(Propagator, ImpactMixin):

    def _propagate_orbits(self, orbits: OrbitType, times: TimestampType) -> OrbitType:
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
        sim = rebound.Simulation()
        sim.ri_ias15.min_dt = 1e-15
        sim.ri_ias15.adaptive_mode = 2

        # Set the simulation time, relative to the jd_ref
        start_tdb_time = orbits.coordinates.time.jd().to_numpy()[0] - ephem.jd_ref
        sim.t = start_tdb_time

        output_type = type(orbits)

        orbit_id_mapping, uint_orbit_ids = hash_orbit_ids_to_uint32(
            orbits.orbit_id.to_numpy(zero_copy_only=False)
        )

        if isinstance(orbits, VariantOrbits):
            variantattributes = {}
            for idx, orbit_id in enumerate(orbits.orbit_id.to_numpy(zero_copy_only=False)):
                variantattributes[orbit_id] = {
                    'weight': orbits.weights[idx],
                    'weight_cov': orbits.weights_cov[idx],
                    'object_id': orbits.object_id[idx]
                }

        # Add the orbits as particles to the simulation
        coords_df = orbits.coordinates.to_dataframe()

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

        ax = assist.Extras(sim, ephem)

        # Prepare the times as jd - jd_ref
        integrator_times = times.rescale("tdb").jd()
        integrator_times = pc.subtract(
            integrator_times, ephem.jd_ref
        )
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
                        time=Timestamp.from_jd(pa.repeat(sim.t + ephem.jd_ref, sim.N), scale="tdb"),
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
                orbit_ids = [orbit_id_mapping[h] for h in orbit_id_hashes]
                object_ids = [variantattributes[orbit_id]["object_id"] for orbit_id in orbit_ids]
                weight = [variantattributes[orbit_id]["weight"] for orbit_id in orbit_ids]
                weights_covs = [variantattributes[orbit_id]["weight_cov"] for orbit_id in orbit_ids]
                time_step_results = VariantOrbits.from_kwargs(
                    orbit_id=orbit_ids,
                    object_id=object_ids,
                    weights=weight,
                    weights_cov=weights_covs,
                    coordinates=CartesianCoordinates.from_kwargs(
                        x=step_xyzvxvyvz[:, 0],
                        y=step_xyzvxvyvz[:, 1],
                        z=step_xyzvxvyvz[:, 2],
                        vx=step_xyzvxvyvz[:, 3],
                        vy=step_xyzvxvyvz[:, 4],
                        vz=step_xyzvxvyvz[:, 5],
                        time=Timestamp.from_jd(pa.repeat(sim.t + ephem.jd_ref, sim.N), scale="tdb"),
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


    def _detect_impacts(self, orbits: OrbitType, num_days: int) -> Tuple[VariantOrbits, EarthImpacts]:
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
            particle_ids = np.char.add(np.char.add(orbit_ids, np.repeat("-", len(orbit_ids))), variant_ids)
            particle_ids = np.array(particle_ids, dtype="object")
        
        orbit_id_mapping, uint_orbit_ids = hash_orbit_ids_to_uint32(
            particle_ids
        )

        # Add the orbits as particles to the simulation
        coords_df = orbits.coordinates.to_dataframe()

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
        
        ax = assist.Extras(sim, ephem)
        sim.ri_ias15.min_dt = 1e-15
        sim.ri_ias15.adaptive_mode = 2


        # Prepare the times as jd - jd_ref
        final_integrator_time = orbits.coordinates.time.add_days(num_days).jd().to_numpy()[0]
        final_integrator_time = final_integrator_time - ephem.jd_ref
        results = None

        # Step through each time, move the simulation forward and
        # collect the results.
        earth_impacts = None
        smallest_distance = None
        smallest_distance_time = None
        past_integrator_time = False
        # for i in [final_integrator_time]:
            # sim.integrate(i)
        while past_integrator_time is False:
            sim.steps(1)
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
                    time=Timestamp.from_jd(pa.repeat(sim.t + ephem.jd_ref, sim.N), scale="tdb"),
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
            orbit_ids, variant_ids = zip(*[particle_id.split("-") for particle_id in particle_ids])

            # Do a quick check to make sure the order of the particles has stayed the same
            # otherwise we would misassign the weights
            np.testing.assert_array_equal(orbits.orbit_id.to_numpy(zero_copy_only=False).astype(str), orbit_ids)
            np.testing.assert_array_equal(orbits.variant_id.to_numpy(zero_copy_only=False).astype(str), variant_ids)

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
                    time=Timestamp.from_jd(pa.repeat(sim.t + ephem.jd_ref, sim.N), scale="tdb"),
                    origin=Origin.from_kwargs(
                        code=pa.repeat(
                            "SOLAR_SYSTEM_BARYCENTER",
                            sim.N,
                        )
                    ),
                    frame="equatorial",
                ),
            )
        
        results = time_step_results.set_column(
            "coordinates",
            transform_coordinates(
                time_step_results.coordinates,
                origin_out=OriginCodes.SUN,
                frame_out="ecliptic",
            ),
        )

            # earth_geo = get_perturber_state(OriginCodes.EARTH, results.coordinates.time[0], origin=OriginCodes.SUN)

            # diff = results.coordinates.values - earth_geo.values
            # normalized_distance = np.min(np.linalg.norm(diff[:, :3], axis=1) * 149597870.691)
            # if smallest_distance is None or normalized_distance < smallest_distance:
            #     smallest_distance = normalized_distance
            #     smallest_distance_time = results.coordinates.time[0].to_astropy().isot

        impacts = sim._extras_ref.get_impacts()
        for i, impact in enumerate(impacts):
                particle_id = orbit_id_mapping.get(impact["hash"], f"unknown-{i}")
                time = Timestamp.from_jd([impact["time"]], scale="tdb")

                coordinates=CartesianCoordinates.from_kwargs(
                        x=[impact["x"]],
                        y=[impact["y"]],
                        z=[impact["z"]],
                        vx=[impact["vx"]],
                        vy=[impact["vy"]],
                        vz=[impact["vz"]],
                        time=time,
                        origin=Origin.from_kwargs(
                            code=["SOLAR_SYSTEM_BARYCENTER"],
                        ),
                        frame="equatorial",
                    )
                coordinates = transform_coordinates(
                    coordinates,
                    origin_out=OriginCodes.SUN,
                    frame_out="ecliptic",
                )

                orbit_id = particle_id
                variant_id = None
                if isinstance(orbits, VariantOrbits):
                    orbit_id, variant_id = particle_id.split("-")
                
                new_impacts = EarthImpacts.from_kwargs(
                    orbit_id=[orbit_id],
                    distance=[impact["distance"]],
                    coordinates=coordinates,
                    variant_id=[variant_id]
                )
                if earth_impacts is None:
                    earth_impacts = new_impacts
                else:
                    earth_impacts = qv.concatenate([earth_impacts, new_impacts])

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
                variant_id=[]
            )
        return results, earth_impacts


    def _generate_ephemeris(
        self, orbits: OrbitType, observers: ObserverType
    ) -> EphemerisType:
        raise NotImplementedError("Ephemeris generation is not implemented for ASSIST.")
