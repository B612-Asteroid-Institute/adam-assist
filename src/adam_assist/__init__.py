import hashlib
import os
import pathlib
from ctypes import c_uint32
from importlib.resources import files
from typing import Dict, List, Optional, Tuple
import ray
import concurrent.futures

import assist
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv
import rebound
import urllib3
from adam_core.orbits.variants import VariantEphemeris, VariantOrbits
from adam_core.orbits import Orbits
from adam_core.coordinates import (CartesianCoordinates, Origin,
                                   transform_coordinates)
from adam_core.coordinates.origin import OriginCodes
from adam_core.propagator.propagator import (EphemerisType, ObserverType,
                                             OrbitType, Propagator,
                                             TimestampType, propagation_worker_ray)
from adam_core.time import Timestamp
from quivr.concat import concatenate
from typing import Literal
import concurrent.futures
import logging
from abc import ABC, abstractmethod
from typing import List, Literal, Optional, Type, Union

import numpy as np
import numpy.typing as npt
import quivr as qv

from adam_core.ray_cluster import initialize_use_ray

DATA_DIR = os.getenv("ASSIST_DATA_DIR", "~/.adam_assist_data")


class EarthImpacts(qv.Table):
    orbit_id = qv.StringColumn()
    # Distance from earth center in km
    distance = qv.Float64Column()
    coordinates = CartesianCoordinates.as_column()


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


def initialize_assist(data_dir: str = DATA_DIR) -> assist.Extras:
    root_dir = pathlib.Path(data_dir).expanduser()
    ephem = assist.Ephem(
        root_dir.joinpath("linux_p1550p2650.440"),
        root_dir.joinpath("sb441-n16.bsp"),
    )
    sim = rebound.Simulation()
    ax = assist.Extras(sim, ephem)
    return sim, ephem


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

@ray.remote
def assist_propagation_worker_ray(
    orbits: OrbitType,
    times: OrbitType,
    adaptive_mode: int,
    min_dt: float,
    propagator: Type["Propagator"],
    **kwargs,
) -> OrbitType:
    prop = propagator(**kwargs)
    propagated = prop._propagate_orbits(orbits, times, adaptive_mode, min_dt)
    return propagated


class ASSISTPropagator(Propagator):
    def _propagate_orbits(
        self, orbits: OrbitType, times: TimestampType, adaptive_mode: int, min_dt: float
    ) -> OrbitType:
            orbits, impacts = self._propagate_orbits_inner(orbits, times, False, adaptive_mode, min_dt)
            return orbits
    
    def propagate_orbits(
        self,
        orbits: OrbitType,
        times: TimestampType,
        covariance: bool = False,
        adaptive_mode: Optional[int] = 0,
        min_dt: Optional[float] = None,
        covariance_method: Literal[
            "auto", "sigma-point", "monte-carlo"
        ] = "monte-carlo",
        num_samples: int = 1000,
        chunk_size: int = 100,
        max_processes: Optional[int] = 1,
        parallel_backend: Literal["cf", "ray"] = "ray",
    ) -> Orbits:
        """
        Propagate each orbit in orbits to each time in times.

        Parameters
        ----------
        orbits : `~adam_core.orbits.orbits.Orbits` (N)
            Orbits to propagate.
        times : Timestamp (M)
            Times to which to propagate orbits.
        covariance : bool, optional
            Propagate the covariance matrices of the orbits. This is done by sampling the
            orbits from their covariance matrices and propagating each sample. The covariance
            of the propagated orbits is then the covariance of the samples.
        covariance_method : {'sigma-point', 'monte-carlo', 'auto'}, optional
            The method to use for sampling the covariance matrix. If 'auto' is selected then the method
            will be automatically selected based on the covariance matrix. The default is 'monte-carlo'.
        num_samples : int, optional
            The number of samples to draw when sampling with monte-carlo.
        chunk_size : int, optional
            Number of orbits to send to each job.
        max_processes : int or None, optional
            Maximum number of processes to launch. If None then the number of
            processes will be equal to the number of cores on the machine. If 1
            then no multiprocessing will be used. If "ray" is the parallel_backend and a ray instance
            is initialized already then this argument is ignored.
        parallel_backend : {'cf', 'ray'}, optional
            The parallel backend to use. 'cf' uses concurrent.futures and 'ray' uses ray. The default is 'cf'.
            To use ray, ray must be installed.

        Returns
        -------
        propagated : `~adam_core.orbits.orbits.Orbits`
            Propagated orbits.
        """

        if max_processes is None or max_processes > 1:
            propagated_list: List[Orbits] = []
            variants_list: List[VariantOrbits] = []

            if parallel_backend == "cf":
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=max_processes
                ) as executor:
                    # Add orbits to propagate to futures
                    futures = []
                    for orbit_chunk in _iterate_chunks(orbits, chunk_size):
                        futures.append(
                            executor.submit(
                                propagation_worker,
                                orbit_chunk,
                                times,
                                self.__class__,
                                **self.__dict__,
                            )
                        )

                    # Add variants to propagate to futures
                    if (
                        covariance is True
                        and not orbits.coordinates.covariance.is_all_nan()
                    ):
                        variants = VariantOrbits.create(
                            orbits, method=covariance_method, num_samples=num_samples
                        )
                        for variant_chunk in _iterate_chunks(variants, chunk_size):
                            futures.append(
                                executor.submit(
                                    propagation_worker,
                                    variant_chunk,
                                    times,
                                    self.__class__,
                                    **self.__dict__,
                                )
                            )

                    for future in concurrent.futures.as_completed(futures):
                        result = future.result()
                        if isinstance(result, Orbits):
                            propagated_list.append(result)
                        elif isinstance(result, VariantOrbits):
                            variants_list.append(result)
                        else:
                            raise ValueError(
                                f"Unexpected result type from propagation worker: {type(result)}"
                            )

            elif parallel_backend == "ray":
                if RAY_INSTALLED is False:
                    raise ImportError(
                        "Ray must be installed to use the ray parallel backend"
                    )

                initialize_use_ray(num_cpus=max_processes)

                # Add orbits and times to object store if
                # they haven't already been added
                if not isinstance(times, ObjectRef):
                    times_ref = ray.put(times)
                else:
                    times_ref = times

                if not isinstance(orbits, ObjectRef):
                    orbits_ref = ray.put(orbits)
                else:
                    orbits_ref = orbits
                    # We need to dereference the orbits ObjectRef so we can
                    # check its length for chunking and determine
                    # if we need to propagate variants
                    orbits = ray.get(orbits_ref)

                # Create futures
                futures = []
                idx = np.arange(0, len(orbits))

                for orbit in orbits:
                    futures.append(
                        assist_propagation_worker_ray.remote(
                            orbit,
                            times_ref,
                            adaptive_mode,
                            min_dt,
                            self.__class__,
                            **self.__dict__,
                        )
                    )
                    if (
                        covariance is True
                        and not orbit.coordinates.covariance.is_all_nan()
                    ):
                        variants = VariantOrbits.create(
                            orbit,
                            method=covariance_method,
                            num_samples=num_samples,
                        )
                        futures.append(
                            assist_propagation_worker_ray.remote(
                                variants,
                                times_ref,
                                adaptive_mode,
                                min_dt,
                                self.__class__,
                                **self.__dict__,
                            )
                        )

                # Get results as they finish (we sort later)
                unfinished = futures
                while unfinished:
                    finished, unfinished = ray.wait(unfinished, num_returns=1)
                    result = ray.get(finished[0])
                    if isinstance(result, Orbits):
                        propagated_list.append(result)
                    elif isinstance(result, VariantOrbits):
                        variants_list.append(result)
                    else:
                        raise ValueError(
                            f"Unexpected result type from propagation worker: {type(result)}"
                        )

            else:
                raise ValueError(f"Unknown parallel backend: {parallel_backend}")

            # Concatenate propagated orbits
            propagated = qv.concatenate(propagated_list)
            if len(variants_list) > 0:
                propagated_variants = qv.concatenate(variants_list)
            else:
                propagated_variants = None

        else:
            propagated = self._propagate_orbits(orbits, times, adaptive_mode, min_dt)

            if covariance is True and not orbits.coordinates.covariance.is_all_nan():
                variants = VariantOrbits.create(
                    orbits, method=covariance_method, num_samples=num_samples
                )
                propagated_variants = self._propagate_orbits(variants, times, adaptive_mode, min_dt)
            else:
                propagated_variants = None

        if propagated_variants is not None:
            propagated = propagated_variants.collapse(propagated)

        return propagated.sort_by(
            ["orbit_id", "coordinates.time.days", "coordinates.time.nanos"]
        )

    def _propagate_orbits_inner(self, orbits: OrbitType, times: TimestampType, detect_impacts: bool, adaptive_mode: int, min_dt: float) -> Tuple[OrbitType, EarthImpacts]:
        # Assert that the time for each orbit definition is the same for the simulator to work
        assert len(pc.unique(orbits.coordinates.time.mjd())) == 1

        # sim, ephem = initialize_assist()

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

        if min_dt is not None:
            sim.ri_ias15.min_dt = min_dt

        if adaptive_mode is not None:
            sim.ri_ias15.adaptive_mode = adaptive_mode

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

        if detect_impacts:
            impacts = sim._extras_ref.get_impacts()
            earth_impacts = None
            for i, impact in enumerate(impacts):
                orbit_id = orbit_id_mapping.get(impact["hash"], f"unknown-{i}")
                time = Timestamp.from_jd([impact["time"]], scale="tdb")
                if earth_impacts is None:
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
                    earth_impacts = EarthImpacts.from_kwargs(
                        orbit_id=[orbit_id],
                        distance=[impact["distance"]],
                        coordinates=coordinates,
                    )
                else:
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
                    earth_impacts = qv.concatenate(earth_impacts, EarthImpacts.from_kwargs(
                        orbit_id=[orbit_id],
                        distance=[impact["distance"]],
                        coordinates=coordinates,
                    ))

            return results, earth_impacts
        else:
            return results, None

    def _generate_ephemeris(
        self, orbits: OrbitType, observers: ObserverType
    ) -> EphemerisType:
        raise NotImplementedError("Ephemeris generation is not implemented for ASSIST.")
