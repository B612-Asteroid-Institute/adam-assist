import glob
import os
import pathlib
from typing import Literal, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pcsv
import pyarrow.parquet as pq
import pytest
import quivr as qv
import seaborn as sns
from adam_core.constants import KM_P_AU
from adam_core.coordinates import SphericalCoordinates
from adam_core.coordinates.cometary import CometaryCoordinates
from adam_core.coordinates.origin import Origin
from adam_core.coordinates.residuals import Residuals
from adam_core.observers import Observers
from adam_core.observers.observers import OBSERVATORY_CODES
from adam_core.orbits import Ephemeris, Orbits
from adam_core.orbits.query.horizons import query_horizons
from adam_core.time import Timestamp
from adam_core.utils.spice import setup_SPICE, sp
from assist import Ephem
from astropy.time import Time
from astroquery.mpc import MPC
from google.cloud import bigquery
from numpy.typing import NDArray
from scipy.stats import ttest_rel

from src.adam_core.propagator.adam_assist import (
    DATA_DIR,
    ASSISTPropagator,
    download_jpl_ephemeris_files,
)

OBJECTS = {
    # This NEA has a close approach around Sept. 26, 2029 (62405)
    # Cover times a year before and 5 years after
    "2005 YY128": Timestamp.from_mjd(
        pc.add(62405, pa.array(range(-365, 5 * 365, 90))), scale="utc"
    ),
    # Do a 10 year propagation of Holman starting 60600 and stepping every 100 days
    # This is roughly twice its period
    "3666": Timestamp.from_mjd(
        pc.add(60600, pa.array(range(0, 3650, 100))), scale="utc"
    ),
    "136199": Timestamp.from_mjd(
        pc.add(60000, pa.array(range(0, 50 * 365, 365))), scale="utc"
    ),
    # Item on the risk list with near term close approach
    "2022 YO1": Timestamp.from_mjd(
        pc.add(60366.26, pa.array(range(-60, 60, 10))), scale="utc"
    ),
    # Apophis around the 2029 close approach
    "99942": Timestamp.from_mjd(
        pc.add(62249, pa.array(range(-100, 100, 10))), scale="utc"
    ),
    # Look at Hektor, jupiter trojan
    "624": Timestamp.from_mjd(
        pc.add(60000, pa.array(range(0, 50 * 365, 365))), scale="utc"
    ),
    # The following set taken from adam_core.utils.helpers.orbits
    # as examples of different classification
    # 594913 'Aylo'chaxnim (2020 AV2)
    "2020 AV2": Timestamp.from_mjd(
        pc.add(60000, pa.array(range(-365, 5 * 365, 90))), scale="utc"
    ),
    # 163693 Atira (2003 CP20)
    "2003 CP20": Timestamp.from_mjd(
        pc.add(60000, pa.array(range(0, 3 * 365, 50))), scale="utc"
    ),
    # (2010 TK7)
    "2010 TK7": Timestamp.from_mjd(
        pc.add(60000, pa.array(range(0, 3 * 365, 50))), scale="utc"
    ),
    # 3753 Cruithne (1986 TO)
    "1986 TO": Timestamp.from_mjd(
        pc.add(60000, pa.array(range(0, 3 * 365, 50))), scale="utc"
    ),
    # 54509 YORP (2000 PH5)
    "2000 PH5": Timestamp.from_mjd(
        pc.add(60000, pa.array(range(0, 3 * 365, 50))), scale="utc"
    ),
    # 2063 Bacchus (1977 HB)
    "1977 HB": Timestamp.from_mjd(
        pc.add(60000, pa.array(range(0, 3 * 365, 50))), scale="utc"
    ),
    # 1221 Amor (1932 EA1)
    "1932 EA1": Timestamp.from_mjd(
        pc.add(60000, pa.array(range(0, 3 * 365, 50))), scale="utc"
    ),
    # 433 Eros (A898 PA)
    "A898 PA": Timestamp.from_mjd(
        pc.add(60000, pa.array(range(0, 3 * 365, 50))), scale="utc"
    ),
    # 3908 Nyx (1980 PA)
    "1980 PA": Timestamp.from_mjd(
        pc.add(60000, pa.array(range(0, 3 * 365, 50))), scale="utc"
    ),
    # 434 Hungaria (A898 RB)
    "A898 RB": Timestamp.from_mjd(
        pc.add(60000, pa.array(range(0, 3 * 365, 50))), scale="utc"
    ),
    # 1876 Napolitania (1970 BA)
    "1970 BA": Timestamp.from_mjd(
        pc.add(60000, pa.array(range(0, 3 * 365, 50))), scale="utc"
    ),
    # 2001 Einstein (1973 EB)
    "1973 EB": Timestamp.from_mjd(
        pc.add(60000, pa.array(range(0, 3 * 365, 50))), scale="utc"
    ),
    # 6 He
    "A847 NA": Timestamp.from_mjd(
        pc.add(60000, pa.array(range(0, 3 * 365, 50))), scale="utc"
    ),
    # 6522 Aci (1991 NQ)
    "1991 NQ": Timestamp.from_mjd(
        pc.add(60000, pa.array(range(0, 3 * 365, 50))), scale="utc"
    ),
    # 10297 Lynnejones (1988 RJ13)
    "1988 RJ13": Timestamp.from_mjd(
        pc.add(60000, pa.array(range(0, 3 * 365, 50))), scale="utc"
    ),
    # 17032 Edlu (1999 FM9)
    "1999 FM9": Timestamp.from_mjd(
        pc.add(60000, pa.array(range(0, 3 * 365, 50))), scale="utc"
    ),
    # 202930 Ivezic (1998 SG172)
    "1998 SG172": Timestamp.from_mjd(
        pc.add(60000, pa.array(range(0, 3 * 365, 50))), scale="utc"
    ),
    # 911 Agamemnon (A919 FB)
    "A919 FB": Timestamp.from_mjd(
        pc.add(60000, pa.array(range(0, 3 * 365, 50))), scale="utc"
    ),
    # 1143 Odysseus (1930 BH)
    "1930 BH": Timestamp.from_mjd(
        pc.add(60000, pa.array(range(0, 3 * 365, 50))), scale="utc"
    ),
    # 1172 Aneas (1930 UA)
    "1930 UA": Timestamp.from_mjd(
        pc.add(60000, pa.array(range(0, 3 * 365, 50))), scale="utc"
    ),
    # 3317 Paris (1984 KF)
    "1984 KF": Timestamp.from_mjd(
        pc.add(60000, pa.array(range(0, 3 * 365, 50))), scale="utc"
    ),
    # 5145 Pholus (1992 AD)
    "1992 AD": Timestamp.from_mjd(
        pc.add(60000, pa.array(range(0, 3 * 365, 50))), scale="utc"
    ),
    # 5335 Damocles (1991 DA)
    "1991 DA": Timestamp.from_mjd(
        pc.add(60000, pa.array(range(0, 3 * 365, 50))), scale="utc"
    ),
    # 15760 Albion (1992 QB1)
    "1992 QB1": Timestamp.from_mjd(
        pc.add(60000, pa.array(range(0, 3 * 365, 50))), scale="utc"
    ),
    # 15788 (1993 SB)
    "1993 SB": Timestamp.from_mjd(
        pc.add(60000, pa.array(range(0, 3 * 365, 50))), scale="utc"
    ),
    # 15789 (1993 SC)
    "1993 SC": Timestamp.from_mjd(
        pc.add(60000, pa.array(range(0, 3 * 365, 50))), scale="utc"
    ),
    # 1I/'Oumuamua (A/2017 U1)
    "A/2017 U1": Timestamp.from_mjd(
        pc.add(60000, pa.array(range(0, 3 * 365, 50))), scale="utc"
    ),
}


## Begin Joachim's code, to be pulled into mpcq
class MPCObservations(qv.Table):
    obsid = qv.LargeStringColumn()
    primary_designation = qv.LargeStringColumn()
    trksub = qv.LargeStringColumn(nullable=True)
    provid = qv.LargeStringColumn(nullable=True)
    permid = qv.LargeStringColumn(nullable=True)
    submission_id = qv.LargeStringColumn()
    obssubid = qv.LargeStringColumn(nullable=True)
    obstime = Timestamp.as_column()
    ra = qv.Float64Column()
    dec = qv.Float64Column()
    rmsra = qv.Float64Column(nullable=True)
    rmsdec = qv.Float64Column(nullable=True)
    mag = qv.Float64Column(nullable=True)
    rmsmag = qv.Float64Column(nullable=True)
    band = qv.LargeStringColumn(nullable=True)
    stn = qv.LargeStringColumn()
    updated_at = Timestamp.as_column(nullable=True)
    created_at = Timestamp.as_column(nullable=True)
    status = qv.LargeStringColumn()


def query_mpc_observations(provids):

    query = f"""
    SELECT DISTINCT obsid, primary_designation, trksub, permid, provid, submission_id, obssubid, obstime, ra, dec, rmsra, rmsdec, mag, rmsmag, band, stn, updated_at, created_at, status
    FROM `moeyens-thor-dev.mpc_sbn_aipublic.obs_sbn` AS obs_sbn
    INNER JOIN (
        SELECT unpacked_primary_provisional_designation AS primary_designation, unpacked_secondary_provisional_designation AS secondary_designation
        FROM moeyens-thor-dev.mpc_sbn_aipublic.current_identifications
        WHERE unpacked_primary_provisional_designation IN ({", ".join([f'"{id}"' for id in provids])})
        OR unpacked_secondary_provisional_designation IN ({", ".join([f'"{id}"' for id in provids])})
    ) AS identifications
    ON obs_sbn.provid = identifications.primary_designation OR obs_sbn.provid = identifications.secondary_designation
    """

    client = bigquery.Client()

    query_job = client.query(query)

    # Wait for the query to finish
    results = query_job.result()

    # Convert the results to a PyArrow table
    table = results.to_arrow()

    obstime = Time(
        table["obstime"].to_numpy(zero_copy_only=False),
        format="datetime64",
        scale="utc",
    )
    created_at = Time(
        table["created_at"].to_numpy(zero_copy_only=False),
        format="datetime64",
        scale="utc",
    )
    updated_at = Time(
        table["updated_at"].to_numpy(zero_copy_only=False),
        format="datetime64",
        scale="utc",
    )

    mpcobs = MPCObservations.from_kwargs(
        obsid=table["obsid"],
        primary_designation=table["primary_designation"],
        trksub=table["trksub"],
        provid=table["provid"],
        permid=table["permid"],
        submission_id=table["submission_id"],
        obssubid=table["obssubid"],
        obstime=Timestamp.from_astropy(obstime),
        ra=table["ra"],
        dec=table["dec"],
        rmsra=table["rmsra"],
        rmsdec=table["rmsdec"],
        mag=table["mag"],
        rmsmag=table["rmsmag"],
        band=table["band"],
        stn=table["stn"],
        updated_at=Timestamp.from_astropy(updated_at),
        created_at=Timestamp.from_astropy(created_at),
        status=table["status"],
    )
    return mpcobs.sort_by(
        [
            ("primary_designation", "ascending"),
            ("obstime.days", "ascending"),
            ("obstime.nanos", "ascending"),
        ]
    )


class MPCOrbits(qv.Table):

    id = qv.Int64Column()
    provid = qv.LargeStringColumn()
    created_at = Timestamp.as_column()
    updated_at = Timestamp.as_column()
    a = qv.Float64Column(nullable=True)
    q = qv.Float64Column(nullable=True)
    e = qv.Float64Column(nullable=True)
    i = qv.Float64Column(nullable=True)
    node = qv.Float64Column(nullable=True)
    argperi = qv.Float64Column(nullable=True)
    peri_time = qv.Float64Column(nullable=True)
    mean_anomaly = qv.Float64Column(nullable=True)
    epoch = Timestamp.as_column()

    def to_orbits(self):

        orbits = Orbits.from_kwargs(
            orbit_id=self.id,
            object_id=self.provid,
            coordinates=CometaryCoordinates.from_kwargs(
                q=self.q,
                e=self.e,
                i=self.i,
                raan=self.node,
                ap=self.argperi,
                tp=self.peri_time,
                time=self.epoch,
                origin=Origin.from_kwargs(
                    code=pa.repeat("SUN", len(self)),
                ),
                frame="ecliptic",
            ).to_cartesian(),
        )
        return orbits


def query_mpc_orbits(provids):

    query = f"""
    SELECT DISTINCT id, unpacked_primary_provisional_designation, created_at, updated_at, a, q, e, i, node, argperi, peri_time, mean_anomaly, epoch_mjd
    FROM `moeyens-thor-dev.mpc_sbn_aipublic.mpc_orbits` AS mpc_orbits
    INNER JOIN (
        SELECT unpacked_primary_provisional_designation AS primary_designation, unpacked_secondary_provisional_designation AS secondary_designation
        FROM moeyens-thor-dev.mpc_sbn_aipublic.current_identifications
        WHERE unpacked_primary_provisional_designation IN ({", ".join([f'"{id}"' for id in provids])})
        OR unpacked_secondary_provisional_designation IN ({", ".join([f'"{id}"' for id in provids])})
    ) AS identifications
    ON mpc_orbits.unpacked_primary_provisional_designation = identifications.primary_designation OR mpc_orbits.unpacked_primary_provisional_designation = identifications.secondary_designation
    """

    client = bigquery.Client()

    query_job = client.query(query)

    # Wait for the query to finish
    results = query_job.result()

    # Convert the results to a PyArrow table
    table = results.to_arrow()

    created_at = Time(
        table["created_at"].to_numpy(zero_copy_only=False),
        format="datetime64",
        scale="utc",
    )
    updated_at = Time(
        table["updated_at"].to_numpy(zero_copy_only=False),
        format="datetime64",
        scale="utc",
    )

    mpcorbits = MPCOrbits.from_kwargs(
        id=table["id"],
        provid=table["unpacked_primary_provisional_designation"],
        created_at=Timestamp.from_astropy(created_at),
        updated_at=Timestamp.from_astropy(updated_at),
        a=table["a"],
        q=table["q"],
        e=table["e"],
        i=table["i"],
        node=table["node"],
        argperi=table["argperi"],
        peri_time=table["peri_time"],
        mean_anomaly=table["mean_anomaly"],
        epoch=Timestamp.from_mjd(table["epoch_mjd"], scale="utc"),
    )

    return mpcorbits.sort_by(
        [
            ("provid", "ascending"),
            ("epoch.days", "ascending"),
            ("epoch.nanos", "ascending"),
        ]
    )


## End Joachim's code


def chart_residuals(
    path: str,
    env_name: str,
    object_name: str,
    horizons_vectors: Orbits,
    assist_vectors: Orbits,
    residuals: NDArray,
) -> str:

    times = horizons_vectors.coordinates.time.mjd().to_pylist()
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.plot(times, residuals, label="delta Position")
    # In the x label I want to the offset from the time[0] in parenthesis
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.1f}\n({x - times[0]:.1f})")
    )
    ax.set_title(f"{env_name} {object_name} Residuals")
    ax.set_xlabel("MJD (delta t_0)")
    ax.set_ylabel("Position Residuals (KM)")
    ax.legend()

    # Save the plot to a file
    fig.savefig(path)


def fetch_horizons_vectors(output_folder: str = "./horizons") -> None:
    """
    Query Horizons for the object_id at the given times and save the results to a parquet file
    """
    pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)
    horizons_path = f"{output_folder}/horizons.parquet"
    if not pathlib.Path(horizons_path).exists():
        horizons_vectors = None
        for object_id, times in OBJECTS.items():
            horizons_vectors_ = query_horizons([object_id], times)
            # Replace the object id with the original object_id
            horizons_vectors_ = horizons_vectors_.set_column(
                "object_id",
                pa.array(
                    pa.repeat(object_id, len(horizons_vectors_)), type=pa.large_string()
                ),
            )
            if horizons_vectors is None:
                horizons_vectors = horizons_vectors_
            else:
                horizons_vectors = qv.concatenate([horizons_vectors, horizons_vectors_])
        horizons_vectors.to_parquet(horizons_path)
    horizons_vectors = Orbits.from_parquet(horizons_path)
    return horizons_vectors


# Grab arbitrary run name
def test_horizons_residuals():
    """
    Test how closely assist matches SBDB and Horizons over time
    """
    download_jpl_ephemeris_files()
    env_name = os.environ.get("ASSIST_VERSION", "local")

    pathlib.Path("./outputs").mkdir(parents=True, exist_ok=True)

    horizons_vectors = fetch_horizons_vectors()

    fieldnames = [field[0] for field in Ephem._fields_]
    is_spk = "spk_global" in fieldnames

    ephem_files = (
        ("440", "linux_p1550p2650.440", "sb441-n16.bsp"),
        ("441", "linux_m13000p17000.441", "sb441-n16.bsp"),
    )

    if is_spk:
        ephem_files = (
            ("440", "de440.bsp", "sb441-n16.bsp"),
            ("441", "de441.bsp", "sb441-n16.bsp"),
        )

    for ephem_version, planets_file, asteroids_file in ephem_files:
        job_name = f"{env_name}_{ephem_version}"
        for object_id, times in OBJECTS.items():

            safe_object_name = object_id.replace("/", "_")
            residuals_path = (
                f"./outputs/{safe_object_name}_horizons_{job_name}.parquet"
            )

            print(f"Running: {object_id} {job_name}")
            if pathlib.Path(residuals_path).exists():
                print(f"File found, skipping {object_id} {job_name}")
                continue


            # Get the horizons vectors for this object_id
            object_horizons_vectors = horizons_vectors.select("object_id", object_id)

            props = ASSISTPropagator(
                planets_path=pathlib.Path(DATA_DIR).expanduser().joinpath(planets_file),
                asteroids_path=pathlib.Path(DATA_DIR)
                .expanduser()
                .joinpath(asteroids_file),
            )
            # We use the first vector as the initial state
            assist_vectors = props.propagate_orbits(
                object_horizons_vectors[0], object_horizons_vectors.coordinates.time
            )

            # Calculate the positional distance between the two vectors
            residuals = np.linalg.norm(
                assist_vectors.coordinates.r - object_horizons_vectors.coordinates.r,
                axis=1,
            )

            # Convert from AU to meters
            residuals *= KM_P_AU

            # Convert the rediduals to pyarrow and save as parquet
            residuals_pa = pa.array(residuals)
            residuals_table = pa.table(
                {
                    "residuals": residuals_pa,
                    "object_id": pa.repeat(object_id, len(residuals)),
                    "time": object_horizons_vectors.coordinates.time.mjd(),
                    "env": pa.repeat(job_name, len(residuals)),
                }
            )

            pq.write_table(residuals_table, residuals_path)


def fetch_mpc_data(output_folder: str = "./mpc") -> Tuple[Orbits, MPCObservations]:
    """
    Query the MPC for the ephemeris of the objects in OBJECTS
    """
    if not pathlib.Path(f"./{output_folder}/mpc_observations.parquet").exists():
        pathlib.Path(f"./{output_folder}").mkdir(parents=True, exist_ok=True)
        all_obs = None
        orbits = Orbits.empty()
        all_obs = query_mpc_observations([object_id for object_id in OBJECTS.keys()])
        mpc_orbits = query_mpc_orbits([object_id for object_id in OBJECTS.keys()])
        orbits = mpc_orbits.to_orbits()
        all_obs.to_parquet(f"./{output_folder}/mpc_observations.parquet")
        orbits.to_parquet(f"./{output_folder}/mpc_orbits.parquet")
    all_obs = MPCObservations.from_parquet(f"./{output_folder}/mpc_observations.parquet")
    orbits = Orbits.from_parquet(f"./{output_folder}/mpc_orbits.parquet")
    return orbits, all_obs


def _observers_from_mpc_observations(mpc_observations: MPCObservations) -> Observers:
    """
    Convert the MPC observations to Observers
    """
    observers = Observers.from_codes(
        codes=mpc_observations.stn, times=mpc_observations.obstime
    )
    return observers


def _collect_supported_observatories(
    mpc_observations: MPCObservations,
) -> pa.LargeStringArray:
    """
    Get the unique codes from mpc_observations and check via observers if we have coverage in the spice files
    """
    unique_codes = set(mpc_observations.stn.unique().to_pylist())
    # Get the intersection of the unique codes and the supported codes
    allowed_codes = unique_codes.intersection(OBSERVATORY_CODES)
    # These codes are not supported by the spice files
    allowed_codes.remove("C53")
    allowed_codes.remove("C51")
    return pa.array(list(allowed_codes), type=pa.large_string())


def _filter_supported_mpc_observations(
    mpc_observations: MPCObservations,
) -> MPCObservations:
    """
    Filter the MPC observations to only include those with supported observatories
    """
    supported_codes = _collect_supported_observatories(mpc_observations)
    mask = pc.is_in(mpc_observations.stn, supported_codes)
    mpc_observations = mpc_observations.apply_mask(mask)

    # Remove observations before 1990
    mask = pc.greater(mpc_observations.obstime.mjd(), 47892)
    mpc_observations = mpc_observations.apply_mask(mask)
    return mpc_observations


def _select_object_observations(
    mpc_observations: MPCObservations, object_id: str
) -> MPCObservations:
    """
    Select the observations for a given object_id
    """
    mask = pc.or_(
        pc.or_(
            pc.fill_null(
                pc.equal(mpc_observations.primary_designation, object_id), False
            ),
            pc.fill_null(pc.equal(mpc_observations.provid, object_id), False),
        ),
        pc.fill_null(pc.equal(mpc_observations.permid, object_id), False),
    )
    return mpc_observations.apply_mask(mask)


def test_mpc_residuals():
    """
    Test how closely assist matches the MPC ephemeris
    """
    # Find the total number of loaded kernels
    download_jpl_ephemeris_files()
    env_name = os.environ.get("ASSIST_VERSION", "local")

    pathlib.Path("./outputs").mkdir(parents=True, exist_ok=True)

    orbits, mpc_observations = fetch_mpc_data()

    mpc_observations = _filter_supported_mpc_observations(mpc_observations)

    fieldnames = [field[0] for field in Ephem._fields_]
    is_spk = "spk_global" in fieldnames

    ephem_files = (
        ("440", "linux_p1550p2650.440", "sb441-n16.bsp"),
        ("441", "linux_m13000p17000.441", "sb441-n16.bsp"),
    )

    if is_spk:
        ephem_files = (
            ("440", "de440.bsp", "sb441-n16.bsp"),
            ("441", "de441.bsp", "sb441-n16.bsp"),
        )

    for ephem_version, planets_file, asteroids_file in ephem_files:
        job_name = f"{env_name}_{ephem_version}"

        for object_id in OBJECTS.keys():
            safe_object_name = object_id.replace("/", "_")
            on_sky_difference_path = f"./outputs/{safe_object_name}_mpc_{job_name}.parquet"

            print(f"Running: {object_id} {job_name}")
            if pathlib.Path(on_sky_difference_path).exists():
                print(f"File found, skipping {object_id} {job_name}")
                continue


            props = ASSISTPropagator(
                planets_path=pathlib.Path(DATA_DIR).expanduser().joinpath(planets_file),
                asteroids_path=pathlib.Path(DATA_DIR)
                .expanduser()
                .joinpath(asteroids_file),
            )
            orbit = orbits.select("object_id", object_id)
            # Select observations from primary_designation, provid or permid
            object_mpc_observations = _select_object_observations(
                mpc_observations, object_id
            )
            if len(object_mpc_observations) == 0:
                print(f"No observations found for {object_id}")
                continue
            object_mpc_observations = object_mpc_observations.sort_by(
                [("obstime.days", "ascending"), ("obstime.nanos", "ascending")]
            )

            observers = _observers_from_mpc_observations(object_mpc_observations)

            props = ASSISTPropagator()
            assist_ephem = props.generate_ephemeris(orbit, observers, covariance=True)

            # Compare ra/dec
            # Get the difference in magnitude for the lon/lats
            on_sky_difference = np.linalg.norm(
                assist_ephem.coordinates.values[:, 1:3]
                - object_mpc_observations.to_dataframe()[["ra", "dec"]].to_numpy(),
                axis=1,
            )

            # Convert from decimal degrees to milliarcseconds
            on_sky_difference *= 3600000

            # Convert the rediduals to pyarrow and save as parquet
            on_sky_difference_pa = pa.array(on_sky_difference)
            on_sky_difference_table = pa.table(
                {
                    "residuals": on_sky_difference_pa,
                    "object_id": pa.repeat(object_id, len(on_sky_difference)),
                    "time": object_mpc_observations.obstime.mjd(),
                    "env": pa.repeat(job_name, len(on_sky_difference)),
                }
            )

            pq.write_table(on_sky_difference_table, on_sky_difference_path)


def compare_residuals(residual_type: Literal["horizons", "mpc"]):
    """
    Load in the parquet files in outputs.
    Calculate the mean residuals per time step for each unique environment and object.
    """
    residuals_files = glob.glob(f"./outputs/*_{residual_type}_*.parquet")
    residuals_tables = [pq.read_table(file) for file in residuals_files]

    # Combine all the tables
    residuals_table = pa.concat_tables(residuals_tables)

    # Compute mean residual per time step for each object and environment
    grouped = residuals_table.group_by(["object_id", "env"]).aggregate(
        [
            ("residuals", "mean"),
            ("residuals", "stddev"),
            ("residuals", "count"),
            ("time", "min"),
            ("time", "max"),
        ]
    )

    # Represent time min and max as datetime strings instead of MJD
    time_min_str = [
        time.strftime("%Y-%m-%d")
        for time in list(
            Timestamp.from_mjd(grouped["time_min"]).to_astropy().to_datetime()
        )
    ]
    time_max_str = [
        time.strftime("%Y-%m-%d")
        for time in list(
            Timestamp.from_mjd(grouped["time_max"]).to_astropy().to_datetime()
        )
    ]
    grouped = grouped.add_column(0, "time_min_str", pa.array(time_min_str))
    grouped = grouped.add_column(0, "time_max_str", pa.array(time_max_str))

    to_print = grouped.to_pandas()[
        [
            "object_id",
            "env",
            "residuals_mean",
            "residuals_stddev",
            "residuals_count",
            "time_min_str",
            "time_max_str",
        ]
    ]
    to_print = to_print.sort_values(
        by=["time_min_str", "residuals_mean"], ascending=[True, True]
    )
    print(to_print.to_string(index=False))

    # Convert to pandas DataFrame for easier manipulation
    df = grouped.to_pandas()

    # Normalize residuals by dividing by the maximum residual for each object
    df["max_residual"] = df.groupby("object_id")["residuals_mean"].transform("max")
    df["normalized_residual"] = df["residuals_mean"] / df["max_residual"]

    # Aggregate across objects
    summary = (
        df.groupby("env")
        .agg({"normalized_residual": ["mean", "median"], "residuals_count": "sum"})
        .reset_index()
    )

    # Replace normalized residual with 1-minus
    summary["normalized_residual"] = 1 - summary["normalized_residual"]

    # A larger 1-minus normalized residuals indicates better performance
    # against the worst performer
    summary = summary.sort_values(("normalized_residual", "mean"), ascending=False)

    # print without the index
    print(summary.to_string(index=False))

    statistical_test(df)


def statistical_test(df):
    """
    Perform paired t-tests between each pair of environments and print the results in a table.

    Parameters:
    df (DataFrame): The DataFrame containing 'env' and 'normalized_residual' columns.
    """
    envs = df["env"].unique()
    results = []

    for i in range(len(envs)):
        for j in range(i + 1, len(envs)):
            env1 = envs[i]
            env2 = envs[j]
            # Ensure the data is aligned by 'object_id' to maintain pairing
            data1 = df[df["env"] == env1].set_index("object_id")["normalized_residual"]
            data2 = df[df["env"] == env2].set_index("object_id")["normalized_residual"]
            # Only keep common object_ids to ensure proper pairing
            common_objects = data1.index.intersection(data2.index)
            paired_data1 = data1.loc[common_objects]
            paired_data2 = data2.loc[common_objects]
            # Perform paired t-test
            stat, p = ttest_rel(paired_data1, paired_data2)
            results.append({"Environment 1": env1, "Environment 2": env2, "p-value": p})

    # Create a DataFrame for the results
    results_df = pd.DataFrame(results)

    # Format p-values for readability
    results_df["p-value"] = results_df["p-value"].apply(lambda x: f"{x:.3e}")

    # Print the results in a nicely formatted table
    print("\nPaired t-test Results:")
    print(results_df.to_string(index=False))


def plot_residuals(df):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="env", y="normalized_residual", data=df)
    plt.title("Normalized Residuals per Environment")
    plt.ylabel("Normalized Residual")
    plt.xlabel("Environment")
    # Make sure the labels are angled to make them readable
    plt.xticks(rotation=45, ha="right")
    plt.show()
