import glob
import os
import pathlib

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
from adam_core.observations
from adam_core.constants import KM_P_AU
from adam_core.observers import Observers
from adam_core.orbits import Orbits, Ephemeris
from adam_core.orbits.query.horizons import query_horizons
from adam_core.orbits.query.sbdb import query_sbdb
from adam_core.time import Timestamp
from adam_core.coordinates import SphericalCoordinates
from assist import Ephem
from astropy.mpc import MPC
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

            print(f"{object_id} {job_name}")

            # Get the horizons vectors for this object_id
            object_horizons_vectors = horizons_vectors.select("object_id", object_id)
            print(object_horizons_vectors)

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
            safe_object_name = object_id.replace("/", "_")
            residuals_path = (
                f"./outputs/{safe_object_name}_residuals_{job_name}.parquet"
            )
            pq.write_table(residuals_table, residuals_path)


def fetch_mpc_observations():
    """
    Query the MPC for the ephemeris of the objects in OBJECTS
    """
    if not pathlib.Path("./outputs/mpc_observations.parquet").exists():
        pathlib.Path("./outputs").mkdir(parents=True, exist_ok=True)
        all_obs = None
        orbits = Orbits.empty()
        for object_id, _ in OBJECTS.items():
            obs = MPC.get_observations(object_id).to_pandas()
            if all_obs is None:
                all_obs = obs
            else:
                all_obs = pd.concat([all_obs, obs])
            # Get orbits from sbdb
            orbit = query_sbdb([object_id])
            orbit = orbit.set_column(
                "object_id",
                pa.array(pa.repeat(object_id, len(orbit)), type=pa.large_string()),
            )
            orbits = qv.concatenate([orbits, orbit])
    
        all_obs.to_parquet("./outputs/mpc_observations.parquet")
        orbits.to_parquet("./outputs/sbdb_orbits.parquet")
    all_obs = pq.read_table("./outputs/mpc_observations.parquet")
    orbits = Orbits.from_parquet("./outputs/sbdb_orbits.parquet")
    return orbits, all_obs



def _observers_from_mpc_observations(mpc_observations: pa.Table) -> Observers:
    """
    Convert the MPC observations to Observers
    """
    observers = Observers.empty()
    # Traverse over unique observatories
    for observatory_code in mpc_observations["observatory"].unique():
        observatory_obs = mpc_observations.select("observatory", observatory_code)
        times = Timestamp.from_iso8601([obstime.replace(' UTC', '').replace(' ', 'T') for obstime in observatory_obs["obstime"].to_pylist()])
        observer = Observers.from_code(
            code=observatory_code,
            time=times
        )
        assert len(observatory_obs) == len(observer)
        observers = qv.concatenate([observers, observer])

    return observers

def _ephemeris_from_mpc_observations(mpc_observations: pa.Table) -> Ephemeris:
    coordinates = SphericalCoordinates.from_kwargs(



def test_mpc_residuals():
    """
    Produces ephemeris for objects in MPC using MPC Orbits and observations
    """
    download_jpl_ephemeris_files()
    env_name = os.environ.get("ASSIST_VERSION", "local")

    pathlib.Path("./outputs").mkdir(parents=True, exist_ok=True)

    sbdb_orbits, mpc_observations = fetch_mpc_observations()
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
        for object_id, _ in OBJECTS.items():

            print(f"{object_id} {job_name}")

            # Get the horizons vectors for this object_id
            orbit = sbdb_orbits.select("object_id", object_id)
            object_mpc_observations = mpc_observations.select("designation", object_id)

            # Generate observers from the MPC observations
            observers = _observers_from_mpc_observations(object_mpc_observations)

            props = ASSISTPropagator(
                planets_path=pathlib.Path(DATA_DIR).expanduser().joinpath(planets_file),
                asteroids_path=pathlib.Path(DATA_DIR)
                .expanduser()
                .joinpath(asteroids_file),
            )
            # We use the first vector as the initial state
            assist_ephem = props.generate_ephemeris(
                orbit, observers, covariance=True
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
            safe_object_name = object_id.replace("/", "_")
            residuals_path = (
                f"./outputs/{safe_object_name}_residuals_{job_name}.parquet"
            )
            pq.write_table(residuals_table, residuals_path)




def compare_residuals_old():
    """
    Load in the parquet files in outputs.
    Calculate the cumulative absolute residuals for each unique environment,
    normalized by object_id.
    """
    residuals_files = glob.glob("./outputs/*_residuals_*.parquet")
    residuals_tables = [pq.read_table(file) for file in residuals_files]

    # Combine all the tables
    residuals_table = pa.concat_tables(residuals_tables)

    pq.write_table(residuals_table, "./outputs/all_residuals.parquet")

    # Convert residuals to absolute values
    abs_residuals = pc.abs(residuals_table.column("residuals"))
    # residuals_table = residuals_table.drop_columns("residuals")
    residuals_table = residuals_table.add_column(0, "abs_residuals", abs_residuals)

    # # Group by object_id and env to calculate sum of residuals and count per object_id
    grouped_by_object_env = residuals_table.group_by(["object_id", "env"])

    env_object_sums = grouped_by_object_env.aggregate(
        [("abs_residuals", "sum"), ("abs_residuals", "count")]
    )

    # # Normalize residuals by the count of residuals for each object_id
    # normalized_residuals = pc.divide(grouped_by_object.column("residuals_sum"), grouped_by_object.column("residuals_count"))

    # grouped_by_object = grouped_by_object.drop_columns(["residuals_sum", "residuals_count"])
    # grouped_by_object = grouped_by_object.add_column(0, "normalized_residuals", normalized_residuals)

    # Find the maximum residual for each object_id
    max_object_residuals = env_object_sums.group_by("object_id").aggregate(
        [("abs_residuals_sum", "max")]
    )

    env_object_sums = env_object_sums.join(max_object_residuals, "object_id")

    # Normalize the residuals by the max residual for each object_id
    normalized_residuals = pc.divide(
        env_object_sums.column("abs_residuals_sum"),
        env_object_sums.column("abs_residuals_sum_max"),
    )

    # Invert it so we get the difference from 1
    distance_from_worst_residuals = pc.subtract(1, normalized_residuals)

    env_object_sums = env_object_sums.add_column(
        0, "distance_from_worst_residuals", distance_from_worst_residuals
    )

    # Update pandas to print out full tables (rows)
    pd.set_option("display.max_rows", None)

    # Sort by object_id and then normalized_residuals
    print(
        env_object_sums.sort_by(
            [
                ("object_id", "ascending"),
                ("distance_from_worst_residuals", "descending"),
            ]
        ).to_pandas()[["object_id", "env", "distance_from_worst_residuals"]]
    )

    # Group by env
    print(
        env_object_sums.group_by("env")
        .aggregate([("distance_from_worst_residuals", "sum")])
        .sort_by([("distance_from_worst_residuals_sum", "descending")])
        .to_pandas()
    )


def compare_residuals():
    """
    Load in the parquet files in outputs.
    Calculate the mean residuals per time step for each unique environment and object.
    """
    residuals_files = glob.glob("./outputs/*_residuals_*.parquet")
    residuals_tables = [pq.read_table(file) for file in residuals_files]

    # Combine all the tables
    residuals_table = pa.concat_tables(residuals_tables)

    # Compute mean residual per time step for each object and environment
    grouped = residuals_table.group_by(["object_id", "env"]).aggregate(
        [("residuals", "mean"), ("residuals", "stddev"), ("residuals", "count")]
    )

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

    summary = summary.sort_values(("normalized_residual", "mean"), ascending=True)

    print(summary)

    statistical_test(df)



def statistical_test(df):
    """
    Perform paired t-tests between each pair of environments and print the results in a table.
    
    Parameters:
    df (DataFrame): The DataFrame containing 'env' and 'normalized_residual' columns.
    """
    envs = df['env'].unique()
    results = []

    for i in range(len(envs)):
        for j in range(i+1, len(envs)):
            env1 = envs[i]
            env2 = envs[j]
            # Ensure the data is aligned by 'object_id' to maintain pairing
            data1 = df[df['env'] == env1].set_index('object_id')['normalized_residual']
            data2 = df[df['env'] == env2].set_index('object_id')['normalized_residual']
            # Only keep common object_ids to ensure proper pairing
            common_objects = data1.index.intersection(data2.index)
            paired_data1 = data1.loc[common_objects]
            paired_data2 = data2.loc[common_objects]
            # Perform paired t-test
            stat, p = ttest_rel(paired_data1, paired_data2)
            results.append({
                'Environment 1': env1,
                'Environment 2': env2,
                'p-value': p
            })

    # Create a DataFrame for the results
    results_df = pd.DataFrame(results)

    # Format p-values for readability
    results_df['p-value'] = results_df['p-value'].apply(lambda x: f'{x:.3e}')

    # Print the results in a nicely formatted table
    print("\nPaired t-test Results:")
    print(results_df.to_string(index=False))

def plot_residuals(df):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='env', y='normalized_residual', data=df)
    plt.title('Normalized Residuals per Environment')
    plt.ylabel('Normalized Residual')
    plt.xlabel('Environment')
    # Make sure the labels are angled to make them readable
    plt.xticks(rotation=45, ha='right')
    plt.show()