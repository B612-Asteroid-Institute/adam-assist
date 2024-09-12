import glob
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pcsv
import pyarrow.parquet as pq
import pytest
from adam_core.constants import KM_P_AU
from adam_core.orbits import Orbits
from adam_core.orbits.query.horizons import query_horizons
from adam_core.time import Timestamp
from numpy.typing import NDArray

from src.adam_core.propagator.adam_assist import (
    ASSISTPropagator,
    download_jpl_ephemeris_files,
)

OBJECTS = {
    # This NEA has a close approach around Sept. 26, 2029 (62405)
    # Cover times a year before and 5 years after
    "2005 YY128": Timestamp.from_mjd(
        pc.add(62405, pa.array(range(-365, 5 * 365, 90))), scale="utc"
    ),
    # Do a 10 year propagation of Hollman starting 60600 and stepping every 100 days
    # This is roughly twice its period
    "3666": Timestamp.from_mjd(
        pc.add(60600, pa.array(range(0, 3650, 100))), scale="utc"
    ),
    "2020 AV2": Timestamp.from_mjd(
        pc.add(60000, pa.array(range(-365, 5 * 365, 90))), scale="utc"
    ),
    "136199": Timestamp.from_mjd(
        pc.add(60000, pa.array(range(0, 50 * 365, 365))), scale="utc"
    ),
    # Item on the risk list with near term close approach
    "2022 YO1": Timestamp.from_mjd(
        pc.add(60366.26, pa.array(range(-60, 60, 10))), scale="utc"
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


# Grab arbitrary run name
def test_complete_residuals():
    """
    Test how closely assist matches SBDB and Horizons over time
    """
    download_jpl_ephemeris_files()
    env_name = os.environ.get("ASSIST_VERSION", "local")

    from assist.ephem import Ephem

    pathlib.Path("./outputs").mkdir(parents=True, exist_ok=True)

    for object_id, times in OBJECTS.items():
        # Get the SBDB vectors
        horizons_vectors = query_horizons([object_id], times)

        props = ASSISTPropagator()
        # We use the first vector as the initial state
        assist_vectors = props.propagate_orbits(
            horizons_vectors[0], horizons_vectors.coordinates.time
        )

        # Calculate the positional distance between the two vectors
        residuals = np.linalg.norm(
            assist_vectors.coordinates.r - horizons_vectors.coordinates.r,
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
                "time": horizons_vectors.coordinates.time.mjd(),
                "env": pa.repeat(env_name, len(residuals)),
            }
        )
        residuals_path = f"./outputs/{object_id}_residuals_{env_name}.parquet"
        pq.write_table(residuals_table, residuals_path)

        chart_path = f"./outputs/{object_id}_position_residuals_{env_name}.png"
        chart_residuals(
            chart_path, env_name, object_id, horizons_vectors, assist_vectors, residuals
        )


def compare_residuals():
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
    residuals_table = residuals_table.drop_columns("residuals")
    residuals_table = residuals_table.add_column(0, "residuals", abs_residuals)

    # Group by object_id and env to calculate sum of residuals and count per object_id
    grouped_by_object = residuals_table.group_by(["object_id", "env"]).aggregate(
        [("residuals", "sum"), ("residuals", "count")]
    )

    # Normalize residuals by the count of residuals for each object_id
    normalized_residuals = pc.divide(grouped_by_object.column("residuals_sum"), grouped_by_object.column("residuals_count"))
    
    grouped_by_object = grouped_by_object.drop_columns(["residuals_sum", "residuals_count"])
    grouped_by_object = grouped_by_object.add_column(0, "normalized_residuals", normalized_residuals)

    # # Group by environment and sum the normalized residuals
    # grouped_by_env = grouped_by_object.group_by(["env"]).aggregate(
    #     [("normalized_residuals", "sum")]
    # )

    # # Sort by the sum of normalized residuals
    # cumulative_residuals = grouped_by_env.sort_by("normalized_residuals_sum")

    print(grouped_by_object.sort_by([("object_id", "ascending"),("normalized_residuals", "ascending")]).to_pandas())

    # print(cumulative_residuals.to_pandas())s