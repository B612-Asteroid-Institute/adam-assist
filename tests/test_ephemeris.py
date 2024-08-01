import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest
from adam_core.observers import Observers
from adam_core.orbits.query.horizons import query_horizons, query_horizons_ephemeris
from adam_core.time import Timestamp
from astropy import units as u
from numpy.testing import assert_allclose

from src.adam_core.propagator.adam_assist import (
    ASSISTPropagator,
    download_jpl_ephemeris_files,
)


def test_ephemeris():
    """
    Test the accurate of the ephemeris generator by comparing the propagated orbit to the JPL ephemeris
    """
    download_jpl_ephemeris_files()
    prop = ASSISTPropagator()
    OBJECT_IDS = [
        "2020 AV2",
        "2003 CP20",
        "2010 TK7",
        "1986 TO",
        "2000 PH5",
        "1977 HB",
        "1932 EA1",
        "A898 PA",
        "1980 PA",
        "A898 RB",
        "1970 BA",
        "1973 EB",
        "A847 NA",
        "1991 NQ",
        "1988 RJ13",
        "1999 FM9",
        "1998 SG172",
        "A919 FB",
        "1930 BH",
        "1930 UA",
        "1984 KF",
        "1992 AD",
        "1991 DA",
        "1992 QB1",
        "1993 SB",
        "1993 SC",
        "A/2017 U1",
        # We don't currently have an easy way to propagate Pallas (or the other asteroids in DE441) as they perturb themselves.
        # Instead one would have to use ASSIST's .get_particle function to get the state directly from the spice kernel.
        # "A802 FA",
    ]

    start_time_mjd = Timestamp.from_mjd([60000], scale="utc")
    delta_times = Timestamp.from_mjd(
        pc.add(start_time_mjd.mjd()[0], pa.array([-300, -150, 0, 150, 300])),
        scale="utc",
    )
    observers = Observers.from_code("500", delta_times)

    on_sky_rtol = 1e-7
    one_milliarcsecond = 2.77778e-7  # 1 milliarcsecond
    millisecond_in_days = 1.1574074074074073e-8

    light_time_rtol = 1e-8
    light_time_atol = 3 * millisecond_in_days  # milliseconds in units of days

    for object_id in OBJECT_IDS:
        horizons_start_vector = query_horizons([object_id], start_time_mjd)
        horizons_ephem = query_horizons_ephemeris([object_id], observers)
        requested_vs_received = pc.subtract(
            observers.coordinates.time.mjd(), horizons_ephem.coordinates.time.mjd()
        )

        np.testing.assert_array_less(
            np.abs(requested_vs_received.to_numpy(zero_copy_only=False)),
            millisecond_in_days,
            err_msg=f"Horizons returned significantly different epochs than were requested.",
        )

        assist_ephem = prop.generate_ephemeris(
            horizons_start_vector, observers, covariance=True
        )

        ephem_times_difference = pc.subtract(
            assist_ephem.coordinates.time.mjd(), horizons_ephem.coordinates.time.mjd()
        )
        np.testing.assert_array_less(
            np.abs(ephem_times_difference.to_numpy(zero_copy_only=False)),
            millisecond_in_days,
            err_msg=f"ASSIST produced significantly different epochs than Horizons for {object_id}",
        )

        np.testing.assert_allclose(
            assist_ephem.light_time.to_numpy(zero_copy_only=False),
            horizons_ephem.light_time.to_numpy(zero_copy_only=False),
            rtol=light_time_rtol,
            atol=light_time_atol,
            err_msg=f"Failed lighttime for {object_id}",
        )

        assert_allclose(
            assist_ephem.coordinates.lon.to_numpy(zero_copy_only=False),
            horizons_ephem.coordinates.lon.to_numpy(zero_copy_only=False),
            rtol=on_sky_rtol,
            atol=one_milliarcsecond,
            err_msg=f"Failed RA for {object_id}",
        )

        assert_allclose(
            assist_ephem.coordinates.lat.to_numpy(zero_copy_only=False),
            horizons_ephem.coordinates.lat.to_numpy(zero_copy_only=False),
            rtol=on_sky_rtol,
            atol=one_milliarcsecond,
            err_msg=f"Failed Dec for {object_id}",
        )

        # Get the difference in magnitude for the lon/lats
        on_sky_difference = np.linalg.norm(
            assist_ephem.coordinates.values[:, 1:3]
            - horizons_ephem.coordinates.values[:, 1:3],
            axis=1,
        )

        np.testing.assert_array_less(
            on_sky_difference,
            2 * one_milliarcsecond,
            err_msg=f"Failed on sky for {object_id}",
        )
