import pyarrow as pa
import pyarrow.compute as pc
import pytest
from adam_core.observers import Observers
from adam_core.orbits.query.horizons import query_horizons, query_horizons_ephemeris
from adam_core.time import Timestamp
from numpy.testing import assert_allclose

from src.adam_core.propagator.adam_assist import ASSISTPropagator


def test_ephemeris():
    """
    Test the accurate of the ephemeris generator by comparing the propagated orbit to the JPL ephemeris
    """
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
        "A802 FA",
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
    ]

    mjds = [60000]
    times = Timestamp.from_mjd(mjds)
    objects = query_horizons(OBJECT_IDS, times)
    # Pick a series of mjd around 2024-07-25
    delta_times = Timestamp.from_mjd(pc.add(times.mjd()[0], pa.array([-10, -5, 0, 5, 10])))
    observers = Observers.from_code("500", delta_times)
    rtol = 1e3
    atol = 1e-4

    for object, object_id in zip(objects, OBJECT_IDS):
        assist_ephem = prop.generate_ephemeris(object, observers)
        jpl_ephem = query_horizons_ephemeris([object_id], observers)

        # Compare the two ephemeris
        assert_allclose(assist_ephem.light_time, jpl_ephem["lighttime"] / (60 * 24), rtol=rtol, atol=1 / (86400 * 1000), err_msg=f"Failed lighttime for {object.object_id}")
        assert_allclose(assist_ephem.coordinates.lon, jpl_ephem["RA"], rtol=rtol, atol=atol, err_msg=f"Failed RA for {object.object_id}")
        assert_allclose(assist_ephem.coordinates.lat, jpl_ephem["DEC"], rtol=rtol, atol=atol, err_msg=f"Failed DEC for {object.object_id}")