import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest
from adam_core.coordinates import CartesianCoordinates, Origin
from adam_core.coordinates.residuals import Residuals
from adam_core.orbits import Orbits
from adam_core.orbits.query.horizons import query_horizons
from adam_core.time import Timestamp
from astropy import units as u

from adam_assist import ASSISTPropagator

DEFAULT_POSITION_TOLERANCE = (50 * u.m).to(u.au).value
DEFAULT_VELOCITY_TOLERANCE = (1 * u.mm / u.s).to(u.au / u.day).value


OBJECTS = {
    "2020 AV2": {
        "position": DEFAULT_POSITION_TOLERANCE,
        "velocity": DEFAULT_VELOCITY_TOLERANCE,
    },
    "2003 CP20": {
        "position": DEFAULT_POSITION_TOLERANCE,
        "velocity": DEFAULT_VELOCITY_TOLERANCE,
    },
    "2010 TK7": {
        "position": DEFAULT_POSITION_TOLERANCE,
        "velocity": DEFAULT_VELOCITY_TOLERANCE,
    },
    "1986 TO": {
        "position": DEFAULT_POSITION_TOLERANCE,
        "velocity": DEFAULT_VELOCITY_TOLERANCE,
    },
    "2000 PH5": {
        # Accomodate 2 km uncertainty
        "position": (2 * u.km).to(u.au).value, 
        "velocity": DEFAULT_VELOCITY_TOLERANCE,
    },
    "1977 HB": {
        "position": (500 * u.m).to(u.au).value,
        "velocity": DEFAULT_VELOCITY_TOLERANCE,
    },
    "1932 EA1": {
        "position": DEFAULT_POSITION_TOLERANCE,
        "velocity": DEFAULT_VELOCITY_TOLERANCE,
    },
    "A898 PA": {
        "position": DEFAULT_POSITION_TOLERANCE,
        "velocity": DEFAULT_VELOCITY_TOLERANCE,
    },
    "1980 PA": {
        "position": (200 * u.m).to(u.au).value,
        "velocity": DEFAULT_VELOCITY_TOLERANCE,
    },
    "A898 RB": {
        "position": DEFAULT_POSITION_TOLERANCE,
        "velocity": DEFAULT_VELOCITY_TOLERANCE,
    },
    "1970 BA": {
        "position": DEFAULT_POSITION_TOLERANCE,
        "velocity": DEFAULT_VELOCITY_TOLERANCE,
    },
    "1973 EB": {
        "position": DEFAULT_POSITION_TOLERANCE,
        "velocity": DEFAULT_VELOCITY_TOLERANCE,
    },
    "A847 NA": {
        "position": DEFAULT_POSITION_TOLERANCE,
        "velocity": DEFAULT_VELOCITY_TOLERANCE,
    },
    "1991 NQ": {
        "position": DEFAULT_POSITION_TOLERANCE,
        "velocity": DEFAULT_VELOCITY_TOLERANCE,
    },
    "1988 RJ13": {
        "position": DEFAULT_POSITION_TOLERANCE,
        "velocity": DEFAULT_VELOCITY_TOLERANCE,
    },
    "1999 FM9": {
        "position": DEFAULT_POSITION_TOLERANCE,
        "velocity": DEFAULT_VELOCITY_TOLERANCE,
    },
    "1998 SG172": {
        "position": DEFAULT_POSITION_TOLERANCE,
        "velocity": DEFAULT_VELOCITY_TOLERANCE,
    },
    "A919 FB": {
        "position": DEFAULT_POSITION_TOLERANCE,
        "velocity": DEFAULT_VELOCITY_TOLERANCE,
    },
    "1930 BH": {
        "position": DEFAULT_POSITION_TOLERANCE,
        "velocity": DEFAULT_VELOCITY_TOLERANCE,
    },
    "1930 UA": {
        "position": DEFAULT_POSITION_TOLERANCE,
        "velocity": DEFAULT_VELOCITY_TOLERANCE,
    },
    "1984 KF": {
        "position": DEFAULT_POSITION_TOLERANCE,
        "velocity": DEFAULT_VELOCITY_TOLERANCE,
    },
    "1992 AD": {
        "position": DEFAULT_POSITION_TOLERANCE,
        "velocity": DEFAULT_VELOCITY_TOLERANCE,
    },
    "1991 DA": {
        "position": DEFAULT_POSITION_TOLERANCE,
        "velocity": DEFAULT_VELOCITY_TOLERANCE,
    },
    "1992 QB1": {
        "position": DEFAULT_POSITION_TOLERANCE,
        "velocity": DEFAULT_VELOCITY_TOLERANCE,
    },
    "1993 SB": {
        "position": DEFAULT_POSITION_TOLERANCE,
        "velocity": DEFAULT_VELOCITY_TOLERANCE,
    },
    "1993 SC": {
        "position": DEFAULT_POSITION_TOLERANCE,
        "velocity": DEFAULT_VELOCITY_TOLERANCE,
    },
    "A/2017 U1": {
        "position": DEFAULT_POSITION_TOLERANCE,
        "velocity": DEFAULT_VELOCITY_TOLERANCE,
    },
    # We don't currently have an easy way to propagate Pallas (or the other asteroids in DE441) as they perturb themselves.
    # Instead one would have to use ASSIST's .get_particle function to get the state directly from the spice kernel.
    # "A802 FA",
}


def test_propagate():
    """
    Test the accurate of the ephemeris generator by comparing the propagated orbit to the JPL ephemeris
    """
    
    prop = ASSISTPropagator()
    millisecond_in_days = 1.1574074074074073e-8

    start_time_mjd = Timestamp.from_mjd([60000], scale="tdb")
    delta_times = Timestamp.from_mjd(
        pc.add(start_time_mjd.mjd()[0], pa.array([-300, -150, 0, 150, 300])),
        scale="tdb",
    )

    for object_id in OBJECTS.keys():
        # We need to start with the same initial conditions as Horizons
        horizons_start = query_horizons([object_id], start_time_mjd)
        horizons_propagated_orbits = query_horizons([object_id], delta_times)
        assist_propagated_orbits = prop.propagate_orbits(
            horizons_start, horizons_propagated_orbits.coordinates.time, covariance=True
        )

        ephem_times_difference = pc.subtract(
            assist_propagated_orbits.coordinates.time.mjd(), horizons_propagated_orbits.coordinates.time.mjd()
        )
        np.testing.assert_array_less(
            np.abs(ephem_times_difference.to_numpy(zero_copy_only=False)),
            millisecond_in_days,
            err_msg=f"ASSIST produced significantly different epochs than Horizons for {object_id}",
        )
        # Calculate the absolute magnitude of position and velocity vectors
        absolute_position = np.linalg.norm(
            assist_propagated_orbits.coordinates.r
            - horizons_propagated_orbits.coordinates.r,
            axis=1,
        )

        absolute_velocity = np.linalg.norm(
            assist_propagated_orbits.coordinates.v
            - horizons_propagated_orbits.coordinates.v,
            axis=1,
        )
        pos_tol = OBJECTS.get(object_id).get("position")
        vel_tol = OBJECTS.get(object_id).get("velocity")

        np.testing.assert_array_less(absolute_position, pos_tol, f"Failed position for {object_id}")
        np.testing.assert_array_less(absolute_velocity, vel_tol, f"Failed velocity for {object_id}")


def test_propagate_different_input_times(mocker):
    """
    Ensure that we can pass in vectors with different epochs
    """
    
    prop = ASSISTPropagator()
    watched_propagate_orbits_inner = mocker.spy(prop, "_propagate_orbits_inner")
    orbits = Orbits.from_kwargs(
        orbit_id=["1", "2", "3", "4"],
        object_id=["1", "2", "3", "4"],
        coordinates=CartesianCoordinates.from_kwargs(
            x=[1, 1, 1, 1],
            y=[1, 1, 1, 1],
            z=[1, 1, 1, 1],
            vx=[1, 1, 1, 1],
            vy=[1, 1, 1, 1],
            vz=[1, 1, 1, 1],
            time=Timestamp.from_mjd([60000, 60000, 60000, 60001], scale="tdb"),
            frame="ecliptic",
            origin=Origin.from_kwargs(code=pa.repeat("SOLAR_SYSTEM_BARYCENTER", 4)),
        ),
    )

    propagated_orbits = prop.propagate_orbits(orbits, Timestamp.from_mjd([60005, 60006], scale="tdb"))

    assert watched_propagate_orbits_inner.call_count == 2, "Inner function should be called once for each unique input epoch"

    assert len(propagated_orbits.coordinates.time.unique()) == 2
    assert len(propagated_orbits) == 8, "Should have input orbits x epochs number of results"