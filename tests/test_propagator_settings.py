import pyarrow as pa
import pytest
from adam_core.coordinates import CartesianCoordinates, Origin
from adam_core.orbits import Orbits
from adam_core.time import Timestamp

from adam_assist import ASSISTPropagator


@pytest.fixture
def basic_orbit():
    """Create a basic test orbit"""
    return Orbits.from_kwargs(
        orbit_id=["test"],
        coordinates=CartesianCoordinates.from_kwargs(
            x=[1.0],
            y=[0.0],
            z=[0.0],
            vx=[0.0],
            vy=[1.0],
            vz=[0.0],
            time=Timestamp.from_mjd([60000], scale="tdb"),
            origin=Origin.from_kwargs(code=["SUN"]),
            frame="ecliptic",
        ),
    )

def test_default_settings():
    """Test that default settings are applied correctly"""
    prop = ASSISTPropagator()
    assert prop.min_dt == 1e-15
    assert prop.initial_dt == 0.001
    assert prop.adaptive_mode == 2

def test_custom_settings():
    """Test that custom settings are applied correctly"""
    prop = ASSISTPropagator(min_dt=1e-12, initial_dt=0.01, adaptive_mode=1)
    assert prop.min_dt == 1e-12
    assert prop.initial_dt == 0.01
    assert prop.adaptive_mode == 1

def test_invalid_min_dt():
    """Test that invalid min_dt raises ValueError"""
    with pytest.raises(ValueError, match="min_dt must be positive"):
        ASSISTPropagator(min_dt=-1e-15)
    
    with pytest.raises(ValueError, match="min_dt must be positive"):
        ASSISTPropagator(min_dt=0)

def test_invalid_initial_dt():
    """Test that invalid initial_dt raises ValueError"""
    with pytest.raises(ValueError, match="initial_dt must be positive"):
        ASSISTPropagator(initial_dt=-0.001)
    
    with pytest.raises(ValueError, match="initial_dt must be positive"):
        ASSISTPropagator(initial_dt=0)

def test_min_dt_greater_than_initial():
    """Test that min_dt > initial_dt raises ValueError"""
    with pytest.raises(ValueError, match="min_dt must be smaller than initial_dt"):
        ASSISTPropagator(min_dt=0.1, initial_dt=0.01)

def test_propagation_with_different_settings(basic_orbit):
    """Test that propagation works with different settings"""
    # Test with default settings
    prop_default = ASSISTPropagator()
    
    # Test with more restrictive settings
    prop_restrictive = ASSISTPropagator(min_dt=1e-12, initial_dt=0.0001)
    
    # Test with less restrictive settings
    prop_loose = ASSISTPropagator(min_dt=1e-9, initial_dt=0.01)
    
    # Propagate for 10 days with each propagator
    target_time = Timestamp.from_mjd([60010], scale="tdb")
    
    result_default = prop_default.propagate_orbits(basic_orbit, target_time)
    result_restrictive = prop_restrictive.propagate_orbits(basic_orbit, target_time)
    result_loose = prop_loose.propagate_orbits(basic_orbit, target_time)
    
    # All should produce results
    assert len(result_default) == 1
    assert len(result_restrictive) == 1
    assert len(result_loose) == 1
    
    # Results should be similar but not identical due to different step sizes
    # Using a relatively loose tolerance since we expect some differences
    tolerance = 1e-6
    
    default_pos = result_default.coordinates.values[0, :3]
    restrictive_pos = result_restrictive.coordinates.values[0, :3]
    loose_pos = result_loose.coordinates.values[0, :3]
    
    assert abs(default_pos - restrictive_pos).max() < tolerance
    assert abs(default_pos - loose_pos).max() < tolerance