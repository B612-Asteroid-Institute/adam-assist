# adam-assist

[![PyPI - Version](https://img.shields.io/pypi/v/adam-assist.svg)](https://pypi.org/project/adam-assist)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/adam-assist.svg)](https://pypi.org/project/adam-assist)

-----

**Table of Contents**


- [Installation](#installation)
- [Usage](#usage)
  - [Propagating Orbits](#propagating-orbits)
  - [Generating Ephemerides](#generating-ephemerides)


## Overview
`adam-assist` is a pluggable propagator class for the `adam-core` package that uses [ASSIST](https://github.com/matthewholman/assist) for propagating orbits.


## Installation

```console
pip install adam-assist
```

## Usage

### Propagating Orbits

Here we initialize a set of `adam_core.orbit.Orbit` objects from the JPL Small Bodies Database and propagate them using the `AdamAssistPropagator` class. You can manually initialize the orbits as well.

```python
from adam_core.orbits.query.sbdb import query_sbdb
from adam_core.time import Timestamp
from adam_assist import ASSISTPropagator

# Query the JPL Small Bodies Database for a set of orbits
sbdb_orbits = query_sbdb(["2020 AV2", "A919 FB", "1993 SB"])
times = Timestamp.from_mjd([60000, 60365, 60730], scale="tdb")


propagator = ASSISTPropagator()

propagated = propagator.propagate_orbits(sbdb_orbits, times)
```

Of course you can define your own orbits as well.

```python
import pyarrow as pa
from adam_core.orbits import Orbit
from adam_core.coordinates import CartesianCoordinates, Origin
from adam_core.time import Timestamp
from adam_assist import ASSISTPropagator

# Define an orbit
orbits = Orbit.from_kwargs(
  orbit_id=["1", "2", "3"],
  coordinates=CartesianCoordinates.from_kwargs(
    # use realistic cartesian coords in AU and AU/day
    x=[-1.0, 0.0, 1.0],
    y=[-1.0, 0.0, 1.0],
    z=[-1.0, 0.0, 1.0],
    vx=[-0.1, 0.0, 0.1],
    vy=[-0.1, 0.0, 0.1],
    vz=[-0.1, 0.0, 0.1],
    time=Timestamp.from_mjd([60000, 60365, 60730], scale="tdb"),
    origin=Origin.from_kwargs(code=pa.repeat("SUN", 3)),
    frame="eliptic"
  ),
)

propagator = ASSISTPropagator()

propagated = propagator.propagate_orbits(orbits)
```

### Generating Ephemerides

The `ASSISTPropagator` class uses the `adam-core` default ephemeris generator to generate ephemerides from the `ASSIST` propagated orbits. The default ephemeris generator accounts for light travel time and aberration. See `adam_core.propagator.propagator.EphemerisMixin` for implementation details.


```python
from adam_core.orbits.query.sbdb import query_sbdb
from adam_core.time import Timestamp
from adam_core.observers import Observers
from adam_assist import ASSISTPropagator

# Query the JPL Small Bodies Database for a set of orbits
sbdb_orbits = query_sbdb(["2020 AV2", "A919 FB", "1993 SB"])
times = Timestamp.from_mjd([60000, 60365, 60730], scale="utc")
observers = Observers.from_code("399", times)
propagator = ASSISTPropagator()

ephemerides = propagator.generate_ephemeris(sbdb_orbits, observers)
```

## Configuration

When initializing the `ASSISTPropagator`, you can configure several parameters that control the integration. 
These parameters are passed directly to REBOUND's IAS15 integrator. The IAS15 integrator is a high accuracy integrator that uses adaptive timestepping to maintain precision while optimizing performance.

- `min_dt`: Minimum timestep for the integrator (default: 1e-15 days)
- `initial_dt`: Initial timestep for the integrator (default: 0.001 days)  
- `adaptive_mode`: Controls the adaptive timestep behavior (default: 2)

These parameters are passed directly to REBOUND's IAS15 integrator. The IAS15 integrator is a high accuracy integrator that uses adaptive timestepping to maintain precision while optimizing performance.

Example:

```python
propagator = ASSISTPropagator(min_dt=1e-12, initial_dt=0.0001, adaptive_mode=2)
```

