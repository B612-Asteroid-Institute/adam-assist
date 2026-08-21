# adam-assist

[![PyPI - Version](https://img.shields.io/pypi/v/adam-assist.svg)](https://pypi.org/project/adam-assist)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/adam-assist.svg)](https://pypi.org/project/adam-assist)

-----

**Table of Contents**


- [Installation](#installation)
- [Usage](#usage)
  - [Propagating Orbits](#propagating-orbits)
  - [Non-Gravitational Forces](#non-gravitational-forces)
  - [Generating Ephemerides](#generating-ephemerides)
- [Benchmarking](#benchmarking)


## Overview
`adam-assist` is a pluggable propagator class for the `adam-core` package that uses [ASSIST](https://github.com/matthewholman/assist) for propagating orbits.


## Installation

```console
pip install adam-assist
```

Native wheels support CPython 3.11-3.13 on manylinux 2.17+ x86-64/AArch64 and
macOS Apple silicon/Intel. Windows is currently unsupported because
``libassist-sys 1.2.1`` wraps upstream ASSIST code that requires POSIX
``sys/mman.h`` memory mapping; no Windows port is bundled. Musllinux is also
unsupported.

## Usage

### Propagating Orbits

Here we initialize a set of `adam_core.orbit.Orbit` objects from the JPL Small Bodies Database and propagate them using the Rust-backed `ASSISTPropagator` class. You can manually initialize the orbits as well.

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

### Non-Gravitational Forces

Orbits whose `non_gravitational_parameters` carry Marsden-style `A1`/`A2`/`A3`
accelerations (au/d^2) are propagated with ASSIST's non-gravitational force.
The g(r) law is selected by the `ALN`/`NK`/`NM`/`NN`/`R0` columns: null
constants mean the asteroid convention g(r) = (1 au / r)^2, and an explicit
tuple (for example the standard Marsden comet law, or a custom shape such as
1I/'Oumuamua's) is applied simulation-wide. Batches mixing different g(r)
tuples are automatically split into one simulation per force law, since
ASSIST holds the constants per simulation rather than per particle.

**Supported** (regression-tested against JPL Horizons to tens–hundreds of
metres over ±300 days): symmetric inverse-square asteroid solutions
(e.g. 99942 Apophis), the standard comet Marsden law (e.g. C/2022 E3), and
custom symmetric (ALN, NK, NM, NN, R0) tuples (e.g. 1I/'Oumuamua).

**Not supported**: the asymmetric-outgassing time offset `DT` (dropped with
a warning by adam-core's importers — solutions that estimate DT, such as
67P or 81P, retain km-scale model error through perihelion and should not
be treated as JPL-parity), thermophysical Yarkovsky (`AMRAT`/`RHO`) models,
and estimated (rather than fixed) g(r) shape parameters. Non-finite
A-values, partial constants tuples, and degenerate `ALN <= 0` / `R0 <= 0`
values are rejected with a `ValueError` rather than silently altering the
force.

### Generating Ephemerides

`ASSISTPropagator.generate_ephemeris` performs propagation, light-time geometry, optional covariance sampling/collapse, aberration, and photometry in the Rust backend behind one public Python call. Local parallelism uses Rayon rather than adam-core's former Python/Ray composition.


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

## Benchmarking

Run the complete current-only suite with:

```console
pdm run benchmark-current
```

It reuses the existing propagation, nongrav, ephemeris/covariance, collision,
and orbit-determination workload builders. Results include current public
Python timings, genuine Rust-owned `std::time::Instant` timings where
available, public/native overhead, and exact workload shapes. It does not
require a frozen Python environment or baseline timing cache, and all ASSIST
workloads use `max_processes=1`. Use `--quick` for a smoke run or select, for
example, `--domains nongrav ephemeris covariance --lanes tiny small`. Release
CI runs the complete 35-workload grid with native timing required.

The deterministic test suite uses reviewed, hash-pinned frozen outputs for all
formerly two-runtime propagation, covariance, ephemeris, collision, typed
mapping, and OD/Vallado parity cases. Normal pytest and the current benchmark
suite do not import or launch the archived ASSIST oracle and never skip because
a legacy virtual environment is absent.

## Configuration

When initializing the `ASSISTPropagator`, you can configure several parameters that control the integration. 
These parameters are passed directly to REBOUND's IAS15 integrator. The IAS15 integrator is a high accuracy integrator that uses adaptive timestepping to maintain precision while optimizing performance.

- `min_dt`: Minimum timestep for the integrator (default: 1e-12 days)
- `initial_dt`: Initial timestep for the integrator (default: 0.001 days)
- `epsilon`: Controls the adaptive timestep behavior (default: 1e-6)
- `adaptive_mode`: Controls the adaptive timestep behavior (default: 1)

These parameters are passed directly to REBOUND's IAS15 integrator. The IAS15 integrator is a high accuracy integrator that uses adaptive timestepping to maintain precision while optimizing performance.

Example:

```python
propagator = ASSISTPropagator(
  min_dt=1e-12,
  initial_dt=0.0001,
  epsilon=1e-6,
  adaptive_mode=1
)
```

When initializing the `ASSISTPropagator`, you can configure several parameters that control the integration. 
These parameters are passed directly to REBOUND's IAS15 integrator. The IAS15 integrator is a high accuracy integrator that uses adaptive timestepping to maintain precision while optimizing performance.

## Default SPK Files

The asteroids SPK file sb441-n16.bsp contains the 16 largest asteroids in the solar system. They are listed here by number for reference:

1 Ceres
3 Juno
4 Vesta
7 Iris
10 Hygiea
15 Eunomia
16 Psyche
31 Euphrosyne
52 Europa
65 Cybele
70 Panopaea
87 Sylvia
88 Thisbe
107 Camilla
511 Davida
704 Interamnia