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

### adam-core compatibility and exact-source validation

The `0.6.0` source line requires `adam-core>=0.7.0,<0.8.0` and the coherent
`0.7` Rust crate line. It is validated against the exact reviewed adam-core
source commit `0b49938eb0be04214a8d07b206227d31b248394a`, which has
package/crate version
`0.7.0` and defines the private `adam_core.propagator.v2` ABI. Released
`adam-core 0.6.1` is outside the declared range and has no v2 ABI.

Core `0.7.0` is not yet available from PyPI/crates.io. Active PDM and Cargo
registry locks are therefore intentionally absent instead of retaining the
incompatible `0.6.1` graph or inventing checksums. The machine-checked
`migration/source_pair_contract.json` pins this exact pre-release pairing.
Normal CI builds both projects from those sources, installs the built ASSIST
wheel with its complete `dev` extra, and verifies `astropy` plus every frozen
NAIF data version before tests and benchmarks. It is the repository's only
workflow. No release-candidate or publication automation exists while Core
`0.7.0` is unpublished and registry locks are absent.
`migration/scripts/verify_workflow_policy.py` pins the exact read-only workflow
content and explicitly checks its triggers, permissions, credentials boundary,
source-pair matrix, and required validation commands.
Release automation and authentic registry locks require a separate reviewed
post-Core-publication commit.
The retained migration helpers are documented as inert local tools in
`migration/scripts/RELEASE_TOOLING.md`; they are not CI release entry points.

For reproducible local Rust validation, use one detached Core tree for all
four Core crates. Patched Cargo commands create a temporary untracked lock;
remove it afterward and verify that no tracked or untracked changes remain.
The command below runs only the all-feature `cargo test --all-targets` lane.
It does not run formatting, Clippy, doctests, or docs. The authoritative CI
invokes separate `fmt`, `check`, warning-denied Clippy, test, doctest, and
warning-denied doc commands across default, all, and no-default features:

```console
git clone --no-hardlinks https://github.com/B612-Asteroid-Institute/adam_core.git ../adam-core-0b49938e
git -C ../adam-core-0b49938e checkout --detach 0b49938eb0be04214a8d07b206227d31b248394a
CORE="$(cd ../adam-core-0b49938e && pwd)"
test "$(git -C "$CORE" rev-parse HEAD)" = 0b49938eb0be04214a8d07b206227d31b248394a
python migration/scripts/verify_source_pair_contract.py --core-tree "$CORE"

cargo test --manifest-path rust/adam_assist_rs/Cargo.toml \
  --all-targets --features python,extension-module \
  --config "patch.crates-io.adam_core.path='$CORE/rust/adam_core'" \
  --config "patch.crates-io.adam_core_rs_coords.path='$CORE/rust/adam_core_rs_coords'" \
  --config "patch.crates-io.adam_core_rs_spice.path='$CORE/rust/adam_core_rs_spice'" \
  --config "patch.crates-io.adam_core_rs_kernel_data.path='$CORE/rust/adam_core_rs_kernel_data'"
rm -f rust/adam_assist_rs/Cargo.lock
test -z "$(git status --porcelain)"
```

### Pure Rust

Rust-only consumers use `AssistPropagator` directly for propagation. For
Core-owned canonical OD workflows, `AssistPropagator::external_propagator()`
returns the owned ABI-v2 view accepted by `adam_core::od::OdPropagator::external`.

```rust,no_run
use adam_assist::{AssistPropagator, AssistResult};

fn main() -> AssistResult<()> {
    let propagator = AssistPropagator::from_default_kernels()?;
    let _integrator = propagator.integrator();
    Ok(())
}
```

The default resolver loads DE440 and SB441-n16 through adam-core's environment,
installed-package, cache, and checksummed-fetch policy. Explicit/offline users
can call `AssistPropagator::from_paths` and can disable default resolver support
with `default-features = false`. Kernel files are immutable package-managed
resources and must remain unchanged for the provider's lifetime. Explicit-path
construction does not hash the 765 MiB pair. It binds a small digest of the DAF
file record, comments, complete summary chain, mapped length, and retained file
identity immediately around ASSIST's eager parse. The first actual kernel use—
ordinary propagation or external-provider metadata—checks that binding and
hashes a stable snapshot of the complete retained bytes.

Ordinary detectable replacement, truncation, corruption, or changes between
construction and first use return an error; exact restoration on the original
mapping recovers. These checks protect provenance for stable resources, not
against a hostile process mutating an existing shared mapping. Concurrent
mutation is unsupported, and a legitimate kernel update requires constructing
a new provider. Chebyshev data-region bytes that ASSIST did not read during
construction remain live through its mapping and are intentionally adopted by
the first complete fingerprint when the eager/mapping binding is intact.

All `adam_assist::Ephemeris::from_paths` calls in one linked crate instance,
including those used by `AssistData`/`AssistPropagator`, share one
process-global mutex around upstream ASSIST's non-reentrant SPK parser.
Propagation and ephemeris queries are not serialized. This guarantee does not
cover callers that construct `libassist_sys::Ephemeris` directly, a separately
linked copy of the crate/library, or external C/FFI users of upstream ASSIST.

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
The g(r) law is selected by the `ALN`/`NK`/`NM`/`NN`/`R0` columns. Each null
field independently takes its asteroid default (`1`, `0`, `2`, `5.093`, `1`),
so an all-null row means g(r) = (1 au / r)^2 and a partial row combines its
present values with those defaults. An explicit tuple (for example the standard
Marsden comet law, or a custom shape such as 1I/'Oumuamua's) is applied
simulation-wide. Batches mixing different g(r) tuples are automatically split
into one simulation per force law, since ASSIST holds the constants per
simulation rather than per particle.

**Supported** (regression-tested against JPL Horizons to tens–hundreds of
metres over ±300 days): symmetric inverse-square asteroid solutions
(e.g. 99942 Apophis), the standard comet Marsden law (e.g. C/2022 E3), and
custom symmetric (ALN, NK, NM, NN, R0) tuples (e.g. 1I/'Oumuamua).

**Not supported**: the asymmetric-outgassing time offset `DT` (dropped with
a warning by adam-core's importers — solutions that estimate DT, such as
67P or 81P, retain km-scale model error through perihelion and should not
be treated as JPL-parity), thermophysical Yarkovsky (`AMRAT`/`RHO`) models,
and estimated (rather than fixed) g(r) shape parameters. Non-finite
A-values or present shape constants, and degenerate `ALN <= 0` / `R0 <= 0`
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

It reuses the existing propagation, nongrav, ephemeris/covariance, and
collision workload builders. Orbit determination remains exclusively an
adam-core product; adam-assist supplies its private batched propagation capsule
instead of provider-owned OD work units. Results include current public Python
timings, genuine Rust-owned `std::time::Instant` timings where available,
public/native overhead, and exact workload shapes. It does not require a frozen
Python environment or baseline timing cache, and all ASSIST workloads use
`max_processes=1`. Use `--quick` for a smoke run or select, for example,
`--domains nongrav ephemeris covariance --lanes tiny small`. Release CI runs
all 30 ASSIST product workloads with native timing required. After wheel
acceptance, release candidates also run all 54 live JPL Horizons propagation
and ephemeris accuracy cases.

The deterministic test suite uses reviewed, hash-pinned frozen outputs for
propagation, covariance, ephemeris, collision, typed mapping, and the minimal
Core-owned capsule integration fixture. Normal pytest and the current benchmark
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