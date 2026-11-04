# adam_assist

Pure-Rust ASSIST propagation and ephemeris orchestration for the public
[`adam_core`](https://crates.io/crates/adam_core) contracts.

This crate is the deliberate GPL-3.0 boundary around ASSIST and REBOUND. It
provides the same Rust backend used by the `adam-assist` Python package without
requiring Python, PyO3, NumPy, or a Python runtime. Linux and macOS are
supported; Windows is currently unavailable because upstream ASSIST uses POSIX
memory mapping through `libassist-sys`.

## Integration status

The `0.6.0` source line requires the coherent adam-core `0.7` crate line and
its private propagator-v2 contract. It is source-pair validated against exact
Core commit `0b49938eb0be04214a8d07b206227d31b248394a` (version `0.7.0`).
Core `0.7.0` is not yet on crates.io, so the active registry lock is
intentionally absent and crate release remains blocked until publication.

## Kernel setup

The default `kernel-data` feature uses adam-core's deterministic resolution
chain: environment override, installed Python data package, local cache, then a
checksummed wheel fetch. Set `ADAM_CORE_KERNEL_DE440` and
`ADAM_CORE_KERNEL_SB441_N16` to use specific DE440 and SB441-n16 files.
`ADAM_CORE_KERNEL_OFFLINE=1` disables downloads. Regardless of which source
wins, `from_default_kernels` verifies the reviewed file size and SHA-256 before
ASSIST opens either ephemeris.

```no_run
use adam_assist::{AssistPropagator, AssistResult};

fn main() -> AssistResult<()> {
    let propagator = AssistPropagator::from_default_kernels()?;
    let _integrator = propagator.integrator();
    Ok(())
}
```

For explicit data ownership or offline deployments, call
`AssistData::from_paths` or `AssistPropagator::from_paths` with DE440 and
SB441-n16 paths. Disable the default resolver with
`default-features = false`.

Core's canonical standalone workflows accept `OdPropagator`, not arbitrary
predictor callbacks. Convert ASSIST to its owned, provenance-complete ABI-v2
view and keep that owner alive for the workflow call:

```no_run
use adam_assist::AssistPropagator;
use adam_core::od::{AssociatedObservations, OdBatchResult, OdConfig, OdPropagator};
use std::path::Path;

fn run(
    planets: &Path,
    asteroids: &Path,
    input: &AssociatedObservations,
    config: &OdConfig,
) -> Result<OdBatchResult, Box<dyn std::error::Error>> {
    let assist = AssistPropagator::from_paths(planets, asteroids)?;
    let external = assist.external_propagator()?;
    let provider = OdPropagator::external(external.external());
    Ok(adam_core::run_od(input, config, &provider)?)
}
```

The same provider works with Core-owned `run_iod`, `refine`, and
`evaluate_orbits`. ASSIST supplies propagation, non-gravitational dynamics,
STMs, and immutable configuration/data fingerprints only.

## Public Rust surface

- `AssistPropagator` implements adam-core's typed `Propagator` contract.
- `AssistExternalPropagator` safely owns the validated ABI-v2 view used by
  Core's canonical OD workflows.
- `AssistData` owns the shared ASSIST ephemeris and its content fingerprint.
- `assist_propagation` exposes lower-level orbit, nongravitational,
  covariance/STM, and pooling operations.
- Collision and ephemeris helpers preserve the Python package's reviewed
  scientific semantics.

`libassist-sys` and `librebound-sys` provide the versioned FFI and RAII layers.
The adam-core dependencies provide generic coordinate, propagation, SPICE,
and kernel-data contracts. The manifest declares the coherent Core `0.7` line; exact pre-publication
validation is pinned to the commit above.

## Optional Python extension

The `python` and `extension-module` features build the internal
`adam_assist._native` module used by Python wheels. They are disabled for normal
Rust consumers.

## Validation

Until Core `0.7.0` is public, run these gates from the repository's exact
source-pair checkout rather than inventing a registry lock. The root README
shows the complete four-crate path-patch command used by CI. That command runs
Rust 1.87 formatting, strict Clippy, all targets, tests, doctests, and this
compiled standalone example. Registry package verification remains blocked
until authentic Core artifacts exist.
