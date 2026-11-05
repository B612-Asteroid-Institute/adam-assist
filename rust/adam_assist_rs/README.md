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

#[cfg(feature = "kernel-data")]
fn main() -> AssistResult<()> {
    let propagator = AssistPropagator::from_default_kernels()?;
    let _integrator = propagator.integrator();
    Ok(())
}

#[cfg(not(feature = "kernel-data"))]
fn main() {}
```

For explicit data ownership or offline deployments, call
`AssistData::from_paths` or `AssistPropagator::from_paths` with DE440 and
SB441-n16 paths. Every path-owned explicit, default, and Python constructor
shares one process-global lock around upstream ASSIST/SPICE ephemeris parsing;
propagation does not hold that lock and remains parallel. Callers that instead
wrap a self-constructed `Ephemeris` with `AssistData::new` are responsible for
serializing that raw upstream construction and cannot export kernel
provenance. Disable the default resolver with `default-features = false`.

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

A created adapter retains immutable metadata for the exact kernel handles that
ASSIST loaded. If filesystem metadata later changes, the owner rehashes the
affected retained handle (and a replaced path) under a per-provider lock.
Byte-identical changes after the first fingerprint refresh the metadata
snapshot; changed bytes fail closed while preserving the last trusted state.
A pre-fingerprint metadata change has no trusted digest to compare and requires
reconstructing the provider after restoring the file. An already-created adapter
continues to report its original fingerprint, but its propagation callbacks
revalidate provenance and reject changed bytes before dynamics. Creating a new
adapter from that owner also rejects the change. Restoring the exact bytes
allows a fresh verification to recover.

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
and kernel-data contracts. The manifest declares the coherent Core `0.7` line; exact reviewed-source
validation is pinned to the commit above; public registry locking remains deferred.

## Optional Python extension

The `python` and `extension-module` features build the internal
`adam_assist._native` module used by Python wheels. They are disabled for normal
Rust consumers.

## Validation

Until Core `0.7.0` is public, validate against a detached checkout of exact
reviewed Core commit `0b49938eb0be04214a8d07b206227d31b248394a` rather than
inventing registry state. The root README shows the four-crate path patch for
one all-feature `cargo test --all-targets` lane. That command does **not** run
formatting, Clippy, doctests, or docs; CI invokes separate Rust 1.87 commands
for default, all-feature, and no-default-feature `check`, strict Clippy, tests,
doctests, and warning-denied docs. Registry package verification remains
blocked until authentic Core artifacts exist.
