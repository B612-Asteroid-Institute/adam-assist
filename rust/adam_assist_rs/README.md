# adam_assist

Pure-Rust ASSIST propagation and ephemeris orchestration for the public
[`adam_core`](https://crates.io/crates/adam_core) contracts.

This crate is the deliberate GPL-3.0 boundary around ASSIST and REBOUND. It
provides the same Rust backend used by the `adam-assist` Python package without
requiring Python, PyO3, NumPy, or a Python runtime. Linux and macOS are
supported; Windows is currently unavailable because upstream ASSIST uses POSIX
memory mapping through `libassist-sys`.

## Installation

```toml
[dependencies]
adam_core = "=0.1.0-rc.4"
adam_assist = "=0.4.0-rc.6"
```

## Kernel setup

The default `kernel-data` feature uses adam-core's deterministic resolution
chain: environment override, installed Python data package, local cache, then a
checksummed wheel fetch. Set both `ADAM_CORE_RS_ASSIST_PLANETS_PATH` and
`ADAM_CORE_RS_ASSIST_ASTEROIDS_PATH` to use specific DE440 and SB441-n16
files. `ADAM_CORE_KERNEL_OFFLINE=1` disables downloads.

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

## Public Rust surface

- `AssistPropagator` implements adam-core's typed `Propagator` contract.
- `AssistData` owns the shared ASSIST ephemeris.
- `assist_propagation` exposes lower-level orbit, nongravitational,
  covariance/STM, and pooling operations.
- Collision and ephemeris helpers preserve the Python package's reviewed
  scientific semantics.

`libassist-sys` and `librebound-sys` provide the versioned FFI and RAII layers.
The exact `=0.1.0-rc.4` adam-core crate dependencies provide generic coordinate,
propagation, SPICE, and kernel-data contracts.

## Optional Python extension

The `python` and `extension-module` features build the internal
`adam_assist._native` module used by Python wheels. They are disabled for normal
Rust consumers.

## Validation

```bash
cargo +1.87.0 fmt --check
cargo +1.87.0 clippy --all-targets --all-features -- -D warnings
cargo +1.87.0 test --locked
cargo +1.87.0 package --locked
```
