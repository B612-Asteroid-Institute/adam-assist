# adam_assist

Pure-Rust ASSIST propagation and ephemeris orchestration for the public
[`adam_core`](https://crates.io/crates/adam_core) contracts.

This crate is the deliberate GPL-3.0 boundary around ASSIST and REBOUND. It
provides the same Rust backend used by the `adam-assist` Python package without
requiring Python, PyO3, NumPy, or a Python runtime. Linux and macOS are
supported; Windows is currently unavailable because upstream ASSIST uses POSIX
memory mapping through `libassist-sys`.

## Integration boundary

`adam_assist` implements adam-core's typed propagation contracts and provides
an owned external-provider view for Core's canonical orbit-determination
workflows. It owns ASSIST dynamics, STMs, and provider provenance, but no
orbit-determination, fitting, rejection, or linkage behavior. `Cargo.toml` is
the authority for compatible adam-core crate versions and feature selection.

## Kernel setup

The default `kernel-data` feature uses adam-core's deterministic resolution
chain: environment override, installed Python data package, local cache, then a
checksummed wheel fetch. Set `ADAM_CORE_KERNEL_DE440` and
`ADAM_CORE_KERNEL_SB441_N16` to use specific DE440 and SB441-n16 files.
`ADAM_CORE_KERNEL_OFFLINE=1` disables downloads. Regardless of which source
wins, `from_default_kernels` loads through retained handles, hashes those exact
bytes, and verifies the pinned file size and SHA-256 before returning.

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
SB441-n16 paths. Every path-owned explicit, default, Python, and raw
`adam_assist::Ephemeris::from_paths` constructor in one linked crate instance
shares one process-global lock at the lowest exported raw construction entry
point. There is no outer double lock; propagation and ephemeris queries do not
hold the lock and remain parallel. Direct `libassist_sys::Ephemeris`, separately
linked copies, and external C/FFI construction are outside the guarantee.
Callers that wrap a self-constructed `Ephemeris` with `AssistData::new` cannot
export kernel provenance. Disable the default resolver with
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

Kernel files are immutable package-managed resources and must remain unchanged
for the provider's lifetime. Construction retains the exact handles ASSIST
loaded and, under the parser lock, binds a small digest of the DAF file record,
comments, complete summary chain, mapped length, and retained file identity.
The first actual kernel use—direct propagation, shard execution, or
adapter/capsule metadata—checks that binding, hashes a stable snapshot, and
establishes trust in the full current bytes.

Ordinary detectable replacement, truncation, corruption, or changes between
construction and first use return an error; exact restoration on the original
mapping recovers. These checks are provenance consistency checks, not a
security boundary against hostile shared-mapping writers. Concurrent mutation
is unsupported, and legitimate kernel updates require a new provider.
Chebyshev data bytes not read during construction remain live through ASSIST's
mapping and are intentionally adopted by the first complete fingerprint when
the eager/mapping binding is intact.

Every call through this crate's public `Ephemeris::from_paths` wrapper shares
one process-global mutex with provider construction. The mutex covers only
upstream ephemeris construction and the provider's adjacent eager-region
observations; it deliberately does not serialize propagation or ephemeris
queries. Direct `libassist_sys::Ephemeris` construction, separately linked
copies of this crate/library, and external upstream ASSIST C/FFI callers are
outside this guarantee.

## Public Rust surface

- `AssistPropagator` implements adam-core's typed `Propagator` contract.
- `AssistExternalPropagator` safely owns the validated ABI-v2 view used by
  Core's canonical OD workflows.
- `AssistData` owns the shared ASSIST ephemeris and its content fingerprint.
- `assist_propagation` exposes lower-level orbit, nongravitational,
  covariance/STM, and pooling operations.
- Collision and ephemeris helpers preserve the Python package's scientific
  semantics.

`libassist-sys` and `librebound-sys` provide the versioned FFI and RAII layers.
The adam-core dependencies provide generic coordinate, propagation, SPICE,
and kernel-data contracts. The crate manifest is authoritative for dependency
requirements.

## Optional Python extension

The `python` and `extension-module` features build the internal
`adam_assist._native` module used by Python wheels. They are disabled for normal
Rust consumers.

## Validation

From a source checkout with declared dependencies resolved, run the standard
Rust quality lanes separately:

```console
cargo fmt --all -- --check
cargo check --all-targets
cargo clippy --all-targets -- -D warnings
cargo test --all-targets
cargo test --doc
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps
```

Repository CI additionally exercises the declared feature combinations,
packaged-crate consumers, and kernel-backed integration tests.
