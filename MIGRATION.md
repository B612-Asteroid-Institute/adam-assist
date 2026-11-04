# Rust backend migration

`adam-assist` now owns the GPL Rust backend that was incubated as
`rust/adam_core_rs_assist` in the adam-core migration repository. The compiled
extension is packaged as `adam_assist._native`; the stable public import remains:

```python
from adam_assist import ASSISTPropagator
```

The Python class is a compatibility veneer. Public propagation, covariance
sampling/collapse, ephemeris generation, and collision detection each enter the
Rust backend once; local parallelism is Rayon-owned. `max_processes` is retained
as the compatible thread-limit control. Orbit determination is owned by
`adam-core`, which consumes only ASSIST's propagation-only ABI-v2 provider. In
Python that provider is transported by the private native capsule; standalone
Rust borrows the same validated table from `AssistExternalPropagator`.

## Ownership boundary

- `libassist-sys` and `librebound-sys` own reusable FFI and RAII bindings.
- `adam-assist` owns ASSIST dynamics, propagation/ephemeris/collision product
  behavior, package semantics, its GPL `adam_assist` Rust crate, and its GPL
  Python wheel.
- `adam-core` owns permissive generic contracts, orbit determination, fitting,
  rejection, linkage orchestration, and cross-package integration.
- The ABI-v2 provider is the only native Core↔ASSIST execution capability; the
  private Python `propagator_capsule` transports that table. Both routes carry
  propagation, STM, and provenance metadata, never OD algorithms.

The temporary identified snapshots under `rust/vendor` were removed after the
adam-core Rust crates were published. The stable `0.4.0` Rust crate exact-pins
public `adam_core_rs_coords`, `adam_core_rs_spice`, and optional default
`adam_core_rs_kernel_data` dependencies to `=0.5.7`; its Python metadata
exact-pins `adam-core==0.5.7`.


## Orbit-determination API migration

This breaking removal starts the adam-assist `0.6.0` source line. Released
`0.5.0` artifacts retain their historical methods; this candidate does not
report itself as `0.5.0`-compatible.

This line exports only `adam_core.propagator.v2`, declares Python
`adam-core>=0.7.0,<0.8.0` and coherent Rust `0.7` requirements, and is validated
against the exact final Core `0.7.0` candidate commit
`0b49938eb0be04214a8d07b206227d31b248394a`. Core `0.7.0` is not yet on the
registries: active locks are intentionally absent and release/publication stays
blocked until public artifacts exist and their real checksums can be locked.

The migration-era methods below were provider implementation hooks, not an
adam-assist OD product, and are removed from `ASSISTPropagator`:

| Removed adam-assist method | Canonical adam-core replacement |
|---|---|
| `fit_least_squares`, `fit_least_squares_evaluated` | Persist a canonical IOD handoff, then call `refine(iod_orbits, iod_members, source_catalog, ASSISTPropagator(), config.for_refinement())` |
| `od_fit` | `run_od(source_catalog, associations, ASSISTPropagator(), config)` |
| `initial_orbit_determination` | `run_iod(source_catalog, associations, ASSISTPropagator(), config.for_iod())` |
| `vallado_least_squares` | No exact Vallado-trace replacement; use canonical `refine(...)` with an explicit `RefinementRunConfig` and migrate result handling |

Every canonical Python workflow returns exactly `(FittedOrbits,
FittedOrbitMembers, OrbitDeterminationFailures)`. Explicit `Associations`
own grouping, and the self-contained stage configuration owns fitting,
rejection, resources, and the observation pipeline. There are no provider
callbacks, hidden model defaults, or ASSIST-owned linkage loops.

Pass one `ASSISTPropagator` instance as the propagation-only provider. Core
acquires its private capsule once after static preflight and owns every
candidate, solver, rejection, and grouping decision. Standalone Rust callers
use `AssistPropagator::external_propagator()`, borrow its validated external
view into `adam_core::od::OdPropagator::external`, and call the same Core-owned
`run_iod`, `run_od`, `refine`, or `evaluate_orbits` functions.

## Parity

`migration/parity` contains the two-runtime legacy oracle. Numerical reference
runs use pinned legacy adam-core and adam-assist 0.3.9. Historical OD/Vallado
oracle code and benchmark evidence remain immutable migration evidence; they
are not active provider APIs or current benchmark workloads. The complete
published API authority is the `adam-assist==0.3.10` wheel identified in
`migration/public_surface/adam_assist_0_3_10.json`; governance tests preserve
its four modules, root exports, module-level helpers/constants, class methods,
positional signatures, and defaults. Current Rust tests exercise the public
`adam_assist.ASSISTPropagator` import, never a private experimental package
name.

## Release candidate

The opt-in preview versions are:

```text
crates.io: adam_core = "=0.1.0-rc.5"
crates.io: adam_assist = "=0.4.0-rc.7"
PyPI: adam-core==0.5.6rc6
PyPI: adam-assist==0.4.0rc7
```

Cargo and pip exclude these from ordinary stable resolution. Historical public
crates used trusted publishing, but the current Core `0.7.0` transition has no
release-candidate or publication workflows at all. Only read-only exact-source
CI exists until Core is public and authentic PyPI/Cargo locks can be generated.
The scripts retained under `migration/scripts` are inert local post-publication
tooling, not authorized CI entry points. Locks and release automation will be
added together only in a separately reviewed, explicitly approved commit. Rust
1.87 remains authoritative for formatting, strict Clippy, tests, documentation,
packaging, wheels, and clean consumer resolution. The adam-assist RC metadata
must exact-pin the core RC before publication. Its 12 native wheels contain the
Python compatibility veneer and compiled `adam_assist._native` extension. Each
public propagation, ephemeris, covariance, or collision operation crosses into
adam-assist once; adam-core generic algorithms and ASSIST orchestration then
compose Rust-to-Rust inside that extension.

## Stable 0.4.0 historical release provenance

The first stable line was Python/Rust `adam-assist 0.4.0`, exact-pinned to
Python and public Rust `adam-core 0.5.7`. The preliminary stable manifest commit
paired wheel validation with accepted Core wheels after the Core crates were
public. After Core wheels became public, the final stable commit refreshed the
frozen PDM lock normally. Both Cargo and PDM locks therefore resolve public Core
`0.5.7` artifacts without temporary path patches or index overrides.

Release scripts distinguish preview and stable channels. Stable publication
targets protected production environments named `crates-io` and `pypi`; the
matching GitHub environments and registry trusted publishers must be configured
before dispatch. Both publishers inspect existing registry state first and can
resume a partial upload only when already-public files are unyanked and
checksum-identical to the accepted artifacts. Tags, environment changes, and
registry uploads remain separate human approval boundaries.

## Stable 0.5.0 Core 0.6 compatibility

The science-code-unchanged `adam-assist 0.5.0` successor raises its compatible
Core floor to the `0.6` line. Python metadata declares
`adam-core>=0.6,<0.7`; Cargo manifests use `0.6` requirements while preserving
the optional/default-feature split. The release-authoritative PDM and Cargo
locks select public Core `0.6.1` artifacts with registry hashes and checksums.
They contain no Git/path Core source, provisional patch, `zip`, or `zopfli`;
kernel-data resolves exact `rawzip 0.5.2` under Rust 1.87.

Each wheel lane builds ASSIST once from those public locks, runs the exact wheel
with public Core `0.6.1` through immutable Core v0.6.1 clean-room tooling, then
checks the same ASSIST wheel with public Python Core `0.6.0` and `0.6.1`.
The packaged Rust crate and fresh no-lock consumers resolve one coherent public
Core `0.6.1` graph. Core `0.6.0` is not used as an exact fresh Rust lane because
its immutable kernel-data manifest references the now wholly-yanked `zip 2.x`
line; Core `0.6.1` is the compatible registry repair.

No propagation, force model, covariance, collision, ephemeris, numerical,
public-API, or science-fixture behavior changes in this release. Build-once
artifact provenance and separate merge, tag, protected-environment, crates.io,
and PyPI approval boundaries remain unchanged.

## Stable 0.4.1 Core 0.5 compatibility

The packaging-only `adam-assist 0.4.1` successor declares Python
`adam-core>=0.5.7,<0.6` and Rust `adam_core* >=0.5.7, <0.6.0`. Core `0.5.7`
is the tested lower bound, Core `0.5.8` is the current reproducible lock
selection, and future compatible `0.5.x` releases are intentionally accepted.
Breaking Python, Arrow/schema, or Rust contracts require a Core `0.6` release.
There is no adam-assist science or public-API behavior change.

Both PDM and Cargo locks resolve Core `0.5.8` exclusively from the public PyPI
and crates.io registries, including wheel hashes and crate checksums. Release
CI builds each adam-assist wheel once from those registry-only locks. Every
platform/Python lane pairs those bytes with the matching public Core `0.5.8`
wheel in Core's immutable clean-room driver, which verifies installed-artifact
provenance, forward/backward and same-epoch propagation, observer ephemeris,
and offline embedded-kernel policy. CI then tests the identical ASSIST wheel
bytes with public Python Core `0.5.7` and `0.5.8`. Pure-Rust consumers likewise
compile the one packaged crate against both exact public Core lines and reject
a graph containing duplicate or mixed Core crate versions.

Historical publication automation verified the registry-only locks and rejected
Git/path sources or provisional patches. Passing final artifact acceptance does not
authorize a tag or registry upload; those remain separate approval boundaries.
