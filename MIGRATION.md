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
against the exact reviewed Core `0.7.0` source commit
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

## Core 0.7 / ASSIST 0.6 change inventory

This section is the PR-level migration and validation inventory for the exact
reviewed-source pair. It describes source state only: it is not a release,
compatibility, publication, or registry claim.

### Lineage and replay accounting

The implementation starts from current adam-assist `main`
`a8ec6d22b422da055730ff7c0e076f51afe8b0ba`, not directly from the accepted
semantic candidate. The 12 accepted commits ending at
`00ec21e3888d737becc6b180d1cb7ea3d7524df3` were replayed as
`c5dd7a2..6ef5c5a`:

| Accepted change | Disposition on current main |
|---|---|
| Expose ASSIST to Core OD as a native external propagator | Retained, then narrowed to the propagation-only ABI-v2 table and safe `AssistExternalPropagator` owner. |
| Implement ASSIST STMs through `Propagator::propagate_stm` | Retained; six-state STMs, non-gravitational partial plumbing, covariance propagation, and row-failure classification remain Assist-owned propagation behavior. |
| Make ASSIST capsule-only for Core integration | Retained; Python supplies the private capsule and owns no OD behavior. |
| Correct the capsule-only migration | Retained. |
| Adopt Core propagator-v2 | Adapted to the append-only provenance suffix required by exact Core `0b49938e`. |
| Align Marsden defaults with Core v2 | Retained field-by-field for `ALN`, `NK`, `NM`, `NN`, and `R0`. |
| Set the Core 0.7 floor | Retained as Python `>=0.7.0,<0.8.0` and coherent Rust `0.7` requirements. |
| Harden source-pair and release gates | Retained and strengthened with exact-commit wheel and Rust lanes. |
| Enforce structural release policy | Retained. |
| Remove prerelease publication workflows | Retained; only read-only CI remains. |
| Close transition-policy tripwires | Retained. |
| Reject local Core dependency overrides | Retained in package metadata; CI-only `--config patch.crates-io` values point at the one exact reviewed Core checkout. |

Stable patch IDs match for 11 of the 12 replayed commits. The sole patch-ID
difference is the deleted `Cargo.lock` hunk in the Core-floor change: current
main had already advanced `spicekit` from `0.2.2`/`17c159ac…` to
`0.2.3`/`be636b2f…` in `8eba85d`. The lock remains intentionally absent;
exact-source validation resolves `spicekit 0.2.3` through Core. The replay
parent tree `6ef5c5a^{tree}` is exactly the accepted
`00ec21e^{tree}` (`4b1ec10b63fd95e3c99e6ac565b20fb841afd468`), so the
current-main lock correction is preserved without semantic drift. Every other
file and behavior already present at the current-main base remains inherited;
the replay does not reset, replace, or roll back later-main work.

### Propagation, science, and runtime inventory

Retained unchanged in behavior:

- ASSIST gravity and non-gravitational dynamics, IAS15 integration, explicit
  kernel paths, epoch handling, frame/origin transformations, collision
  detection, ephemeris generation, sampled covariance behavior, Rayon thread
  limits, and per-row propagation failures;
- state-transition matrices, non-gravitational partials, covariance transport,
  and null/inactive Marsden-row semantics;
- the public Python propagation, ephemeris, collision, pickling, positional
  signature, and helper/constant surface tracked by
  `migration/public_surface/manifest.json`;
- direct standalone Rust use of `AssistPropagator` through Core's propagation
  traits, independent of Python and independent of Core OD orchestration.

Added or adapted for the exact Core source:

- ABI-v2 provider metadata now includes content-derived IAS15 configuration
  and kernel-data fingerprints. Every IAS15 option is encoded by exact floating
  bit pattern or enum value; kernel identity includes semantic slot, byte size,
  and SHA-256, not a filesystem path.
- Explicit-path construction retains stable open handles and asks ASSIST to
  load through aliases to those handles. Kernel hashing is lazy, thread-safe,
  and cached per immutable `AssistData` instance. Every metadata acquisition
  verifies size, device, inode, modification time, and change time on both the
  retained handle and canonical path, so truncation, same-size rewrites,
  restored-mtime rewrites, and path replacement fail closed instead of
  returning stale provenance.
- `AssistExternalPropagator` owns the raw ABI-v2 table and exposes only a
  lifetime-bound validated borrow. Callers never own a context pointer or
  manage capsule/table destruction. Python's capsule and standalone Rust use
  the same `OwnedExternalPropagator<AssistPropagator>` implementation.
- `AssistData::new(Ephemeris)` remains valid for direct propagation but cannot
  claim unavailable kernel provenance; external-provider construction fails
  with the explicit `construct it from paths` error.

No dynamics formula, integrator default, state ordering, failure code,
propagation output, or ABI callback behavior is intentionally changed by the
provenance work. Bitwise propagation comparisons against the accepted tree are
the numerical guard.

### Core-owned OD surface removal and adapter boundary

ASSIST does not own IOD, least-squares fitting, rejection, linkage, profile or
resource selection, evaluation, or workflow orchestration. The old Python and
PyO3 `fit_least_squares`, `fit_least_squares_evaluated`, `od_fit`,
`initial_orbit_determination`, and `vallado_least_squares` routes and native
work units remain removed. There are no compatibility aliases.

Python only constructs the provider, returns the private
`adam_core.propagator.v2` capsule, and delegates. Core owns decoding,
validation, grouping, fitting, typed scientific failures, memberships,
composition, and results. Standalone Rust either calls propagation traits on
`AssistPropagator` directly or borrows `AssistExternalPropagator::external()`
into Core's `OdPropagator`; no Python, PyO3 callback, or Assist OD layer is
required. The example `rust/adam_assist_rs/examples/core_owned_od.rs` documents
that lifetime boundary.

### Tests, workflows, documentation, and metadata

Repository validation covers:

- direct Rust propagation contracts plus compiled standalone Core-owned OD;
- the external owner/lifetime and direct-trait/adapter boundaries;
- all malformed capsule suffix, size, ABI, reserved-word, capability,
  callback, name/version, and two-body-flag tamper classes;
- exact `run_od == refine(run_iod)` composition with and without provider
  Jacobians, deterministic repeats, evaluate parity, two-group ordering,
  typed failure isolation, mixed-epoch/mixed-non-grav batching, public-vs-ABI
  propagation, and STM finite-difference behavior with a real ASSIST provider;
- lazy fingerprint call count, concurrent first access, cache reuse,
  same-size/restored-mtime mutation, replacement, and truncation;
- default, all-feature, and no-default-feature `check`, warning-denied Clippy,
  tests, doctests, and warning-denied docs on Rust 1.87;
- Python 3.11–3.13 source-pair wheels, lint, formatting, import compilation,
  strict mypy, installed versions, wheel dependency floors, workflow policy,
  public-surface governance, and current propagation-only benchmarks;
- source-tree and packaged-crate consumers. The science fixture needed by
  crate unit tests is duplicated under `rust/adam_assist_rs/testdata`; a test
  pins it byte-for-byte to the historical migration artifact so packaged
  `--all-targets` does not depend on repository-external files.

The machine-readable source-pair state is `exact-reviewed-source`, while
`lock_state` and `registry_state` continue to say that authentic public Core
0.7 artifacts do not yet exist. CI patches all four Core crates only for the
exact-source validation checkout. Package manifests contain registry-version
requirements only and intentionally contain no lock, path, Git, or fabricated
checksum state.

### Accepted-test disposition

The accepted `00ec` capsule tests are accounted for as follows:

- retained or adapted: capsule-only routing (now including fingerprints and
  Core metadata), refinement STM (expanded to refinement plus evaluation),
  ordering/non-grav/row failures, partial Marsden defaults, custom/inactive STM
  behavior, capsule lifetime, Python-only rejection, and one-acquisition IOD;
- replaced: the distinct-IOD/refinement-capsules test is replaced by
  `run_od_uses_one_assist_capsule_for_both_stages`, because one immutable
  provider identity now serves both Core-owned stages;
- intentionally omitted:
  `test_core_refine_preserves_physical_and_custom_nongrav_metadata` and
  `test_core_refine_preserves_but_does_not_activate_source_only_nongrav`.
  Injecting a changed non-gravitational row into a canonical IOD handoff would
  invalidate Core's state fingerprint, so canonical `refine` correctly rejects
  that mutated handoff before propagation. Non-gravitational dynamics remain
  covered through direct traits, ABI batching, STM/partial tests, and fresh
  canonical provider workflows; ASSIST does not bypass Core's immutable
  handoff identity.

### Historical evidence and deliberate omissions

`migration/scripts/benchmark_assist_od.py` is removed because it called the
removed Assist-owned OD methods and could no longer be an active benchmark.
`migration/artifacts/assist_od_benchmark_2026-08-12.json` and the isolated
`migration/parity` request handlers remain frozen historical oracle evidence;
they are not current APIs, runnable current-provider benchmarks, or release
claims. Current CI benchmarks only propagation, ephemeris, covariance,
collision, non-gravitational dynamics, and the native adapter boundary.

No Windows or musllinux support is added; upstream ASSIST still requires POSIX
memory mapping. No Core or Assist publication, tag, registry lock, hosted
release qualification, or live-registry claim is included. Those operations
remain separate post-Core-publication work requiring owner authorization.

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
