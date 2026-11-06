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
`0.5.0` artifacts retain their historical methods; the `0.6` line is not
`0.5`-compatible.

This line exports only `adam_core.propagator.v2`, declares Python
`adam-core>=0.7.0,<0.8.0` and coherent Rust `0.7` requirements, and is validated
against public Core `0.7.0` release commit
`a6478f2f280c266d7a5d1442997eb520fc20888f`. Authentic PDM and Cargo locks
resolve the public Core wheels and crates with their registry hashes/checksums;
release automation remains manual and bounded by the repository readiness gate.

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
public Core release pair. It records Core 0.7.0 registry identity and authentic
locks; it is not an Assist 0.6.0 artifact-qualification or publication claim.

### Lineage and replay accounting

The public `assist_propagation` Rust module descends from historical
`assist-rs` revision `f66517aa`, specifically
`src/{propagate.rs, coordinates.rs, orbit.rs, assist_data.rs}`. Its obliquity
constants remain bit-identical to adam-core's
`np.cos/np.sin(Constants.OBLIQUITY)` values. This file-move provenance belongs
in migration history rather than the stable module Rustdoc.

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
| Adopt Core propagator-v2 | Adapted to the append-only provenance suffix required by public Core `a6478f2f`. |
| Align Marsden defaults with Core v2 | Retained field-by-field for `ALN`, `NK`, `NM`, `NN`, and `R0`. |
| Set the Core 0.7 floor | Retained as Python `>=0.7.0,<0.8.0` and coherent Rust `0.7` requirements. |
| Harden source-pair and release gates | Retained and strengthened with exact-commit wheel and Rust lanes. |
| Enforce structural release policy | Retained. |
| Remove prerelease publication workflows | Superseded by owner policy: the reviewed workflows remain manual-only and every job is controlled by the bounded `ADAM_ASSIST_RELEASES_ENABLED` repository variable. |
| Close transition-policy tripwires | Retained. |
| Reject local Core dependency overrides | Retained in package metadata and authentic locks; CI resolves Core only from PyPI/crates.io while separately verifying the exact public source commit. |

Stable patch IDs match for 11 of the 12 replayed commits. The sole patch-ID
difference was the initially deleted `Cargo.lock` hunk in the Core-floor
change: current main had already advanced `spicekit` from
`0.2.2`/`17c159ac…` to `0.2.3`/`be636b2f…` in `8eba85d`. After Core 0.7.0
publication, the authentic registry-only lock was generated with that resolved
line. The replay parent tree `6ef5c5a^{tree}` is exactly the accepted
`00ec21e^{tree}` (`4b1ec10b63fd95e3c99e6ac565b20fb841afd468`), so the
current-main lock correction is preserved without semantic drift. Except for
the deliberate surface/workflow/test removals itemized below, every other file
and behavior already present at the current-main base remains inherited; the
replay does not reset, replace, or roll back later-main work.

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
  load through aliases to those handles. The exported raw
  `Ephemeris::from_paths`, explicit/default provider, and Python constructors
  share one process-global lock at the lowest construction entry point;
  propagation never holds it and remains parallel. Kernel files are immutable
  package-managed resources for the provider lifetime. Construction binds the
  DAF file record, comments, complete summary chain, mapped length, and retained
  identity around parsing; first use verifies that binding and hashes the
  complete current bytes. Ordinary detectable replacement, truncation,
  corruption, or construction-to-first-use changes return an error, and exact
  restoration on the original mapping recovers. These checks are not a security
  boundary: concurrent mutation is unsupported, and legitimate updates require
  a new provider. Chebyshev data bytes unread during construction remain live
  through the mapping and are intentionally adopted by the first full
  fingerprint when the eager/mapping binding is intact.
- `AssistExternalPropagator` owns the raw ABI-v2 table and exposes only a
  lifetime-bound validated borrow. Callers never own a context pointer or
  manage capsule/table destruction. Python's capsule and standalone Rust use
  the same `OwnedExternalPropagator<AssistPropagator>` implementation. A table
  already created before ordinary detectable on-disk changes keeps the
  immutable fingerprint of the exact retained handle ASSIST loaded, while each
  callback revalidates the handle/path and rejects those changes before
  dynamics; asking its owning propagator for a new adapter fails the same
  verification. Concurrent mutation remains outside the supported contract.
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
- lazy fingerprint call count, concurrent first access, cache reuse, benign
  hard-link/chmod/touch changes before first use, retry after transient initial
  failure and first-hash metadata races, metadata-only post-trust refresh,
  same-size/restored-mtime tamper rejection and restoration recovery, and
  retained capsule/new-adapter tamper behavior;
- load-time binding of eagerly parsed constants/summary records, ordinary
  construction-time changes, truncation/SIGBUS prevention, constant-byte
  tamper, inode replacement, no-cache failure, exact restoration, and Python
  entry-point behavior;
- repeated high-contention mixed raw/provider construction (64 constructors in
  one synchronized burst), direct finite constant/mass/segment queries at two
  epochs, a long-horizon finite massive-body control, and bitwise agreement with
  a sequential reference; a lock-removal mutation must fail this stress;
- finite actual, expected, residual, capsule state, STM, and Marsden-partial
  values before every live propagation or ephemeris maximum/tolerance,
  including an all-NaN capsule control that must fail;
- default, all-feature, and no-default-feature `check`, warning-denied Clippy,
  tests, doctests, and warning-denied docs on Rust 1.87;
- Python 3.11–3.13 source-pair wheels, lint, formatting, import compilation,
  strict mypy, installed versions, wheel dependency floors, workflow policy,
  public-surface governance, and current propagation-only benchmarks;
- source-tree and packaged-crate consumers. The science fixture needed by
  crate unit tests is duplicated under `rust/adam_assist_rs/testdata`; a test
  pins it byte-for-byte to the historical migration artifact so packaged
  `--all-targets` does not depend on repository-external files.

The machine-readable source-pair state is `exact-public-release`; its
`lock_state` and `registry_state` require public Core 0.7.0 and authentic
registry-only locks. CI verifies release commit `a6478f2f…`, builds the matching
Core wheel, and consumes all Rust Core crates from the exact checksummed Cargo
lock. Package manifests and locks contain no path, Git, URL, patch, replacement,
or fabricated checksum state.

### Upstream parser follow-up

The crate lock is a containment measure for the non-reentrant global state in
upstream ASSIST's SPK parser. A separate upstream change should make parser
construction reentrant (or expose an upstream-owned process-global guard), add
an upstream mixed-constructor race test, and then permit this wrapper lock to be
removed only after the lock-removal mutation passes there. Until that work is
released and reviewed, all crate-owned raw and provider construction remains on
the single wrapper lock. Direct `libassist_sys`, separately linked copies, and
external C/FFI construction remain explicitly outside that guarantee.

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

### Current-main test disposition

The comparison from current-main base `a8ec6d22` to this propagation-only
source line reports these 19 removed test node IDs. Each removal is deliberate
and accounted for below; no test disappeared merely because it was
inconvenient:

| Removed current-main test | Disposition and current authority |
|---|---|
| `tests/rust/test_adam_assist_rust_od.py::test_fit_least_squares_evaluated_matches_composed_crossings` | Removed with Assist-owned fitting; Core `refine`/`evaluate_orbits` composition is covered by `tests/rust/test_provider_capsule_od.py`. |
| `tests/rust/test_adam_assist_rust_od.py::test_initial_orbit_determination_fused_batch_and_timing` | Removed with Assist-owned IOD; canonical Core `run_iod` batching and one capsule acquisition are covered by `test_provider_capsule_od.py`. |
| `tests/rust/test_adam_assist_rust_od.py::test_native_od_timing_lanes` | Removed with the obsolete Assist OD benchmark lane; the frozen artifact is historical only and current benchmarks are propagation-only. |
| `tests/rust/test_adam_assist_rust_od.py::test_od_fit_below_min_obs_returns_empty` | Removed with `od_fit`; Core owns typed per-group insufficient-observation behavior and the provider test covers failure isolation. |
| `tests/rust/test_adam_assist_rust_od.py::test_od_fit_matches_frozen_legacy_solution` | Removed with `od_fit`; frozen legacy parity remains evidence while canonical Core `run_od` is exercised with the real capsule. |
| `tests/rust/test_adam_assist_rust_od.py::test_public_fit_least_squares_dispatches_to_native` | Removed because that public Assist method no longer exists; Python delegates to Core and the capsule contains propagation callbacks only. |
| `tests/rust/test_adam_assist_rust_od.py::test_rust_od_recovers_truth` | Replaced by real-provider Core `run_iod`/`run_od`/`refine` truth, composition, determinism, and evaluation probes in `test_provider_capsule_od.py`. |
| `tests/rust/test_adam_assist_rust_od.py::test_vallado_least_squares_matches_frozen_parity` | Removed with the noncanonical Vallado hook; the frozen oracle remains migration evidence and Core `refine` is the documented replacement. |
| `tests/test_ci_contract.py::test_every_rust_workflow_uses_the_msrv_toolchain_and_components` | Adapted into the active read-only CI policy plus default/all/no-default Rust 1.87 matrix checks; preserved release workflows remain controlled by bounded release-variable authorization. |
| `tests/test_ci_contract.py::test_normal_ci_runs_direct_rust_quality_and_current_only_benchmark` | Adapted into `test_exact_workflow_inventory_and_gated_release_jobs` and `test_critical_ci_commands_and_benchmark_evidence_are_pinned`. |
| `tests/test_ci_contract.py::test_release_matrix_builds_once_and_clean_room_accepts_registry_wheels` | Restored through the manual 12-lane registry-wheel acceptance workflow, now backed by authentic Core 0.7 locks and exact release tooling. |
| `tests/test_ci_contract.py::test_rust_crate_workflows_package_once_and_publish_tested_bytes` | Restored through the manual build-once crate candidate/publisher workflow with the authentic Core lock; publication still requires separate authorization. |
| `tests/test_core_compatibility_gates.py::test_same_wheel_reports_accept_compatible_core_06_outputs` | Replaced by `test_same_wheel_reports_accept_compatible_core_07_outputs`; Core 0.6 is intentionally incompatible with ABI-v2 provenance. |
| `tests/test_dependency_metadata.py::test_cargo_lock_uses_public_core_registry_artifacts` | Restored as registry-lock verification for all six checksummed Core 0.7 crates. |
| `tests/test_dependency_metadata.py::test_current_benchmark_ci_covers_product_and_core_od_backend_workloads` | Adapted to `test_current_benchmark_ci_covers_provider_product_workloads`; Assist-owned OD workloads are retired. |
| `tests/test_dependency_metadata.py::test_python_lock_uses_public_core_registry_artifacts` | Restored as registry-lock verification for the 12 hashed Core 0.7 wheels. |
| `tests/test_dependency_metadata.py::test_stable_dependencies_declare_core_06_compatibility` | Replaced by `test_dependencies_declare_core_070_floor` and exact source-pair metadata assertions. |
| `tests/test_nongrav.py::test_partial_constants_are_rejected_for_active_force` | Adapted to reviewed field-wise asteroid defaults in `test_partial_constants_fill_asteroid_defaults_per_field` and capsule/default-equivalence tests. |
| `tests/test_public_surface_compatibility.py::test_stable_version_module_is_preserved` | Adapted to `test_next_source_version_module_is_preserved`, preserving the module while asserting the breaking-line version `0.6.0`. |

### Historical evidence and deliberate omissions

`migration/scripts/benchmark_assist_od.py` is removed because it called the
removed Assist-owned OD methods and could no longer be an active benchmark.
`migration/artifacts/assist_od_benchmark_2026-08-12.json` and the isolated
`migration/parity` request handlers remain frozen historical oracle evidence;
they are not current APIs, runnable current-provider benchmarks, or release
claims. Current CI benchmarks only propagation, ephemeris, covariance,
collision, non-gravitational dynamics, and the native adapter boundary.

No Windows or musllinux support is added; upstream ASSIST still requires POSIX
memory mapping. Core 0.7.0 publication and the authentic lock transition are
recorded here; Assist tagging, artifact qualification, and publication remain
separate owner-authorized release operations.

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
crates used trusted publishing. Core `0.7.0` is now public and authentic
registry-only locks are committed. Release-candidate and publication workflows
remain workflow-dispatch-only, and every job requires the bounded repository
variable `ADAM_ASSIST_RELEASES_ENABLED == 'true'`. The variable may be enabled
only for the reviewed release window after environment protections are verified;
its presence does not replace exact-artifact gates or separate publication
authorization. The scripts under `migration/scripts` remain gated release
tooling, not automatic publication entry points. Rust 1.87 remains authoritative for formatting,
strict Clippy, tests, documentation, packaging, wheels, and clean consumer
resolution. The adam-assist RC metadata
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
