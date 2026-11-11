# Changelog

## [Unreleased]

## [0.7.0] - 2026-11-11

### Changed

- Added CPython 3.14 support and upgraded the optional native bindings to
  PyO3/NumPy 0.28.
- Moved the current source compatibility floor to public `adam-core>=0.8.0,<0.9.0`
  and coherent Rust `0.8` crates, with authentic registry-only locks for all 16
  Core wheels and all six Core crates.
- Expanded Python CI to one released-Core Python 3.11-3.14 matrix. Every row
  installs the public Core 0.8.0 wheel; generic Rust and capsule gates validate
  direct `Propagator`/`PropagatorShard` composition and cross-extension ABI-v2
  propagation, STM, lifetime, identity, capabilities, and error behavior.
- Expanded the four-platform release-candidate matrix from 12 to 16 wheels.
- Aligned the Python wheel and Rust crate release identities at `0.7.0` for
  exact-candidate promotion. This release does not change propagation,
  ephemeris, force-model, covariance, or other science/runtime algorithms.

## [0.6.0] - 2026-11-06

### Removed (breaking)

- Removed provider-owned `fit_least_squares`,
  `fit_least_squares_evaluated`, `od_fit`, `initial_orbit_determination`, and
  `vallado_least_squares` methods and their native work units. Use adam-core's
  nondeprecated `refine`, `run_iod`, and `run_od` entry points with an
  `ASSISTPropagator`; see `MIGRATION.md` for exact policy mappings.

### Fixed

- Matched adam-core's per-field Marsden default contract: every null
  `ALN`/`NK`/`NM`/`NN`/`R0` value on an active row independently takes the
  asteroid default, including propagation and fixed-parameter STM paths.
- Made release-pair CI install the wheel's complete `dev` extra and verify
  `astropy`, frozen NAIF data versions, and installed-wheel provenance before
  tests or benchmarks. Release-candidate and publication workflows remain
  manual-only, and every job is gated by the bounded
  `ADAM_ASSIST_RELEASES_ENABLED` repository variable.
- Pinned active CI by content and required-command digests and pinned the exact
  manually gated release-workflow inventory, including
  its full Python matrix, Rust source-pair gates, read-only execution boundary,
  and benchmark artifacts on every trigger. Python dependency-name checks now
  apply full PEP 503 normalization.
- Deferred the 765 MiB kernel content hash from provider construction to the
  first actual kernel use (direct propagation, shard execution, or provider
  metadata). Kernel files are immutable package-managed resources for the
  provider lifetime. Construction binds the small eagerly parsed DAF region set
  and retained mapping; first use verifies that binding and hashes the complete
  bytes. Ordinary detectable replacement, truncation, corruption, or
  construction-to-first-use changes return an error, and restoration on the
  original mapping recovers. Concurrent mutation is unsupported and these
  checks are not an adversarial filesystem security boundary. Chebyshev data
  bytes unread at construction remain live and are intentionally adopted by the
  first complete fingerprint when the eager/mapping binding is intact.
- Moved process-global ASSIST SPK-parser construction serialization into the
  exported raw `Ephemeris::from_paths` constructor, so raw and provider paths
  share one lock without outer double-locking while propagation remains
  parallel. Reworked stress coverage into a synchronized 64-constructor burst
  followed by direct constant/mass/segment queries, avoiding a full 765 MiB hash
  per constructor and making a lock-removal mutation fail.
- Expanded exact-source validation to default/all/no-default Rust checks,
  warning-denied Clippy, tests, doctests, docs, packaged-crate tests, and
  independent capsule tamper/composition/batching/STM probes. Packaged unit
  tests now carry their pinned science fixture inside the crate.

### Changed

- Reduced the Core integration surface to the private native
  `adam_core.propagator.v2` capsule: batched ASSIST propagation, immutable
  Cargo-SemVer identity, complete non-gravitational rows, row failures,
  optional STMs, and the append-only configuration/data fingerprint suffix.
  Standalone Rust callers receive the same provenance-complete provider through
  `AssistPropagator::external_propagator()`. The dependency floor is Python `adam-core>=0.7.0,<0.8.0` and
  coherent Rust `0.7`, validated against public Core `0.7.0` release commit
  `a6478f2f280c266d7a5d1442997eb520fc20888f`. Authentic PDM and Cargo locks
  resolve the 12 public Core wheels and all six public Core crates, respectively,
  without path, Git, URL, patch, or replacement sources. Reviewed release
  workflows remain manual-only, and every job is gated by the bounded
  `ADAM_ASSIST_RELEASES_ENABLED` repository variable; publication remains a
  separate authorized action.
- Removed ASSIST OD workloads from the current benchmark and public-surface
  classifications; OD algorithms, defaults, and reports are now exclusively
  Core-owned. The stale generator that called removed methods is retired;
  frozen benchmark output and isolated oracle evidence remain explicitly
  historical.

## [0.5.0] - 2026-10-16

### Changed

- Raised the compatible Core line to Python `adam-core>=0.6,<0.7` and Cargo
  `0.6`, preserving Rust 1.87 and all feature semantics.
- Finalized PDM and Cargo locks on public `adam-core 0.6.1` registry artifacts,
  including all 12 Python wheel hashes and all Core crate checksums.
- Updated release, clean-room, compatibility, and publication contracts for
  ASSIST `0.5.0`, immutable Core v0.6.1 acceptance tooling, same-wheel Python
  Core 0.6.0/0.6.1 validation, and coherent public Rust Core 0.6.1 resolution.
- Enforced exact PDM 2.29.2 for repository lock and CI behavior and rejected
  provisional Core sources, `zip`, and `zopfli`; kernel-data resolves
  `rawzip 0.5.2`.

### Compatibility

- There are no propagation, force-model, covariance, collision, ephemeris,
  numerical, public-API, or other science/runtime algorithm changes relative
  to `0.4.1`. This is a dependency/release-engineering minor release for the
  Core 0.6 compatibility floor.

## [0.4.1] - 2026-10-06

### Changed

- Declared compatibility with stable `adam-core` releases from `0.5.7` through
  the complete `0.5.x` line in both Python and Rust package metadata.
- Finalized PDM and Cargo locks on public registry artifacts for
  `adam-core 0.5.8`, including wheel hashes and crate checksums.
- Updated all 12 wheel lanes to run the once-built ASSIST wheel and matching
  public Core `0.5.8` wheel through pinned clean-room propagation, ephemeris,
  offline-kernel, and provenance checks, then test the identical ASSIST wheel
  and crate artifacts with public Core `0.5.7` and `0.5.8`.

### Compatibility

- There are no adam-assist science or public-API behavior changes in this
  release; it is a packaging-only successor to `0.4.0`. Core `0.5.7` is the
  tested lower bound, and Core must use `0.6` for breaking contracts.

## [0.4.0] - 2026-09-02

- Published the stable native Python and Rust ASSIST backend paired with
  `adam-core 0.5.7`.
