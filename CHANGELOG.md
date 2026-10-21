# Changelog

## [0.6.0] - Unreleased

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
- Made exact-source CI install the wheel's complete `dev` extra and verify
  `astropy`, frozen NAIF data versions, and installed-wheel provenance before
  tests or benchmarks. Release governance now semantically enforces registry
  locks, locked builds, and compatibility-gated publication/upload steps.

### Changed

- Reduced the Core integration surface to the private native
  `adam_core.propagator.v2` capsule: batched ASSIST propagation, immutable
  Cargo-SemVer identity, complete non-gravitational rows, row failures, and
  optional STMs. The dependency floor is Python `adam-core>=0.7.0,<0.8.0` and
  coherent Rust `0.7`, validated against accepted Core `0.7.0` commit
  `f0f1aec8b3c059f0cc4af054b5ebe8ed7760da19`. Release remains blocked until
  Core `0.7.0` is public and real registry locks can be generated.
- Removed ASSIST OD workloads from the current benchmark and public-surface
  classifications; OD algorithms, defaults, and reports are now exclusively
  Core-owned. Historical migration benchmarks and oracle evidence are retained.

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
