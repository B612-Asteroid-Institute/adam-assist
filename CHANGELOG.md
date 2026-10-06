# Changelog

## [0.4.1] - 2026-10-06

### Changed

- Declared compatibility with stable `adam-core` releases from `0.5.7` through
  the complete `0.5.x` line in both Python and Rust package metadata.
- Finalized PDM and Cargo locks on public registry artifacts for
  `adam-core 0.5.8`, including wheel hashes and crate checksums.
- Updated release acceptance to build ASSIST once and test the identical wheel
  and crate artifacts with public Core `0.5.7` and `0.5.8`.

### Compatibility

- There are no adam-assist science or public-API behavior changes in this
  release; it is a packaging-only successor to `0.4.0`. Core `0.5.7` is the
  tested lower bound, and Core must use `0.6` for breaking contracts.

## [0.4.0] - 2026-09-02

- Published the stable native Python and Rust ASSIST backend paired with
  `adam-core 0.5.7`.
