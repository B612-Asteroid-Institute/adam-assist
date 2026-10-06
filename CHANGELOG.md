# Changelog

## [0.4.1] - Unreleased

### Changed

- Declared compatibility with stable `adam-core` releases from `0.5.7` through
  the complete `0.5.x` line in both Python and Rust package metadata.
- Updated release-candidate and publication workflow defaults for the paired
  stable `adam-assist 0.4.1` line while keeping candidate locks resolved to
  immutable `adam-core 0.5.8` sources.

### Compatibility

- There are no adam-assist science or public-API behavior changes in this
  release; it is a packaging-only successor to `0.4.0`. Core `0.5.7` is the
  tested lower bound, and Core must use `0.6` for breaking contracts.

## [0.4.0] - 2026-09-02

- Published the stable native Python and Rust ASSIST backend paired with
  `adam-core 0.5.7`.
