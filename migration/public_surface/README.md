# adam-assist public-surface governance

Updated 2026-10-20 against frozen updated-upstream
`cb5bb14b5c1c6b27f43595d327f1a7b3f819e5c2`.

`manifest.json` is the complete current source inventory and disposition. The
older `adam_assist_0_3_10.json` remains the immutable published-wheel
compatibility authority; it is not the latest-upstream inventory.

The current surface preserves every useful frozen-upstream name and adds native
timing plus the private `adam_core.propagator.v2` capsule consumed by
adam-core algorithms. Its immutable table declares Cargo-SemVer identity,
thread safety, STM support, and complete non-gravitational-row support.
Orbit determination remains owned and exposed by adam-core; migration-era OD
provider methods are removed from ``ASSISTPropagator``. Public propagation,
sampled 6D/9D covariance, ephemeris generation, and collisions/impacts each
enter the native implementation once. Python retains signatures, quivr/Arrow
wrapping, static perturber data, published helper names, warnings, and errors.

Correctness evidence is classified explicitly:

- **bitwise** for IDs, ordering, set membership, schemas, metadata, and exact
  compatibility helpers;
- **tolerance-based** for deterministic numerical states, covariance,
  ephemerides, collision times, capsule conformance, and JPL Horizons regressions; and
- **statistical** for Monte Carlo covariance sampling/collapse because the
  updated-upstream NumPy and Rust-native RNG streams intentionally differ.

Performance artifacts report frozen updated-upstream Python, the current public
Rust-backed facade, and `std::time::Instant` native Rust separately. Kernel
files and the canonical `libassist-sys`/`librebound-sys` FFI are explicit
provider boundaries, not hidden Python computational fallbacks.
