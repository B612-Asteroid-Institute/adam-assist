# Rust backend migration

`adam-assist` now owns the GPL Rust backend that was incubated as
`rust/adam_core_rs_assist` in the adam-core migration repository. The compiled
extension is packaged as `adam_assist._native`; the stable public import remains:

```python
from adam_assist import ASSISTPropagator
```

The Python class is a compatibility veneer. Public propagation, covariance
sampling/collapse, ephemeris generation, collision detection, and ASSIST-backed
least-squares operations each enter the Rust backend once; local parallelism is
Rayon-owned. `max_processes` is retained as the compatible thread-limit control.

## Ownership boundary

- `assist-rs` owns reusable low-level ASSIST integration primitives.
- `adam-assist` owns package-level ASSIST semantics and its GPL Python wheel.
- `adam-core` owns permissive generic contracts and cross-package integration.

Until adam-core's Rust crates are published, identified snapshots are carried
under `rust/vendor`; see `rust/vendor/README.md`.

## Parity

`migration/parity` contains the two-runtime legacy oracle. The reference runtime
must contain pinned legacy adam-core and adam-assist 0.3.9. Current Rust tests
exercise the public `adam_assist.ASSISTPropagator` import, never a private
experimental package name.
