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

- `libassist-sys` and `librebound-sys` own reusable FFI and RAII bindings.
- `adam-assist` owns propagation/ephemeris orchestration, package semantics,
  and its GPL Python wheel.
- `adam-core` owns permissive generic contracts and cross-package integration.

The temporary identified snapshots under `rust/vendor` were removed after the
adam-core Rust crates were published. The `0.4.0rc2` package exact-pins the
public `adam_core_rs_coords`, `adam_core_rs_spice`, and test-only
`adam_core_rs_kernel_data` crates to `=0.1.0-rc.2`; its Python metadata
exact-pins `adam-core==0.5.6rc2`.

## Parity

`migration/parity` contains the two-runtime legacy oracle. Numerical reference
runs use pinned legacy adam-core and adam-assist 0.3.9. The complete published
API authority is the `adam-assist==0.3.10` wheel identified in
`migration/public_surface/adam_assist_0_3_10.json`; governance tests preserve
its four modules, root exports, module-level helpers/constants, class methods,
positional signatures, and defaults. Current Rust tests exercise the public
`adam_assist.ASSISTPropagator` import, never a private experimental package
name.

## Release candidate

The opt-in Python preview versions are:

```text
adam-core==0.5.6rc2
adam-assist==0.4.0rc2
```

Pip excludes these from ordinary stable resolution. The adam-assist RC metadata
must exact-pin the core RC before publication. Its 12 native wheels contain the
Python compatibility veneer and compiled `adam_assist._native` extension. Each
public propagation, ephemeris, covariance, or collision operation crosses into
adam-assist once; adam-core generic algorithms and ASSIST orchestration then
compose Rust-to-Rust inside that extension.
