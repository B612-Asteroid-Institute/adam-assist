# adam_assist_rs

GPL-3.0 backend crate implementing adam-assist propagation and ephemeris semantics over permissive adam-core contracts.

## Ownership

- `libassist-sys` and `librebound-sys` provide versioned FFI and safe RAII wrappers.
- `adam_assist_rs` owns orbit/frame normalization, propagation, same-epoch batching, simulation pools, STM/covariance transport, collision handling, ephemeris assembly, OD/IOD work units, Rayon scheduling, and package error semantics.
- Exact `=0.1.0-rc.1` `adam_core_rs_*` dependencies provide the published permissive generic contracts.

The crate is packaged as `adam_assist._native`. It is not part of adam-core’s permissive Rust workspace because ASSIST/REBOUND and this adapter form the deliberate GPL boundary.

## Validation

```bash
cargo fmt
cargo clippy --all-targets -- -D warnings
cargo test
```

The four ignored kernel-backed tests resolve DE440 and SB441-n16 through the published `adam_core_rs_kernel_data` policy. Historical `ADAM_CORE_RS_ASSIST_PLANETS_PATH` and `ADAM_CORE_RS_ASSIST_ASTEROIDS_PATH` variables remain optional highest-priority overrides.

Python wheels are built by the repository’s Maturin configuration. Public parity and performance must use identical kernels and the dedicated legacy runtime before making comparative claims.
