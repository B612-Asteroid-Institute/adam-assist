# Registry-locked release tooling

Adam-core `0.8.0` is public and independently verified at release commit
`37fcd1c6f5d38d70335e56c3b095658a466d1f50`. The committed PDM and Cargo locks
resolve only the corresponding public PyPI wheels and crates.io archives, with
registry hashes/checksums and no path, Git, URL, patch, or replacement source.

## CPython 3.14 validation

One Python 3.11-3.14 CI matrix builds only the Assist wheel. Every row installs
the public binary Core 0.8.0 wheel, checks its embedded release identity, and
runs the same installed-wheel suite. The generic Rust lane consumes all six
Core 0.8.0 crates from the authentic Cargo lock and runs direct
`AssistPropagator`/`Propagator`, installed-kernel, and provider-provenance
runtime contracts on Rust 1.87. It compile/type-checks direct
`AssistShard`/`PropagatorShard` composition and trait bounds, while its
`core_owned_od` example compile/type-checks the standalone Rust composition
route. The Python capsule suite separately executes propagation, STM, lifetime,
and error contracts plus Core `run_iod`, `refine`, `run_od`, and
`evaluate_orbits` with the Assist provider.

The mandatory 16-wheel matrix includes Python 3.14. Python and Rust release
identities are aligned at 0.7.0 while retaining the public Core 0.8 line.
Release workflows are `workflow_dispatch`-only and require
`ADAM_ASSIST_RELEASES_ENABLED == 'true'`. The PyPI upload job also requires
`dry_run == false`. The contract's `present-and-version-aligned` value is
descriptive rather than a gate: actual fail-closed behavior comes from the
readiness variable, exact immutable `v0.7.0` tag checkout, candidate-run and
release-version checks, and registry immutability. Published 0.6.0 artifacts
remain immutable.

## Public Core release and lock validation

Verify the exact public release identity and both committed locks, then run
Cargo without overrides. `pdm lock --check` validates the committed lock; it
does not use the resolver pins itself. The dev extra pins `pdm==2.29.2`,
`hishel==1.1.6`, and `unearth==0.18.2` only so future `pdm lock` regeneration
uses the tested client stack: current Hishel is incompatible with PDM 2.29.2's
lock command.

```console
python migration/scripts/verify_source_pair_contract.py
pdm lock --check
python migration/scripts/verify_registry_locks.py \
  --core-version 0.8.0 --python-lock --rust-lock
python -m pip download --only-binary=:all: --no-deps \
  --dest /tmp/adam-core-0.8.0 adam-core==0.8.0
cargo test --manifest-path rust/adam_assist_rs/Cargo.toml \
  --locked --all-targets --features python
cargo test --manifest-path rust/adam_assist_rs/Cargo.toml \
  --locked --lib --no-default-features
cargo metadata --manifest-path rust/adam_assist_rs/Cargo.toml \
  --locked --format-version 1 > /tmp/adam-assist-cargo-metadata.json
python migration/scripts/verify_rust_core_line.py \
  --metadata /tmp/adam-assist-cargo-metadata.json --core-version 0.8.0
test -z "$(git status --porcelain)"
```

The release-candidate workflows repeat these checks before building one crate
archive and 16 platform/Python wheels. The wheel matrix checks out the exact
Core release only for immutable clean-room acceptance tooling and provenance;
it installs public Core artifacts and never rebuilds Core. Publication
workflows reverify the accepted artifacts and upload them without rebuilding.
Run both candidate workflows at the exact reviewed commit, create the immutable
`v0.7.0` tag only after both candidates pass, publish and verify the tested
Rust crate first, then pass its public checksum to the tested-wheel publisher.
Verify all 16 public wheel hashes before creating the GitHub Release from the
same tag. Rust 1.87 remains the MSRV for formatting, Clippy, tests,
documentation, packaging, and clean consumers.
