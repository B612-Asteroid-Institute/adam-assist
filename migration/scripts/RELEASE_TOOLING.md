# Registry-locked release tooling

Adam-core `0.7.0` is public and independently verified at release commit
`a6478f2f280c266d7a5d1442997eb520fc20888f`. The committed PDM and Cargo locks
resolve only the corresponding public PyPI wheels and crates.io archives, with
registry hashes/checksums and no path, Git, URL, patch, or replacement source.

## CPython 3.14 validation

The Python 3.11-3.14 CI matrix uses public Core `a6478f2f` for Python
3.11-3.13. Its 3.14 row builds Python Core from merged commit `8dc01202` and
builds Assist with the registry-only Cargo lock. The same row validates a
direct Rust pairing of all six Core crates with
`migration/locks/adam-assist-core-8dc01202.Cargo.lock`.

The mandatory 16-wheel matrix includes Python 3.14, so a wheel publication run
requires matching Core artifacts and aligned Assist versions and dependencies.

Release workflows are `workflow_dispatch`-only and require
`ADAM_ASSIST_RELEASES_ENABLED == 'true'`. The PyPI upload job also requires
`dry_run == false`.

## Public Core release and lock validation

Use the exact public Core release checkout as the source/provenance authority,
then verify both committed locks and run Cargo without overrides:

```console
git clone --no-hardlinks https://github.com/B612-Asteroid-Institute/adam_core.git ../adam-core-a6478f2f
git -C ../adam-core-a6478f2f checkout --detach a6478f2f280c266d7a5d1442997eb520fc20888f
CORE="$(cd ../adam-core-a6478f2f && pwd)"
test "$(git -C "$CORE" rev-parse HEAD)" = a6478f2f280c266d7a5d1442997eb520fc20888f
test "$(git -C "$CORE" rev-parse 'v0.7.0^{commit}')" = a6478f2f280c266d7a5d1442997eb520fc20888f
python migration/scripts/verify_source_pair_contract.py --core-tree "$CORE"

pdm lock --check
python migration/scripts/verify_registry_locks.py \
  --core-version 0.7.0 --python-lock --rust-lock
cargo test --manifest-path rust/adam_assist_rs/Cargo.toml \
  --locked --all-targets --features python
cargo test --manifest-path rust/adam_assist_rs/Cargo.toml \
  --locked --lib --no-default-features
test -z "$(git status --porcelain)"
```

The release-candidate workflows repeat these checks before building one crate
archive and 16 platform/Python wheels. Publication workflows reverify the
accepted artifacts and upload them without rebuilding. Rust 1.87 remains the
MSRV for formatting, Clippy, tests, documentation, packaging, and clean
consumers.
