# Registry-locked release tooling

Adam-core `0.7.0` is public and independently verified at release commit
`a6478f2f280c266d7a5d1442997eb520fc20888f`. The committed PDM and Cargo locks
resolve only the corresponding public PyPI wheels and crates.io archives, with
registry hashes/checksums and no path, Git, URL, patch, or replacement source.

Release-candidate and publication workflows remain `workflow_dispatch`-only.
Every release job requires the bounded repository variable
`ADAM_ASSIST_RELEASES_ENABLED` to equal `true`; `publish-pypi` additionally
retains its `dry_run == false` condition. The variable is configured only for
the reviewed release window after environment protections and the exact final
candidate are accepted. Enabling it does not itself authorize a registry
upload, tag, or GitHub release.

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

The release-candidate workflows repeat these checks before building the one
crate archive and 12 platform/Python wheel artifacts. Publication workflows
accept only those successful exact-commit candidate runs, reverify checksums,
and upload the accepted bytes without rebuilding. Rust 1.87 remains the
authoritative MSRV for formatting, strict Clippy, tests, documentation,
packaging, and clean consumers.
