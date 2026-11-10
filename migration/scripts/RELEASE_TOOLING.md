# Registry-locked release tooling

Adam-core `0.7.0` is public and independently verified at release commit
`a6478f2f280c266d7a5d1442997eb520fc20888f`. The committed PDM and Cargo locks
resolve only the corresponding public PyPI wheels and crates.io archives, with
registry hashes/checksums and no path, Git, URL, patch, or replacement source.

## PyPI wheel publication is intentionally frozen

Adam-assist `0.6.0` is already published on PyPI and crates.io; its files and
tag are immutable. This CPython 3.14 source candidate cannot be released under
that version. Any release of these bytes requires a separately authorized
Assist version bump plus matching package, workflow, input, assertion, and
artifact-identity updates. This change does not select that version.

The paired Core source at `cd4b24f9` removes the public
`adam_core::od::algorithms::external_provider_metadata` Rust API and moves the
public Arrow/Parquet boundary from 54 to 58 while still carrying a 0.7.0 source
label. A SemVer-compliant Core successor is therefore likely 0.8, outside every
current Assist Python and Rust range. A future PyPI wheel release requires
separately authorized Core wheels and Assist range, lock, workflow, and version
alignment. Until all of those prerequisites land, the mandatory 16-cell
acceptance matrix freezes every Assist publication to PyPI, including
3.11-3.13-only patches. It does not gate crates.io; nevertheless, changed crate
bytes cannot be republished under immutable 0.6.0 and require a separately
authorized Assist version plus Rust Core range/lock alignment.

In the CPython 3.14 wheel check, the Assist wheel is built with its registry-only
Cargo lock and embeds registry Core 0.7 Rust crates. The separately loaded
non-release Python Core wheel is built from `cd4b24f9`; the external capsule ABI
between the extensions is unchanged. Only the separate direct-Rust lane
path-patches all six Core crates to `cd4b24f9`. Exact source-built wheel and
direct-Rust evidence never substitutes for immutable registry artifacts. The
dedicated `migration/locks/adam-assist-core-cd4b24f9.Cargo.lock` is used only
for this exact disposable direct-Rust pairing; the registry-only crate lock
remains the release authority.

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
crate archive and 16 platform/Python wheel artifacts. The CPython 3.14 cells
fail closed while the authorized Core and Assist release alignment above is
absent. Publication workflows accept only successful exact-commit candidate
runs, reverify checksums, and upload the accepted bytes without rebuilding.
Rust 1.87 remains the authoritative MSRV for formatting, strict Clippy, tests, documentation,
packaging, and clean consumers.
