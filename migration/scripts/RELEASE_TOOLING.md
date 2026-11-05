# Preserved but disabled release tooling

The release-related Python modules and reviewed GitHub Actions workflows remain
in the repository so the release infrastructure is not discarded during the
Core 0.7 transition. Exact-source CI is the only active workflow. Every release
job requires the unset-by-default repository variable
`ADAM_ASSIST_RELEASES_ENABLED` to equal `true`; `publish-pypi` additionally
retains its `dry_run == false` condition.

The variable is not configured. It may be enabled only after adam-core 0.7.0 is
public, authentic PyPI and crates.io locks are committed and reviewed, release
environment protections are verified, and separate publication authorization
is granted. Enabling a workflow does not itself authorize a registry upload,
tag, or GitHub release.

## Exact-source Core validation

Before registry artifacts exist, use one detached adam-core tree at the exact
reviewed commit for every Core crate. The source-pair verifier checks that tree
and the package metadata before Cargo runs. Patched Cargo commands create a
temporary untracked lock; remove it afterward and require a clean repository:

```console
git clone --no-hardlinks https://github.com/B612-Asteroid-Institute/adam_core.git ../adam-core-0b49938e
git -C ../adam-core-0b49938e checkout --detach 0b49938eb0be04214a8d07b206227d31b248394a
CORE="$(cd ../adam-core-0b49938e && pwd)"
test "$(git -C "$CORE" rev-parse HEAD)" = 0b49938eb0be04214a8d07b206227d31b248394a
python migration/scripts/verify_source_pair_contract.py --core-tree "$CORE"

cargo test --manifest-path rust/adam_assist_rs/Cargo.toml \
  --all-targets --features python,extension-module \
  --config "patch.crates-io.adam_core.path='$CORE/rust/adam_core'" \
  --config "patch.crates-io.adam_core_rs_coords.path='$CORE/rust/adam_core_rs_coords'" \
  --config "patch.crates-io.adam_core_rs_spice.path='$CORE/rust/adam_core_rs_spice'" \
  --config "patch.crates-io.adam_core_rs_kernel_data.path='$CORE/rust/adam_core_rs_kernel_data'"
rm -f rust/adam_assist_rs/Cargo.lock
test -z "$(git status --porcelain)"
```

This command is a focused all-feature test lane, not the complete acceptance
matrix. Active CI separately runs formatting, default/all/no-default checks,
warning-denied Clippy, tests, doctests, and warning-denied documentation.
