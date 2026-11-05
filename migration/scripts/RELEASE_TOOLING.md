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
