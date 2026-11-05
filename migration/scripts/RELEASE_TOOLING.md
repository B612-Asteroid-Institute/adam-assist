# Preserved but disabled release tooling

The release-related Python modules and reviewed GitHub Actions workflows remain
in the repository so the release infrastructure is not discarded during the
Core 0.7 transition. Exact-source CI is the only active workflow: every job in
the release-candidate and publication workflows is explicitly disabled, so no
registry credential or publication command can execute.

A separate reviewed commit may remove those guards only after adam-core 0.7.0 is
public and authentic PyPI and crates.io locks have been generated. Enabling a
workflow does not itself authorize a registry upload, tag, or GitHub release;
publication still requires explicit human approval.
