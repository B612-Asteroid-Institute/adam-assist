# Inert post-publication tooling

The release-related Python modules in this directory are local validation and
artifact-inspection tools only. No GitHub Actions workflow invokes them, no
release or publication workflow exists, and their presence does not authorize a
registry upload, tag, or GitHub release.

They may be wired into automation only after adam-core 0.7.0 is public, authentic
PyPI and crates.io locks have been generated, and a separate reviewed commit has
added both those locks and the release workflow. Publication still requires
explicit human approval.
