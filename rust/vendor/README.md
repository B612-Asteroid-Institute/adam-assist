# Vendored adam-core Rust contracts

These crates are a source snapshot from `adam-core-rust-migration` commit
`8116c69f` (branch `rust-migration-waves-d-e`), with
`adam_core_rs_coords/src/{propagation.rs,propagation/od.rs,orbit_least_squares.rs}`
updated to the backend-generic OD drivers from adam-core commits `7656d7f7` +
`3e6090f3` (bead personal-dqk):

- `adam_core_rs_autodiff`
- `adam_core_rs_coords`
- `adam_core_rs_spice`

They remain permissive, generic adam-core contracts; adam-assist does not own
their design. They are vendored temporarily so the GPL `adam_assist_rs` wheel
can build independently before those crates are published. Updates must be
copied from an identified adam-core commit and recorded here. Once published,
replace these paths with versioned crate dependencies and delete the snapshot.

Package ownership is intentionally separate:

- `assist-rs`: reusable low-level ASSIST/REBOUND primitives.
- `adam-assist`: GPL package semantics and orchestration (`adam_assist_rs`).
- `adam-core`: generic data, SPICE, propagation, and integration contracts.
