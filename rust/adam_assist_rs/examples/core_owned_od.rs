//! Standalone Rust ASSIST provider for adam-core's canonical OD workflow.

use adam_assist::AssistPropagator;
use adam_core::od::{AssociatedObservations, OdBatchResult, OdConfig, OdPropagator};
use std::path::Path;

fn run(
    planets: &Path,
    asteroids: &Path,
    input: &AssociatedObservations,
    config: &OdConfig,
) -> Result<OdBatchResult, Box<dyn std::error::Error>> {
    let assist = AssistPropagator::from_paths(planets, asteroids)?;
    let external = assist.external_propagator()?;
    let provider = OdPropagator::external(external.external());
    Ok(adam_core::run_od(input, config, &provider)?)
}

fn main() {
    // The caller supplies typed observations and a complete explicit config.
    // Keeping the function item type-checked proves the Python-free route.
    let _ = run;
}
