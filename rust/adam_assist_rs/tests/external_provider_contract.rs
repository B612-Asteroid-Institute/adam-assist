//! Standalone-Rust boundary checks for the propagation-only Core adapter.

use adam_assist::{AssistData, AssistExternalPropagator, AssistPropagator, Ephemeris};
use adam_core::od::OdPropagator;
use adam_core_rs_coords::propagation::Propagator;
use std::path::Path;
use std::sync::Arc;

fn assert_direct_provider<T: Propagator + Clone + Send + Sync>() {}
fn assert_owned_adapter<T: Send>() {}

#[test]
fn public_types_preserve_direct_and_adapter_boundaries() {
    assert_direct_provider::<AssistPropagator>();
    assert_owned_adapter::<AssistExternalPropagator>();
}

fn live_kernel_paths() -> (String, String) {
    let resolver = adam_core_rs_kernel_data::Resolver::from_env();
    let resolve = |variable: &str, id: &str| {
        std::env::var(variable).unwrap_or_else(|_| {
            resolver
                .resolve(id)
                .unwrap_or_else(|error| {
                    panic!("failed to resolve {id} kernel ({error}); set {variable}")
                })
                .display()
                .to_string()
        })
    };
    (
        resolve("ADAM_CORE_RS_ASSIST_PLANETS_PATH", "de440"),
        resolve("ADAM_CORE_RS_ASSIST_ASTEROIDS_PATH", "sb441_n16"),
    )
}

#[test]
#[ignore = "live kernel contract; exact-source CI exercises the same table through Python"]
fn adapter_owns_context_and_bare_ephemeris_cannot_claim_provenance() {
    let (planets, asteroids) = live_kernel_paths();
    let bare = AssistPropagator::new(Arc::new(AssistData::new(
        Ephemeris::from_paths(Path::new(&planets), Path::new(&asteroids)).unwrap(),
    )));
    let bare_error = match bare.external_propagator() {
        Ok(_) => panic!("bare ephemeris unexpectedly produced provider provenance"),
        Err(error) => error,
    };
    assert!(bare_error.to_string().contains("construct it from paths"));

    let assist = AssistPropagator::from_paths(&planets, &asteroids).unwrap();
    let configuration = assist.configuration_fingerprint();
    let data = assist.data_fingerprint().unwrap().to_string();
    let external = assist.external_propagator().unwrap();
    drop(assist);

    let handle = std::thread::spawn(move || {
        let metadata =
            adam_core::od::algorithms::external_provider_metadata(external.external()).unwrap();
        assert_eq!(metadata.configuration_fingerprint, configuration);
        assert_eq!(metadata.data_fingerprint, data);
        let _provider = OdPropagator::external(external.external());
    });
    handle.join().unwrap();
}
