//! High-contention regression for ASSIST's process-global ephemeris parser.
//!
//! The live test is ignored by default because it needs the 765 MiB reviewed
//! kernel pair. Run it repeatedly in exact-source acceptance; raw `Ephemeris`
//! and path-backed provider constructors alternate under contention and share
//! one lowest-level lock, while propagation after construction remains parallel.

use adam_assist::assist_propagation::{assist_propagate_single, Orbit as AssistOrbit};
use adam_assist::{AssistData, AssistPropagator, Ephemeris};
use adam_core_rs_coords::propagation::{
    CovariancePropagation, PropagationOptions, PropagationRequest, Propagator, TwoBodyPropagator,
};
use adam_core_rs_coords::{
    CoordinateBatch, DataFrame, Epoch, ObjectId, OrbitBatch, OrbitId, OriginArray, OriginId,
    SchemaError, TimeArray, TimeScale, TimeScaleProvider,
};
use std::sync::Arc;

struct NoRescale;

impl TimeScaleProvider for NoRescale {
    fn rescale(&self, _times: &TimeArray, _scale: TimeScale) -> Result<TimeArray, SchemaError> {
        Err(SchemaError::InvalidRecordBatch(
            "concurrent construction test should not rescale time".to_string(),
        ))
    }
}

fn kernel_paths() -> (String, String) {
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

fn orbit() -> OrbitBatch {
    OrbitBatch::new(
        vec![OrbitId("construction-stress".to_string())],
        vec![Some(ObjectId("construction-stress".to_string()))],
        CoordinateBatch::cartesian(
            vec![[1.2, 0.1, 0.05, -0.002, 0.016, 0.001]],
            DataFrame::Ecliptic,
            OriginArray::repeat(OriginId::from_code("SUN"), 1),
            Some(TimeArray::new(TimeScale::Tdb, vec![Epoch::new(60_000, 0)]).unwrap()),
            None,
        )
        .unwrap(),
    )
    .unwrap()
}

fn propagate_constructed_data(propagator: &AssistPropagator) -> Vec<[f64; 6]> {
    let results = assist_propagate_single(
        propagator.data(),
        &AssistOrbit::new([1.2, 0.1, 0.05, -0.002, 0.016, 0.001], 60_000.0),
        &[60_001.0, 60_010.0],
        false,
        &propagator.integrator(),
    )
    .unwrap();
    let states = results
        .into_iter()
        .map(|result| result.state)
        .collect::<Vec<_>>();
    assert!(
        states.iter().flatten().all(|value| value.is_finite()),
        "constructed ephemeris produced a non-finite state: {states:?}"
    );
    states
}

fn propagate<P: Propagator>(propagator: &P) -> Vec<[f64; 6]> {
    let targets = TimeArray::new(
        TimeScale::Tdb,
        vec![Epoch::new(60_001, 0), Epoch::new(60_010, 0)],
    )
    .unwrap();
    let options = PropagationOptions {
        covariance: CovariancePropagation::None,
        thread_limit: Some(1),
        chunk_size: Some(1),
        ..PropagationOptions::default()
    };
    let orbits = orbit();
    let request = PropagationRequest::new(&orbits, &targets, options).unwrap();
    let result = propagator.propagate(&request, &NoRescale).unwrap();
    let states = result
        .orbits
        .coordinates
        .values
        .cartesian()
        .unwrap()
        .to_vec();
    assert!(
        states.iter().flatten().all(|value| value.is_finite()),
        "propagation produced a non-finite state: {states:?}"
    );
    states
}

#[test]
#[ignore = "live high-contention test requires the reviewed ASSIST kernel pair"]
fn mixed_raw_and_provider_construction_is_finite_and_bitwise_equal_to_massive_body_reference() {
    const THREADS: usize = 8;
    const CONSTRUCTIONS_PER_THREAD: usize = 8;

    let (planets, asteroids) = kernel_paths();
    let sequential = AssistPropagator::from_paths(&planets, &asteroids).unwrap();
    let reference = propagate_constructed_data(&sequential);
    let canonical = propagate(&sequential);
    let canonical_route_difference = canonical
        .iter()
        .flatten()
        .zip(reference.iter().flatten())
        .map(|(actual, expected)| {
            assert!(actual.is_finite() && expected.is_finite());
            (actual - expected).abs()
        })
        .fold(0.0_f64, f64::max);
    assert!(canonical_route_difference < 1.0e-12);
    let two_body = propagate(&TwoBodyPropagator::default());
    let massive_body_difference = reference
        .iter()
        .flatten()
        .zip(two_body.iter().flatten())
        .map(|(assist, kepler)| {
            assert!(assist.is_finite() && kepler.is_finite());
            (assist - kepler).abs()
        })
        .fold(0.0_f64, f64::max);
    assert!(
        massive_body_difference > 1.0e-8,
        "ASSIST reference does not exhibit the expected massive-body dynamics: {massive_body_difference:e}"
    );
    println!(
        "mixed construction reference: massive-body difference={massive_body_difference:e}, canonical-route difference={canonical_route_difference:e}"
    );

    let reference = Arc::new(reference);
    let workers = (0..THREADS)
        .map(|thread_index| {
            let planets = planets.clone();
            let asteroids = asteroids.clone();
            let reference = Arc::clone(&reference);
            std::thread::spawn(move || {
                for construction_index in 0..CONSTRUCTIONS_PER_THREAD {
                    let propagator = if (thread_index + construction_index) % 2 == 0 {
                        AssistPropagator::from_paths(&planets, &asteroids).unwrap()
                    } else {
                        AssistPropagator::new(Arc::new(AssistData::new(
                            Ephemeris::from_paths(
                                std::path::Path::new(&planets),
                                std::path::Path::new(&asteroids),
                            )
                            .unwrap(),
                        )))
                    };
                    // Exercise the newly constructed ephemeris and ASSIST's
                    // massive-body force setup without performing a separate
                    // 765 MiB provenance hash for every stress iteration.
                    // Canonical propagation above separately proves first-use
                    // trust establishment and exact output agreement.
                    let actual = propagate_constructed_data(&propagator);
                    assert_eq!(actual.len(), reference.len());
                    for (actual, expected) in actual.iter().zip(reference.iter()) {
                        assert!(actual.iter().all(|value| value.is_finite()));
                        assert!(expected.iter().all(|value| value.is_finite()));
                        for axis in 0..6 {
                            assert_eq!(
                                actual[axis].to_bits(),
                                expected[axis].to_bits(),
                                "concurrent construction changed state axis {axis}: {actual:?} != {expected:?}"
                            );
                        }
                    }
                }
            })
        })
        .collect::<Vec<_>>();
    for worker in workers {
        worker.join().unwrap();
    }
}
