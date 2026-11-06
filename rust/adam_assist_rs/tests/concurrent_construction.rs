//! High-contention regression for ASSIST's process-global ephemeris parser.
//!
//! The live test is ignored by default because it needs the 765 MiB reviewed
//! kernel pair. Run it repeatedly in exact-source acceptance. All workers burst
//! through raw-wrapper and path-backed provider construction before any worker
//! inspects its results. Each result then queries the eagerly parsed constants,
//! masses and segment tables directly, so the test remains sensitive to parser
//! races without spacing constructions apart with a full provenance hash.

use adam_assist::assist_propagation::{assist_propagate_single, Orbit as AssistOrbit};
use adam_assist::{AssistData, AssistPropagator, Ephemeris};
use adam_core_rs_coords::propagation::{
    CovariancePropagation, PropagationOptions, PropagationRequest, Propagator, TwoBodyPropagator,
};
use adam_core_rs_coords::{
    CoordinateBatch, DataFrame, Epoch, ObjectId, OrbitBatch, OrbitId, OriginArray, OriginId,
    SchemaError, TimeArray, TimeScale, TimeScaleProvider,
};
use libassist_sys::ffi;
use std::sync::{Arc, Barrier};

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

fn propagate_constructed_data(data: &AssistData) -> Vec<[f64; 6]> {
    let results = assist_propagate_single(
        data,
        &AssistOrbit::new([1.2, 0.1, 0.05, -0.002, 0.016, 0.001], 60_000.0),
        &[60_400.0],
        false,
        &Default::default(),
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
    let targets = TimeArray::new(TimeScale::Tdb, vec![Epoch::new(60_400, 0)]).unwrap();
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

/// Values backed by every eagerly parsed input class: constants, masses and
/// planetary/small-body summary tables at two epochs.
fn ephemeris_signature(data: &AssistData) -> Vec<f64> {
    let ephem = &data.ephem;
    let mut values = vec![ephem.jd_ref(), ephem.c_au_per_day(), ephem.emrat()];
    for time in [0.0, 400.0] {
        for body in [ffi::ASSIST_BODY_SUN, ffi::ASSIST_BODY_JUPITER] {
            let particle = ephem.get_body_state(body, time).unwrap();
            values.extend([
                particle.m,
                particle.x,
                particle.y,
                particle.z,
                particle.vx,
                particle.vy,
                particle.vz,
            ]);
        }
        // SB441-n16 stores position segments; upstream deliberately reports
        // asteroid velocity as NaN because the polynomial derivative is not
        // part of this query contract. Exercise the finite parsed GM/position
        // fields rather than mistaking that sentinel for a construction race.
        let asteroid = ephem
            .get_body_state(ffi::ASSIST_BODY_NPLANETS, time)
            .unwrap();
        values.extend([asteroid.m, asteroid.x, asteroid.y, asteroid.z]);
    }
    assert!(
        values.iter().all(|value| value.is_finite()),
        "ephemeris signature contains non-finite values: {values:?}"
    );
    assert!(values[3 + 7] > 0.0, "Jupiter GM must be positive");
    values
}

enum Constructed {
    Raw(Box<AssistData>),
    Provider(AssistPropagator),
}

impl Constructed {
    fn data(&self) -> &AssistData {
        match self {
            Self::Raw(data) => data.as_ref(),
            Self::Provider(provider) => provider.data().as_ref(),
        }
    }
}

#[test]
#[ignore = "live high-contention test requires the reviewed ASSIST kernel pair"]
fn mixed_raw_and_provider_construction_is_finite_and_bitwise_equal_to_massive_body_reference() {
    const THREADS: usize = 8;
    const CONSTRUCTIONS_PER_THREAD: usize = 8;

    let (planets, asteroids) = kernel_paths();
    let reference_data = AssistData::new(
        Ephemeris::from_paths(
            std::path::Path::new(&planets),
            std::path::Path::new(&asteroids),
        )
        .unwrap(),
    );
    let reference_signature = Arc::new(ephemeris_signature(&reference_data));
    let reference = propagate_constructed_data(&reference_data);
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
        massive_body_difference > 1.0e-4,
        "ASSIST reference does not exhibit the expected massive-body dynamics: {massive_body_difference:e}"
    );
    println!("mixed construction reference: massive-body difference={massive_body_difference:e}");

    let start = Arc::new(Barrier::new(THREADS));
    let constructed = Arc::new(Barrier::new(THREADS));
    let workers = (0..THREADS)
        .map(|thread_index| {
            let planets = planets.clone();
            let asteroids = asteroids.clone();
            let reference_signature = Arc::clone(&reference_signature);
            let start = Arc::clone(&start);
            let constructed = Arc::clone(&constructed);
            std::thread::spawn(move || {
                start.wait();
                let built = (0..CONSTRUCTIONS_PER_THREAD)
                    .map(|construction_index| {
                        if (thread_index + construction_index) % 2 == 0 {
                            Constructed::Provider(
                                AssistPropagator::from_paths(&planets, &asteroids).unwrap(),
                            )
                        } else {
                            Constructed::Raw(Box::new(AssistData::new(
                                Ephemeris::from_paths(
                                    std::path::Path::new(&planets),
                                    std::path::Path::new(&asteroids),
                                )
                                .unwrap(),
                            )))
                        }
                    })
                    .collect::<Vec<_>>();
                // No worker inspects or hashes until all 64 constructions have
                // completed their burst through the shared parser lock.
                constructed.wait();
                for value in built {
                    let actual = ephemeris_signature(value.data());
                    assert_eq!(actual.len(), reference_signature.len());
                    for (index, (actual, expected)) in
                        actual.iter().zip(reference_signature.iter()).enumerate()
                    {
                        assert!(actual.is_finite() && expected.is_finite());
                        assert_eq!(
                            actual.to_bits(),
                            expected.to_bits(),
                            "concurrent construction changed eager value {index}: {actual:?} != {expected:?}"
                        );
                    }
                }
            })
        })
        .collect::<Vec<_>>();
    for worker in workers {
        worker.join().unwrap();
    }
}
