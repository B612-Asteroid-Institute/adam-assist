#![allow(clippy::useless_conversion)] // PyO3 0.22 macro expansion trips this lint on generated wrappers.

use crate::AssistPropagator as RustAssistPropagator;
use crate::{map_origin_code_to_assist_body, CollisionConditionSpec, CollisionDetectionOutput};
use adam_core_rs_coords::propagation::{
    fit_orbit_least_squares_barycentric, fit_orbit_least_squares_evaluated_barycentric,
    iod_fit_linkages_barycentric, od_fit_barycentric, vallado_least_squares_barycentric,
    EvaluatedLeastSquaresFit, IodConfig, IodOutput, ObservationSelectionMethod, OdConfig, OdMethod,
    OdOutput, ValladoConfig, ValladoResult, ValladoStatus,
};
use adam_core_rs_coords::propagation::{
    CovariancePropagation, EpochPolicy, PropagationOptions, PropagationRequest, PropagationResult,
    Propagator,
};
use adam_core_rs_coords::types::Frame;
use adam_core_rs_coords::types::{SchemaResult, TimeScaleProvider};
use adam_core_rs_coords::{
    collapse_propagated_variants_to_orbits, collapse_variant_ephemeris,
    create_sampled_orbit_variants, CoordinateBatch, CoordinateRepresentation, CovarianceBatch,
    CovarianceUnits, EphemerisOptions, EphemerisPhotometryOptions, EphemerisResult, ObjectId,
    ObservatoryCode, ObserverBatch, OrbitBatch, OrbitId, OrbitVariantBatch,
    OrbitVariantSamplingMethod, OriginArray, OriginId, SchemaError, TimeArray, TimeScale,
    VariantId,
};
use adam_core_rs_coords::{CoordinateValues, LeastSquaresConfig};
use adam_core_rs_spice::AdamCoreSpiceBackend;
use assist_rs::{AssistData, Ephemeris, Ias15AdaptiveMode, IntegratorConfig};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::{HashMap, HashSet};
use std::hint::black_box;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

struct PythonTimeProvider;

impl TimeScaleProvider for PythonTimeProvider {
    fn rescale(&self, times: &TimeArray, new_scale: TimeScale) -> SchemaResult<TimeArray> {
        Err(SchemaError::InvalidRecordBatch(format!(
            "adam_assist cannot rescale {} to {} without an explicit provider",
            times.scale.as_str(),
            new_scale.as_str()
        )))
    }
}

#[derive(Clone)]
enum PreparedPropagationInput {
    Orbits(OrbitBatch),
    Variants(OrbitVariantBatch),
}

/// One fully-marshaled OD problem: single orbit, spherical astrometry with
/// covariance, aligned observers, and the shared ephemeris options.
#[derive(Clone)]
struct OdProblem {
    orbit: OrbitBatch,
    observed: CoordinateBatch,
    observers: ObserverBatch,
    options: EphemerisOptions,
}

#[derive(Clone)]
struct IodProblem {
    observed: CoordinateBatch,
    observers: ObserverBatch,
    options: EphemerisOptions,
}

#[derive(Clone)]
#[allow(clippy::large_enum_variant)]
enum NativeBenchmark {
    Propagation {
        input: PreparedPropagationInput,
        target_times: TimeArray,
        options: PropagationOptions,
        covariance_method: OrbitVariantSamplingMethod,
        num_samples: usize,
        seed: Option<u64>,
        sampled_covariance: bool,
    },
    Ephemeris {
        orbits: OrbitBatch,
        observers: ObserverBatch,
        options: EphemerisOptions,
        covariance_method: OrbitVariantSamplingMethod,
        num_samples: usize,
        seed: Option<u64>,
        sampled_covariance: bool,
    },
    Collisions {
        states: Vec<[f64; 6]>,
        epoch_jd_tdb: f64,
        final_jd_tdb: f64,
        conditions: Vec<CollisionConditionSpec>,
    },
    FitEvaluated {
        problem: OdProblem,
        ignore: Vec<bool>,
        config: LeastSquaresConfig,
    },
    OdFit {
        problem: OdProblem,
        config: OdConfig,
    },
    Vallado {
        problem: OdProblem,
        config: ValladoConfig,
    },
    Iod {
        problem: IodProblem,
        linkage_rows: Vec<Vec<usize>>,
        chunk_size: usize,
        config: IodConfig,
    },
}

impl NativeBenchmark {
    fn operation(&self) -> &'static str {
        match self {
            Self::Propagation {
                sampled_covariance: true,
                ..
            } => "sampled_covariance",
            Self::Propagation { .. } => "propagation",
            Self::Ephemeris { .. } => "ephemeris",
            Self::Collisions { .. } => "collisions",
            Self::FitEvaluated { .. } => "fit_least_squares_evaluated",
            Self::OdFit { .. } => "od_fit",
            Self::Vallado { .. } => "vallado_least_squares",
            Self::Iod { .. } => "initial_orbit_determination",
        }
    }
}

#[pyclass]
struct NativeAssistPropagator {
    inner: RustAssistPropagator,
    spice: AdamCoreSpiceBackend,
    benchmark: Mutex<Option<NativeBenchmark>>,
}

#[pymethods]
impl NativeAssistPropagator {
    #[new]
    #[pyo3(signature = (planets_path, asteroids_path, *, min_dt=1.0e-9, initial_dt=1.0e-6, adaptive_mode=1, epsilon=1.0e-6))]
    fn new(
        planets_path: &str,
        asteroids_path: &str,
        min_dt: f64,
        initial_dt: f64,
        adaptive_mode: i32,
        epsilon: f64,
    ) -> PyResult<Self> {
        if min_dt <= 0.0 {
            return Err(PyValueError::new_err("min_dt must be positive"));
        }
        if initial_dt <= 0.0 {
            return Err(PyValueError::new_err("initial_dt must be positive"));
        }
        if min_dt > initial_dt {
            return Err(PyValueError::new_err(
                "min_dt must be smaller than initial_dt",
            ));
        }
        let adaptive_mode = parse_adaptive_mode(adaptive_mode)?;
        let ephem = Ephemeris::from_paths(Path::new(planets_path), Path::new(asteroids_path))
            .map_err(|err| {
                PyRuntimeError::new_err(format!("failed to load ASSIST kernels: {err}"))
            })?;
        let integrator = IntegratorConfig {
            initial_dt: Some(initial_dt),
            min_dt: Some(min_dt),
            adaptive_mode: Some(adaptive_mode),
            epsilon: Some(epsilon),
        };
        let mut spice = AdamCoreSpiceBackend::new();
        spice.furnsh(Path::new(planets_path)).map_err(|err| {
            PyRuntimeError::new_err(format!("failed to load SPICE planets kernel: {err}"))
        })?;
        Ok(Self {
            inner: RustAssistPropagator::with_integrator(
                Arc::new(AssistData::new(ephem)),
                integrator,
            ),
            spice,
            benchmark: Mutex::new(None),
        })
    }

    /// Time the most recently prepared public ASSIST operation entirely in
    /// Rust. PyO3/NumPy input and output conversion stay outside samples.
    #[pyo3(signature = (reps, trials, warmup_reps=1))]
    fn benchmark_last_native(
        &self,
        py: Python<'_>,
        reps: usize,
        trials: usize,
        warmup_reps: usize,
    ) -> PyResult<(String, Vec<Vec<f64>>)> {
        if reps == 0 || trials == 0 {
            return Err(PyValueError::new_err("reps and trials must be >= 1"));
        }
        let benchmark = self
            .benchmark
            .lock()
            .map_err(|_| PyRuntimeError::new_err("native benchmark lock is poisoned"))?
            .clone()
            .ok_or_else(|| {
                PyValueError::new_err(
                    "no prepared ASSIST operation; call a public backend method first",
                )
            })?;
        let operation = benchmark.operation().to_string();
        let samples = py
            .allow_threads(|| {
                let mut trial_samples = Vec::with_capacity(trials);
                for _ in 0..trials {
                    for _ in 0..warmup_reps {
                        run_native_benchmark(self, &benchmark)?;
                    }
                    let mut samples = Vec::with_capacity(reps);
                    for _ in 0..reps {
                        let started = Instant::now();
                        run_native_benchmark(self, &benchmark)?;
                        samples.push(started.elapsed().as_secs_f64());
                    }
                    trial_samples.push(samples);
                }
                Ok::<_, String>(trial_samples)
            })
            .map_err(py_runtime_error)?;
        Ok((operation, samples))
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        orbit_ids,
        object_ids,
        orbit_states,
        orbit_origin_codes,
        orbit_frame,
        orbit_time_scale,
        orbit_time_days,
        orbit_time_nanos,
        observer_codes,
        observer_states,
        observer_origin_codes,
        observer_frame,
        observer_time_scale,
        observer_time_days,
        observer_time_nanos,
        output_time_scale,
        lt_tol=1.0e-12,
        max_iter=1000,
        tol=1.0e-15,
        stellar_aberration=false,
        max_lt_iter=10,
        predict_magnitude_v=false,
        predict_phase_angle=false,
        h_v=None,
        g=None,
        chunk_size=None,
        thread_limit=None,
        covariance=false,
        covariances=None,
        covariance_method="monte-carlo",
        num_samples=1000,
        seed=None
    ))]
    fn generate_ephemeris<'py>(
        &self,
        py: Python<'py>,
        orbit_ids: Vec<String>,
        object_ids: Vec<Option<String>>,
        orbit_states: PyReadonlyArray2<'py, f64>,
        orbit_origin_codes: Vec<String>,
        orbit_frame: &str,
        orbit_time_scale: &str,
        orbit_time_days: Vec<i64>,
        orbit_time_nanos: Vec<i64>,
        observer_codes: Vec<String>,
        observer_states: PyReadonlyArray2<'py, f64>,
        observer_origin_codes: Vec<String>,
        observer_frame: &str,
        observer_time_scale: &str,
        observer_time_days: Vec<i64>,
        observer_time_nanos: Vec<i64>,
        output_time_scale: &str,
        lt_tol: f64,
        max_iter: usize,
        tol: f64,
        stellar_aberration: bool,
        max_lt_iter: usize,
        predict_magnitude_v: bool,
        predict_phase_angle: bool,
        h_v: Option<Vec<Option<f64>>>,
        g: Option<Vec<Option<f64>>>,
        chunk_size: Option<usize>,
        thread_limit: Option<usize>,
        covariance: bool,
        covariances: Option<PyReadonlyArray2<'py, f64>>,
        covariance_method: &str,
        num_samples: usize,
        seed: Option<u64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let orbit_state_rows = states_from_pyarray(orbit_states)?;
        let orbit_covariance_batch = match covariances {
            Some(covariances) => Some(covariance_from_pyarray(
                covariances,
                orbit_state_rows.len(),
            )?),
            None => None,
        };
        let orbit_coordinates = CoordinateBatch::cartesian(
            orbit_state_rows,
            Frame::parse(orbit_frame).map_err(py_value_error)?,
            OriginArray::new(
                orbit_origin_codes
                    .into_iter()
                    .map(OriginId::from_code)
                    .collect(),
            ),
            Some(time_array(
                orbit_time_scale,
                orbit_time_days,
                orbit_time_nanos,
            )?),
            orbit_covariance_batch,
        )
        .map_err(py_value_error)?;
        let orbits = OrbitBatch::new(
            orbit_ids.into_iter().map(OrbitId).collect(),
            object_ids
                .into_iter()
                .map(|value| value.map(ObjectId))
                .collect(),
            orbit_coordinates,
        )
        .map_err(py_value_error)?;
        let observer_coordinates = CoordinateBatch::cartesian(
            states_from_pyarray(observer_states)?,
            Frame::parse(observer_frame).map_err(py_value_error)?,
            OriginArray::new(
                observer_origin_codes
                    .into_iter()
                    .map(OriginId::from_code)
                    .collect(),
            ),
            Some(time_array(
                observer_time_scale,
                observer_time_days,
                observer_time_nanos,
            )?),
            None,
        )
        .map_err(py_value_error)?;
        let observers = ObserverBatch::new(
            observer_codes.into_iter().map(ObservatoryCode).collect(),
            observer_coordinates,
        )
        .map_err(py_value_error)?;
        let options = EphemerisOptions {
            propagation: PropagationOptions {
                chunk_size,
                thread_limit,
                epoch_policy: EpochPolicy::CrossProduct,
                covariance: CovariancePropagation::None,
            },
            lt_tol,
            max_iter,
            tol,
            stellar_aberration,
            max_lt_iter,
            output_time_scale: TimeScale::parse(output_time_scale).map_err(py_value_error)?,
            include_aberrated_coordinates: true,
            photometry: EphemerisPhotometryOptions {
                predict_magnitude_v,
                predict_phase_angle,
                h_v,
                g,
            },
        };
        let benchmark = NativeBenchmark::Ephemeris {
            orbits,
            observers,
            options,
            covariance_method: parse_covariance_method(covariance_method)?,
            num_samples,
            seed,
            sampled_covariance: covariance,
        };
        let result = py
            .allow_threads(|| run_ephemeris_benchmark(self, &benchmark))
            .map_err(py_runtime_error)?;
        self.store_benchmark(benchmark)?;
        ephemeris_result_to_dict(py, &result)
    }

    /// Backend-generic least-squares orbit determination instantiated with
    /// the ASSIST propagator (bead personal-cmy.7). The Gauss-Newton driver
    /// lives in the permissive core (`fit_orbit_least_squares_barycentric`);
    /// this GPL boundary only supplies the propagator, mirroring the
    /// adam-assist packaging decision. Returns
    /// `(state (6,), covariance (36,), chi2, iterations, converged)` in the
    /// input orbit's frame/origin.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    #[pyo3(signature = (
        orbit_ids,
        object_ids,
        orbit_states,
        orbit_origin_codes,
        orbit_frame,
        orbit_time_scale,
        orbit_time_days,
        orbit_time_nanos,
        observed_values,
        observed_covariances,
        observer_codes,
        observer_states,
        observer_origin_codes,
        observer_frame,
        observer_time_scale,
        observer_time_days,
        observer_time_nanos,
        xtol=1e-12,
        ftol=1e-12,
        max_iterations=100,
        lt_tol=1.0e-12,
        eph_max_iter=1000,
        eph_tol=1.0e-15,
        stellar_aberration=false,
        max_lt_iter=10
    ))]
    fn fit_orbit_least_squares<'py>(
        &self,
        py: Python<'py>,
        orbit_ids: Vec<String>,
        object_ids: Vec<Option<String>>,
        orbit_states: PyReadonlyArray2<'py, f64>,
        orbit_origin_codes: Vec<String>,
        orbit_frame: &str,
        orbit_time_scale: &str,
        orbit_time_days: Vec<i64>,
        orbit_time_nanos: Vec<i64>,
        observed_values: PyReadonlyArray2<'py, f64>,
        observed_covariances: PyReadonlyArray2<'py, f64>,
        observer_codes: Vec<String>,
        observer_states: PyReadonlyArray2<'py, f64>,
        observer_origin_codes: Vec<String>,
        observer_frame: &str,
        observer_time_scale: &str,
        observer_time_days: Vec<i64>,
        observer_time_nanos: Vec<i64>,
        xtol: f64,
        ftol: f64,
        max_iterations: usize,
        lt_tol: f64,
        eph_max_iter: usize,
        eph_tol: f64,
        stellar_aberration: bool,
        max_lt_iter: usize,
    ) -> PyResult<(Vec<f64>, Vec<f64>, f64, usize, bool)> {
        let orbit_coordinates = CoordinateBatch::cartesian(
            states_from_pyarray(orbit_states)?,
            Frame::parse(orbit_frame).map_err(py_value_error)?,
            OriginArray::new(
                orbit_origin_codes
                    .into_iter()
                    .map(OriginId::from_code)
                    .collect(),
            ),
            Some(time_array(
                orbit_time_scale,
                orbit_time_days,
                orbit_time_nanos,
            )?),
            None,
        )
        .map_err(py_value_error)?;
        let orbit = OrbitBatch::new(
            orbit_ids.into_iter().map(OrbitId).collect(),
            object_ids
                .into_iter()
                .map(|value| value.map(ObjectId))
                .collect(),
            orbit_coordinates,
        )
        .map_err(py_value_error)?;

        let observer_times =
            time_array(observer_time_scale, observer_time_days, observer_time_nanos)?;
        let observer_origins = OriginArray::new(
            observer_origin_codes
                .into_iter()
                .map(OriginId::from_code)
                .collect(),
        );
        let observed_rows = states_from_pyarray(observed_values)?;
        let n = observed_rows.len();
        let observed_cov = observed_covariances.as_array();
        if observed_cov.nrows() != n || observed_cov.ncols() != 36 {
            return Err(PyValueError::new_err(
                "observed_covariances must have shape (N, 36)",
            ));
        }
        let observed_cov_flat: Vec<f64> = observed_cov.iter().copied().collect();
        let covariance = CovarianceBatch::new(
            n,
            6,
            observed_cov_flat,
            CovarianceUnits::Coordinate(CoordinateRepresentation::Spherical),
        )
        .map_err(py_value_error)?;
        let observed = CoordinateBatch::new(
            CoordinateValues::Spherical(observed_rows),
            Frame::parse(observer_frame).map_err(py_value_error)?,
            observer_origins.clone(),
            Some(observer_times.clone()),
            Some(covariance),
        )
        .map_err(py_value_error)?;

        let observer_coordinates = CoordinateBatch::cartesian(
            states_from_pyarray(observer_states)?,
            Frame::parse(observer_frame).map_err(py_value_error)?,
            observer_origins,
            Some(observer_times),
            None,
        )
        .map_err(py_value_error)?;
        let observers = ObserverBatch::new(
            observer_codes.into_iter().map(ObservatoryCode).collect(),
            observer_coordinates,
        )
        .map_err(py_value_error)?;

        let options = EphemerisOptions {
            propagation: PropagationOptions {
                chunk_size: None,
                thread_limit: None,
                epoch_policy: EpochPolicy::CrossProduct,
                covariance: CovariancePropagation::None,
            },
            lt_tol,
            max_iter: eph_max_iter,
            tol: eph_tol,
            stellar_aberration,
            max_lt_iter,
            output_time_scale: TimeScale::parse(observer_time_scale).map_err(py_value_error)?,
            include_aberrated_coordinates: false,
            photometry: EphemerisPhotometryOptions::default(),
        };
        let config = LeastSquaresConfig {
            xtol,
            ftol,
            max_iterations,
            lt_tol,
            ephemeris_max_iter: eph_max_iter,
            ephemeris_tol: eph_tol,
            stellar_aberration,
            max_lt_iter,
        };
        let fit = py
            .allow_threads(|| {
                fit_orbit_least_squares_barycentric(
                    &self.inner,
                    &orbit,
                    &observed,
                    &observers,
                    &options,
                    &config,
                    &PythonTimeProvider,
                    &self.spice,
                )
            })
            .map_err(|err| {
                PyRuntimeError::new_err(format!("assist least-squares fit failed: {err}"))
            })?;
        Ok((
            fit.state.to_vec(),
            fit.covariance.to_vec(),
            fit.chi2,
            fit.iterations,
            fit.converged,
        ))
    }

    /// Fused `fit_least_squares` work unit (bead personal-dqk): the
    /// Gauss-Newton fit on the non-ignored subset plus the final
    /// `evaluate_orbits`-style residual/statistics pass, all in one crossing.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        orbit_ids,
        object_ids,
        orbit_states,
        orbit_origin_codes,
        orbit_frame,
        orbit_time_scale,
        orbit_time_days,
        orbit_time_nanos,
        observed_values,
        observed_covariances,
        observer_codes,
        observer_states,
        observer_origin_codes,
        observer_frame,
        observer_time_scale,
        observer_time_days,
        observer_time_nanos,
        ignore,
        xtol=1e-12,
        ftol=1e-12,
        max_iterations=100,
        lt_tol=1.0e-12,
        eph_max_iter=1000,
        eph_tol=1.0e-15,
        stellar_aberration=false,
        max_lt_iter=10
    ))]
    fn fit_orbit_least_squares_evaluated<'py>(
        &self,
        py: Python<'py>,
        orbit_ids: Vec<String>,
        object_ids: Vec<Option<String>>,
        orbit_states: PyReadonlyArray2<'py, f64>,
        orbit_origin_codes: Vec<String>,
        orbit_frame: &str,
        orbit_time_scale: &str,
        orbit_time_days: Vec<i64>,
        orbit_time_nanos: Vec<i64>,
        observed_values: PyReadonlyArray2<'py, f64>,
        observed_covariances: PyReadonlyArray2<'py, f64>,
        observer_codes: Vec<String>,
        observer_states: PyReadonlyArray2<'py, f64>,
        observer_origin_codes: Vec<String>,
        observer_frame: &str,
        observer_time_scale: &str,
        observer_time_days: Vec<i64>,
        observer_time_nanos: Vec<i64>,
        ignore: Vec<bool>,
        xtol: f64,
        ftol: f64,
        max_iterations: usize,
        lt_tol: f64,
        eph_max_iter: usize,
        eph_tol: f64,
        stellar_aberration: bool,
        max_lt_iter: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let problem = build_od_problem(
            orbit_ids,
            object_ids,
            orbit_states,
            orbit_origin_codes,
            orbit_frame,
            orbit_time_scale,
            orbit_time_days,
            orbit_time_nanos,
            observed_values,
            observed_covariances,
            observer_codes,
            observer_states,
            observer_origin_codes,
            observer_frame,
            observer_time_scale,
            observer_time_days,
            observer_time_nanos,
            lt_tol,
            eph_max_iter,
            eph_tol,
            stellar_aberration,
            max_lt_iter,
        )?;
        let config = LeastSquaresConfig {
            xtol,
            ftol,
            max_iterations,
            lt_tol,
            ephemeris_max_iter: eph_max_iter,
            ephemeris_tol: eph_tol,
            stellar_aberration,
            max_lt_iter,
        };
        let output = py
            .allow_threads(|| run_fit_evaluated(self, &problem, &ignore, &config))
            .map_err(py_runtime_error)?;
        self.store_benchmark(NativeBenchmark::FitEvaluated {
            problem,
            ignore,
            config,
        })?;

        let dict = PyDict::new_bound(py);
        dict.set_item("state", output.fit.state.to_vec())?;
        dict.set_item("covariance", output.fit.covariance.to_vec())?;
        dict.set_item("fit_chi2", output.fit.chi2)?;
        dict.set_item("iterations", output.fit.iterations)?;
        dict.set_item("converged", output.fit.converged)?;
        evaluation_into_dict(&dict, py, &output.evaluation)?;
        Ok(dict)
    }

    /// The full legacy `od()` differential-correction loop for one orbit in
    /// one crossing (bead personal-dqk): delta bounding, finite/central
    /// perturbation batching, weighted normal equations, condition and
    /// covariance sanity rejections, acceptance bookkeeping, and chi2-ranked
    /// outlier retries.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        orbit_ids,
        object_ids,
        orbit_states,
        orbit_origin_codes,
        orbit_frame,
        orbit_time_scale,
        orbit_time_days,
        orbit_time_nanos,
        observed_values,
        observed_covariances,
        observer_codes,
        observer_states,
        observer_origin_codes,
        observer_frame,
        observer_time_scale,
        observer_time_days,
        observer_time_nanos,
        rchi2_threshold=100.0,
        min_obs=5,
        min_arc_length=1.0,
        contamination_percentage=0.0,
        delta=1e-6,
        max_iter=20,
        method="central",
        lt_tol=1.0e-12,
        eph_max_iter=1000,
        eph_tol=1.0e-15,
        stellar_aberration=false,
        max_lt_iter=10
    ))]
    fn od_fit<'py>(
        &self,
        py: Python<'py>,
        orbit_ids: Vec<String>,
        object_ids: Vec<Option<String>>,
        orbit_states: PyReadonlyArray2<'py, f64>,
        orbit_origin_codes: Vec<String>,
        orbit_frame: &str,
        orbit_time_scale: &str,
        orbit_time_days: Vec<i64>,
        orbit_time_nanos: Vec<i64>,
        observed_values: PyReadonlyArray2<'py, f64>,
        observed_covariances: PyReadonlyArray2<'py, f64>,
        observer_codes: Vec<String>,
        observer_states: PyReadonlyArray2<'py, f64>,
        observer_origin_codes: Vec<String>,
        observer_frame: &str,
        observer_time_scale: &str,
        observer_time_days: Vec<i64>,
        observer_time_nanos: Vec<i64>,
        rchi2_threshold: f64,
        min_obs: usize,
        min_arc_length: f64,
        contamination_percentage: f64,
        delta: f64,
        max_iter: usize,
        method: &str,
        lt_tol: f64,
        eph_max_iter: usize,
        eph_tol: f64,
        stellar_aberration: bool,
        max_lt_iter: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let od_method = parse_od_method(method)?;
        let problem = build_od_problem(
            orbit_ids,
            object_ids,
            orbit_states,
            orbit_origin_codes,
            orbit_frame,
            orbit_time_scale,
            orbit_time_days,
            orbit_time_nanos,
            observed_values,
            observed_covariances,
            observer_codes,
            observer_states,
            observer_origin_codes,
            observer_frame,
            observer_time_scale,
            observer_time_days,
            observer_time_nanos,
            lt_tol,
            eph_max_iter,
            eph_tol,
            stellar_aberration,
            max_lt_iter,
        )?;
        let config = OdConfig {
            rchi2_threshold,
            min_obs,
            min_arc_length,
            contamination_percentage,
            delta,
            max_iter,
            method: od_method,
        };
        let output = py
            .allow_threads(|| run_od_fit(self, &problem, &config))
            .map_err(py_runtime_error)?;
        self.store_benchmark(NativeBenchmark::OdFit { problem, config })?;

        let dict = PyDict::new_bound(py);
        dict.set_item("found", output.found)?;
        dict.set_item("state", output.state.to_vec())?;
        dict.set_item("covariance", output.covariance.to_vec())?;
        dict.set_item("arc_length", output.arc_length)?;
        dict.set_item("num_obs", output.num_obs)?;
        dict.set_item("chi2", output.chi2_total)?;
        dict.set_item("reduced_chi2", output.reduced_chi2)?;
        dict.set_item("iterations", output.iterations)?;
        dict.set_item("improved", output.improved)?;
        let n = output.residual_chi2.len();
        dict.set_item(
            "residual_values",
            shaped_matrix_array(py, &output.residuals, n, 6)?,
        )?;
        dict.set_item("residual_chi2", output.residual_chi2.clone())?;
        dict.set_item("residual_dof", output.residual_dof.clone())?;
        dict.set_item("residual_probability", output.residual_probability.clone())?;
        dict.set_item("outlier", output.outlier.clone())?;
        Ok(dict)
    }

    /// Complete Gauss-IOD orchestration for every linkage in one crossing:
    /// Rust owns member indexing, per-linkage chronological ordering, triplet
    /// selection, candidate scoring/outliers, deduplication, and final order.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        observation_ids,
        observed_values,
        observed_covariances,
        observed_origin_codes,
        observed_frame,
        observed_time_scale,
        observed_time_days,
        observed_time_nanos,
        observer_codes,
        observer_states,
        observer_origin_codes,
        observer_frame,
        observer_time_scale,
        observer_time_days,
        observer_time_nanos,
        linkage_ids,
        member_linkage_ids,
        member_obs_ids,
        min_obs=6,
        min_arc_length=1.0,
        contamination_percentage=0.0,
        rchi2_threshold=200.0,
        observation_selection_method="combinations",
        light_time=true,
        chunk_size=1,
        mu=0.00029591220828559115,
        speed_of_light=173.14463267424034,
        lt_tol=1.0e-12,
        eph_max_iter=1000,
        eph_tol=1.0e-15,
        stellar_aberration=false,
        max_lt_iter=10
    ))]
    fn initial_orbit_determination<'py>(
        &self,
        py: Python<'py>,
        observation_ids: Vec<String>,
        observed_values: PyReadonlyArray2<'py, f64>,
        observed_covariances: PyReadonlyArray2<'py, f64>,
        observed_origin_codes: Vec<String>,
        observed_frame: &str,
        observed_time_scale: &str,
        observed_time_days: Vec<i64>,
        observed_time_nanos: Vec<i64>,
        observer_codes: Vec<String>,
        observer_states: PyReadonlyArray2<'py, f64>,
        observer_origin_codes: Vec<String>,
        observer_frame: &str,
        observer_time_scale: &str,
        observer_time_days: Vec<i64>,
        observer_time_nanos: Vec<i64>,
        linkage_ids: Vec<String>,
        member_linkage_ids: Vec<String>,
        member_obs_ids: Vec<String>,
        min_obs: usize,
        min_arc_length: f64,
        contamination_percentage: f64,
        rchi2_threshold: f64,
        observation_selection_method: &str,
        light_time: bool,
        chunk_size: usize,
        mu: f64,
        speed_of_light: f64,
        lt_tol: f64,
        eph_max_iter: usize,
        eph_tol: f64,
        stellar_aberration: bool,
        max_lt_iter: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        if observation_ids.len() != observed_origin_codes.len()
            || observation_ids.len() != observed_time_days.len()
            || observation_ids.len() != observed_time_nanos.len()
        {
            return Err(PyValueError::new_err("observation columns must align"));
        }
        if member_linkage_ids.len() != member_obs_ids.len() {
            return Err(PyValueError::new_err("linkage member columns must align"));
        }
        let mut observation_index = HashMap::with_capacity(observation_ids.len());
        for (row, id) in observation_ids.iter().enumerate() {
            observation_index.insert(id.as_str(), row);
        }
        let mut rows_by_linkage: HashMap<&str, Vec<usize>> = HashMap::new();
        for (linkage_id, obs_id) in member_linkage_ids.iter().zip(member_obs_ids.iter()) {
            if let Some(&row) = observation_index.get(obs_id.as_str()) {
                rows_by_linkage
                    .entry(linkage_id.as_str())
                    .or_default()
                    .push(row);
            }
        }
        let mut linkage_rows = Vec::with_capacity(linkage_ids.len());
        for linkage_id in &linkage_ids {
            let mut rows = rows_by_linkage
                .remove(linkage_id.as_str())
                .unwrap_or_default();
            rows.sort_by(|&left, &right| {
                observed_time_days[left]
                    .cmp(&observed_time_days[right])
                    .then(observed_time_nanos[left].cmp(&observed_time_nanos[right]))
                    .then(observed_origin_codes[left].cmp(&observed_origin_codes[right]))
            });
            linkage_rows.push(rows);
        }
        let problem = build_iod_problem(
            observed_values,
            observed_covariances,
            observed_origin_codes,
            observed_frame,
            observed_time_scale,
            observed_time_days,
            observed_time_nanos,
            observer_codes,
            observer_states,
            observer_origin_codes,
            observer_frame,
            observer_time_scale,
            observer_time_days,
            observer_time_nanos,
            lt_tol,
            eph_max_iter,
            eph_tol,
            stellar_aberration,
            max_lt_iter,
        )?;
        let config = IodConfig {
            min_obs,
            min_arc_length,
            contamination_percentage,
            rchi2_threshold,
            observation_selection_method: parse_observation_selection_method(
                observation_selection_method,
            )?,
            light_time,
            mu,
            speed_of_light,
        };
        let outputs = py
            .allow_threads(|| run_iod_linkages(self, &problem, &linkage_rows, chunk_size, &config))
            .map_err(py_runtime_error)?;
        self.store_benchmark(NativeBenchmark::Iod {
            problem,
            linkage_rows: linkage_rows.clone(),
            chunk_size,
            config,
        })?;

        let mut seen = HashSet::<[u64; 7]>::new();
        let mut accepted: Vec<(String, Vec<usize>, IodOutput)> = linkage_ids
            .into_iter()
            .zip(linkage_rows)
            .zip(outputs)
            .filter_map(|((linkage_id, rows), output)| {
                if !output.found {
                    return None;
                }
                let key = [
                    output.epoch_mjd.to_bits(),
                    output.state[0].to_bits(),
                    output.state[1].to_bits(),
                    output.state[2].to_bits(),
                    output.state[3].to_bits(),
                    output.state[4].to_bits(),
                    output.state[5].to_bits(),
                ];
                seen.insert(key).then_some((linkage_id, rows, output))
            })
            .collect();
        accepted.sort_by(|left, right| left.0.cmp(&right.0));

        let orbit_ids: Vec<String> = accepted.iter().map(|row| row.0.clone()).collect();
        let states: Vec<[f64; 6]> = accepted.iter().map(|row| row.2.state).collect();
        let epochs: Vec<f64> = accepted.iter().map(|row| row.2.epoch_mjd).collect();
        let arc_lengths: Vec<f64> = accepted.iter().map(|row| row.2.arc_length).collect();
        let num_obs: Vec<usize> = accepted.iter().map(|row| row.2.num_obs).collect();
        let chi2: Vec<f64> = accepted.iter().map(|row| row.2.chi2_total).collect();
        let reduced_chi2: Vec<f64> = accepted.iter().map(|row| row.2.reduced_chi2).collect();
        let mut member_orbit_ids = Vec::new();
        let mut member_obs_ids_out = Vec::new();
        let mut residual_values = Vec::new();
        let mut residual_chi2 = Vec::new();
        let mut residual_dof = Vec::new();
        let mut residual_probability = Vec::new();
        let mut solution = Vec::new();
        let mut outlier = Vec::new();
        for (linkage_id, rows, output) in &accepted {
            for (member, &global_row) in rows.iter().enumerate() {
                member_orbit_ids.push(linkage_id.clone());
                member_obs_ids_out.push(observation_ids[global_row].clone());
                residual_values.extend_from_slice(&output.residuals[member * 6..member * 6 + 6]);
                residual_chi2.push(output.residual_chi2[member]);
                residual_dof.push(output.residual_dof[member]);
                residual_probability.push(output.residual_probability[member]);
                solution.push(output.solution[member]);
                outlier.push(output.outlier[member]);
            }
        }
        let dict = PyDict::new_bound(py);
        dict.set_item("orbit_ids", orbit_ids)?;
        dict.set_item("states", shaped_states_array(py, &states)?)?;
        dict.set_item("epoch_mjd", epochs)?;
        dict.set_item("arc_length", arc_lengths)?;
        dict.set_item("num_obs", num_obs)?;
        dict.set_item("chi2", chi2)?;
        dict.set_item("reduced_chi2", reduced_chi2)?;
        dict.set_item("member_orbit_ids", member_orbit_ids)?;
        dict.set_item("member_obs_ids", member_obs_ids_out)?;
        dict.set_item(
            "residual_values",
            shaped_matrix_array(py, &residual_values, residual_chi2.len(), 6)?,
        )?;
        dict.set_item("residual_chi2", residual_chi2)?;
        dict.set_item("residual_dof", residual_dof)?;
        dict.set_item("residual_probability", residual_probability)?;
        dict.set_item("solution", solution)?;
        dict.set_item("outlier", outlier)?;
        Ok(dict)
    }

    /// The public `LeastSquares.least_squares` Vallado RMS algorithm in one
    /// crossing (bead personal-dqk), including the debug iteration trace.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        orbit_ids,
        object_ids,
        orbit_states,
        orbit_origin_codes,
        orbit_frame,
        orbit_time_scale,
        orbit_time_days,
        orbit_time_nanos,
        observed_values,
        observed_covariances,
        observer_codes,
        observer_states,
        observer_origin_codes,
        observer_frame,
        observer_time_scale,
        observer_time_days,
        observer_time_nanos,
        use_central_difference,
        perturbation_initial_fraction=1e-6,
        perturbation_multiplier=0.5,
        rms_epsilon=1e-3,
        max_iterations=20,
        lt_tol=1.0e-12,
        eph_max_iter=1000,
        eph_tol=1.0e-15,
        stellar_aberration=false,
        max_lt_iter=10
    ))]
    fn vallado_least_squares<'py>(
        &self,
        py: Python<'py>,
        orbit_ids: Vec<String>,
        object_ids: Vec<Option<String>>,
        orbit_states: PyReadonlyArray2<'py, f64>,
        orbit_origin_codes: Vec<String>,
        orbit_frame: &str,
        orbit_time_scale: &str,
        orbit_time_days: Vec<i64>,
        orbit_time_nanos: Vec<i64>,
        observed_values: PyReadonlyArray2<'py, f64>,
        observed_covariances: PyReadonlyArray2<'py, f64>,
        observer_codes: Vec<String>,
        observer_states: PyReadonlyArray2<'py, f64>,
        observer_origin_codes: Vec<String>,
        observer_frame: &str,
        observer_time_scale: &str,
        observer_time_days: Vec<i64>,
        observer_time_nanos: Vec<i64>,
        use_central_difference: bool,
        perturbation_initial_fraction: f64,
        perturbation_multiplier: f64,
        rms_epsilon: f64,
        max_iterations: usize,
        lt_tol: f64,
        eph_max_iter: usize,
        eph_tol: f64,
        stellar_aberration: bool,
        max_lt_iter: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let problem = build_od_problem(
            orbit_ids,
            object_ids,
            orbit_states,
            orbit_origin_codes,
            orbit_frame,
            orbit_time_scale,
            orbit_time_days,
            orbit_time_nanos,
            observed_values,
            observed_covariances,
            observer_codes,
            observer_states,
            observer_origin_codes,
            observer_frame,
            observer_time_scale,
            observer_time_days,
            observer_time_nanos,
            lt_tol,
            eph_max_iter,
            eph_tol,
            stellar_aberration,
            max_lt_iter,
        )?;
        let config = ValladoConfig {
            use_central_difference,
            perturbation_initial_fraction,
            perturbation_multiplier,
            rms_epsilon,
            max_iterations,
        };
        let output = py
            .allow_threads(|| run_vallado(self, &problem, &config))
            .map_err(py_runtime_error)?;
        self.store_benchmark(NativeBenchmark::Vallado { problem, config })?;

        let dict = PyDict::new_bound(py);
        dict.set_item(
            "status",
            match output.status {
                ValladoStatus::Updated => "updated",
                ValladoStatus::Initial => "initial",
                ValladoStatus::NotImproved => "not_improved",
            },
        )?;
        dict.set_item("state", output.state.to_vec())?;
        dict.set_item("covariance", output.covariance.to_vec())?;
        dict.set_item("num_observations", output.num_observations)?;
        dict.set_item(
            "iterations_rchi2",
            output
                .iterations
                .iter()
                .map(|record| record.rchi2)
                .collect::<Vec<_>>(),
        )?;
        dict.set_item(
            "iterations_rms",
            output
                .iterations
                .iter()
                .map(|record| record.rms)
                .collect::<Vec<_>>(),
        )?;
        dict.set_item(
            "iterations_delta_rms",
            output
                .iterations
                .iter()
                .map(|record| record.delta_rms)
                .collect::<Vec<_>>(),
        )?;
        dict.set_item(
            "iterations_converged",
            output
                .iterations
                .iter()
                .map(|record| record.converged)
                .collect::<Vec<_>>(),
        )?;
        dict.set_item(
            "iterations_perturbation",
            output
                .iterations
                .iter()
                .map(|record| record.perturbation)
                .collect::<Vec<_>>(),
        )?;
        dict.set_item(
            "iterations_error",
            output
                .iterations
                .iter()
                .map(|record| record.error.clone())
                .collect::<Vec<_>>(),
        )?;
        let corrections_flat: Vec<f64> = output
            .corrections
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect();
        dict.set_item(
            "corrections",
            shaped_matrix_array(py, &corrections_flat, output.corrections.len(), 6)?,
        )?;
        dict.set_item("exit_message", output.exit_message.clone())?;
        Ok(dict)
    }

    /// Same-epoch collision detection mirroring
    /// `adam_assist.ASSISTPropagator._detect_collisions`. `states` must be
    /// barycentric equatorial (N, 6) at one shared TDB epoch; the epoch and
    /// horizon are TDB Julian dates computed by the Python boundary exactly
    /// as legacy does. Returns per-row survivor states/indices at the final
    /// executed (overshooting) step plus one impact record per
    /// condition-step detection, all in barycentric equatorial with TDB
    /// Julian-date times.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        states,
        epoch_jd_tdb,
        final_jd_tdb,
        condition_bodies,
        condition_distances_km,
        condition_stopping
    ))]
    fn detect_collisions<'py>(
        &self,
        py: Python<'py>,
        states: PyReadonlyArray2<'py, f64>,
        epoch_jd_tdb: f64,
        final_jd_tdb: f64,
        condition_bodies: Vec<String>,
        condition_distances_km: Vec<f64>,
        condition_stopping: Vec<bool>,
    ) -> PyResult<Bound<'py, PyDict>> {
        if condition_bodies.len() != condition_distances_km.len()
            || condition_bodies.len() != condition_stopping.len()
        {
            return Err(PyValueError::new_err(
                "collision condition arrays must share one length",
            ));
        }
        let mut conditions = Vec::with_capacity(condition_bodies.len());
        for ((code, distance_km), stopping) in condition_bodies
            .iter()
            .zip(condition_distances_km)
            .zip(condition_stopping)
        {
            let body = map_origin_code_to_assist_body(code).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "unsupported collision object code for ASSIST ephemeris: {code}"
                ))
            })?;
            conditions.push(CollisionConditionSpec {
                body,
                distance_km,
                stopping,
            });
        }
        let benchmark = NativeBenchmark::Collisions {
            states: states_from_pyarray(states)?,
            epoch_jd_tdb,
            final_jd_tdb,
            conditions,
        };
        let output = py
            .allow_threads(|| run_collision_benchmark(self, &benchmark))
            .map_err(py_runtime_error)?;
        self.store_benchmark(benchmark)?;

        let dict = PyDict::new_bound(py);
        dict.set_item(
            "final_indices",
            output
                .final_indices
                .iter()
                .map(|&index| index as i64)
                .collect::<Vec<_>>(),
        )?;
        dict.set_item(
            "final_states",
            shaped_states_array(py, &output.final_states)?,
        )?;
        dict.set_item("final_time_jd_tdb", output.final_time_jd_tdb)?;
        dict.set_item(
            "impact_indices",
            output
                .impact_indices
                .iter()
                .map(|&index| index as i64)
                .collect::<Vec<_>>(),
        )?;
        dict.set_item(
            "impact_condition_indices",
            output
                .impact_condition_indices
                .iter()
                .map(|&index| index as i64)
                .collect::<Vec<_>>(),
        )?;
        dict.set_item(
            "impact_states",
            shaped_states_array(py, &output.impact_states)?,
        )?;
        dict.set_item("impact_times_jd_tdb", output.impact_times_jd_tdb.clone())?;
        Ok(dict)
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        orbit_ids,
        object_ids,
        states,
        origin_codes,
        frame,
        time_scale,
        time_days,
        time_nanos,
        target_scale,
        target_days,
        target_nanos,
        covariance,
        covariances=None,
        covariance_method="monte-carlo",
        num_samples=1000,
        seed=None,
        chunk_size=None,
        thread_limit=None,
        variant_ids=None,
        weights=None,
        weights_cov=None
    ))]
    fn propagate_orbits<'py>(
        &self,
        py: Python<'py>,
        orbit_ids: Vec<String>,
        object_ids: Vec<Option<String>>,
        states: PyReadonlyArray2<'py, f64>,
        origin_codes: Vec<String>,
        frame: &str,
        time_scale: &str,
        time_days: Vec<i64>,
        time_nanos: Vec<i64>,
        target_scale: &str,
        target_days: Vec<i64>,
        target_nanos: Vec<i64>,
        covariance: bool,
        covariances: Option<PyReadonlyArray2<'py, f64>>,
        covariance_method: &str,
        num_samples: usize,
        seed: Option<u64>,
        chunk_size: Option<usize>,
        thread_limit: Option<usize>,
        variant_ids: Option<Vec<Option<String>>>,
        weights: Option<Vec<Option<f64>>>,
        weights_cov: Option<Vec<Option<f64>>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let state_rows = states_from_pyarray(states)?;
        let input_times = time_array(time_scale, time_days, time_nanos)?;
        let target_times = time_array(target_scale, target_days, target_nanos)?;
        let covariance_batch = match covariances {
            Some(covariances) => Some(covariance_from_pyarray(covariances, state_rows.len())?),
            None => None,
        };
        let coordinates = CoordinateBatch::cartesian(
            state_rows,
            Frame::parse(frame).map_err(py_value_error)?,
            OriginArray::new(origin_codes.into_iter().map(OriginId::from_code).collect()),
            Some(input_times),
            covariance_batch,
        )
        .map_err(py_value_error)?;
        let options = PropagationOptions {
            chunk_size,
            thread_limit,
            epoch_policy: EpochPolicy::CrossProduct,
            covariance: CovariancePropagation::None,
        };
        let input = match variant_ids {
            Some(variant_ids) => {
                if covariance {
                    return Err(PyValueError::new_err(
                        "covariance=True is not supported for VariantOrbits",
                    ));
                }
                let weights = weights
                    .ok_or_else(|| PyValueError::new_err("variant propagation requires weights"))?;
                let weights_cov = weights_cov.ok_or_else(|| {
                    PyValueError::new_err("variant propagation requires weights_cov")
                })?;
                PreparedPropagationInput::Variants(
                    OrbitVariantBatch::new(
                        orbit_ids.into_iter().map(OrbitId).collect(),
                        object_ids
                            .into_iter()
                            .map(|value| value.map(ObjectId))
                            .collect(),
                        variant_ids
                            .into_iter()
                            .map(|value| value.map(VariantId))
                            .collect(),
                        weights,
                        weights_cov,
                        coordinates,
                    )
                    .map_err(py_value_error)?,
                )
            }
            None => PreparedPropagationInput::Orbits(
                OrbitBatch::new(
                    orbit_ids.into_iter().map(OrbitId).collect(),
                    object_ids
                        .into_iter()
                        .map(|value| value.map(ObjectId))
                        .collect(),
                    coordinates,
                )
                .map_err(py_value_error)?,
            ),
        };
        let benchmark = NativeBenchmark::Propagation {
            input,
            target_times,
            options,
            covariance_method: parse_covariance_method(covariance_method)?,
            num_samples,
            seed,
            sampled_covariance: covariance,
        };
        let result = py
            .allow_threads(|| run_propagation_benchmark(self, &benchmark))
            .map_err(py_runtime_error)?;
        self.store_benchmark(benchmark)?;
        propagation_result_to_dict(py, &result)
    }
}

impl NativeAssistPropagator {
    fn store_benchmark(&self, benchmark: NativeBenchmark) -> PyResult<()> {
        *self
            .benchmark
            .lock()
            .map_err(|_| PyRuntimeError::new_err("native benchmark lock is poisoned"))? =
            Some(benchmark);
        Ok(())
    }
}

fn propagate_with_sampled_covariance_native(
    propagator: &RustAssistPropagator,
    orbits: &OrbitBatch,
    target_times: &TimeArray,
    mut options: PropagationOptions,
    method: OrbitVariantSamplingMethod,
    num_samples: usize,
    seed: Option<u64>,
) -> Result<PropagationResult, String> {
    if orbits.coordinates.covariance.is_none() {
        return Err("covariance=True requires input coordinate covariance rows".to_string());
    }
    options.covariance = CovariancePropagation::None;
    let variant_samples =
        create_sampled_orbit_variants(orbits, method, num_samples, seed, 1.0, 0.0, 0.0)
            .map_err(|err| err.to_string())?;
    let nominal_request = PropagationRequest::new(orbits, target_times, options.clone())
        .map_err(|err| err.to_string())?;
    let nominal = propagator
        .propagate(&nominal_request, &PythonTimeProvider)
        .map_err(|err| err.to_string())?;
    let variant_request =
        PropagationRequest::new_variants(&variant_samples.variants, target_times, options)
            .map_err(|err| err.to_string())?;
    let propagated_variants = propagator
        .propagate(&variant_request, &PythonTimeProvider)
        .map_err(|err| err.to_string())?;
    collapse_propagated_variants_to_orbits(
        &nominal,
        &propagated_variants,
        &variant_samples.source_orbit_indices,
    )
    .map_err(|err| err.to_string())
}

fn run_propagation_benchmark(
    propagator: &NativeAssistPropagator,
    benchmark: &NativeBenchmark,
) -> Result<PropagationResult, String> {
    let NativeBenchmark::Propagation {
        input,
        target_times,
        options,
        covariance_method,
        num_samples,
        seed,
        sampled_covariance,
    } = benchmark
    else {
        return Err("prepared benchmark is not propagation".to_string());
    };
    match input {
        PreparedPropagationInput::Orbits(orbits) if *sampled_covariance => {
            propagate_with_sampled_covariance_native(
                &propagator.inner,
                orbits,
                target_times,
                options.clone(),
                *covariance_method,
                *num_samples,
                *seed,
            )
        }
        PreparedPropagationInput::Orbits(orbits) => {
            let request = PropagationRequest::new(orbits, target_times, options.clone())
                .map_err(|err| err.to_string())?;
            propagator
                .inner
                .propagate(&request, &PythonTimeProvider)
                .map_err(|err| err.to_string())
        }
        PreparedPropagationInput::Variants(variants) => {
            let request = PropagationRequest::new_variants(variants, target_times, options.clone())
                .map_err(|err| err.to_string())?;
            propagator
                .inner
                .propagate(&request, &PythonTimeProvider)
                .map_err(|err| err.to_string())
        }
    }
}

fn run_ephemeris_benchmark(
    propagator: &NativeAssistPropagator,
    benchmark: &NativeBenchmark,
) -> Result<EphemerisResult, String> {
    let NativeBenchmark::Ephemeris {
        orbits,
        observers,
        options,
        covariance_method,
        num_samples,
        seed,
        sampled_covariance,
    } = benchmark
    else {
        return Err("prepared benchmark is not ephemeris".to_string());
    };
    let nominal = propagator
        .inner
        .generate_ephemeris(
            orbits,
            observers,
            options,
            &PythonTimeProvider,
            &propagator.spice,
        )
        .map_err(|err| err.to_string())?;
    if !sampled_covariance {
        return Ok(nominal);
    }
    let variant_samples = create_sampled_orbit_variants(
        orbits,
        *covariance_method,
        *num_samples,
        *seed,
        1.0,
        0.0,
        0.0,
    )
    .map_err(|err| err.to_string())?;
    let variant_orbits = OrbitBatch::new(
        variant_samples.variants.orbit_id.clone(),
        variant_samples.variants.object_id.clone(),
        variant_samples.variants.coordinates.clone(),
    )
    .map_err(|err| err.to_string())?;
    let variant_result = propagator
        .inner
        .generate_ephemeris(
            &variant_orbits,
            observers,
            options,
            &PythonTimeProvider,
            &propagator.spice,
        )
        .map_err(|err| err.to_string())?;
    let weights_cov: Vec<f64> = variant_samples
        .variants
        .weights_cov
        .iter()
        .map(|weight| weight.unwrap_or(0.0))
        .collect();
    collapse_variant_ephemeris(
        &nominal,
        &variant_result,
        &variant_samples.source_orbit_indices,
        &weights_cov,
        observers.coordinates.len(),
    )
    .map_err(|err| err.to_string())
}

fn run_collision_benchmark(
    propagator: &NativeAssistPropagator,
    benchmark: &NativeBenchmark,
) -> Result<CollisionDetectionOutput, String> {
    let NativeBenchmark::Collisions {
        states,
        epoch_jd_tdb,
        final_jd_tdb,
        conditions,
    } = benchmark
    else {
        return Err("prepared benchmark is not collision detection".to_string());
    };
    propagator
        .inner
        .detect_collisions_same_epoch(states, *epoch_jd_tdb, *final_jd_tdb, conditions)
        .map_err(|err| err.to_string())
}

fn run_native_benchmark(
    propagator: &NativeAssistPropagator,
    benchmark: &NativeBenchmark,
) -> Result<(), String> {
    match benchmark {
        NativeBenchmark::Propagation { .. } => {
            black_box(run_propagation_benchmark(propagator, benchmark)?);
        }
        NativeBenchmark::Ephemeris { .. } => {
            black_box(run_ephemeris_benchmark(propagator, benchmark)?);
        }
        NativeBenchmark::Collisions { .. } => {
            black_box(run_collision_benchmark(propagator, benchmark)?);
        }
        NativeBenchmark::FitEvaluated {
            problem,
            ignore,
            config,
        } => {
            black_box(run_fit_evaluated(propagator, problem, ignore, config)?);
        }
        NativeBenchmark::OdFit { problem, config } => {
            black_box(run_od_fit(propagator, problem, config)?);
        }
        NativeBenchmark::Vallado { problem, config } => {
            black_box(run_vallado(propagator, problem, config)?);
        }
        NativeBenchmark::Iod {
            problem,
            linkage_rows,
            chunk_size,
            config,
        } => {
            black_box(run_iod_linkages(
                propagator,
                problem,
                linkage_rows,
                *chunk_size,
                config,
            )?);
        }
    }
    Ok(())
}

fn run_fit_evaluated(
    propagator: &NativeAssistPropagator,
    problem: &OdProblem,
    ignore: &[bool],
    config: &LeastSquaresConfig,
) -> Result<EvaluatedLeastSquaresFit, String> {
    fit_orbit_least_squares_evaluated_barycentric(
        &propagator.inner,
        &problem.orbit,
        &problem.observed,
        &problem.observers,
        ignore,
        &problem.options,
        config,
        &PythonTimeProvider,
        &propagator.spice,
    )
    .map_err(|err| err.to_string())
}

fn run_od_fit(
    propagator: &NativeAssistPropagator,
    problem: &OdProblem,
    config: &OdConfig,
) -> Result<OdOutput, String> {
    od_fit_barycentric(
        &propagator.inner,
        &problem.orbit,
        &problem.observed,
        &problem.observers,
        config,
        &problem.options,
        &PythonTimeProvider,
        &propagator.spice,
    )
    .map_err(|err| err.to_string())
}

fn run_vallado(
    propagator: &NativeAssistPropagator,
    problem: &OdProblem,
    config: &ValladoConfig,
) -> Result<ValladoResult, String> {
    vallado_least_squares_barycentric(
        &propagator.inner,
        &problem.orbit,
        &problem.observed,
        &problem.observers,
        config,
        &problem.options,
        &PythonTimeProvider,
        &propagator.spice,
    )
    .map_err(|err| err.to_string())
}

fn run_iod_linkages(
    propagator: &NativeAssistPropagator,
    problem: &IodProblem,
    linkage_rows: &[Vec<usize>],
    chunk_size: usize,
    config: &IodConfig,
) -> Result<Vec<IodOutput>, String> {
    iod_fit_linkages_barycentric(
        &propagator.inner,
        &problem.observed,
        &problem.observers,
        linkage_rows,
        chunk_size,
        config,
        &problem.options,
        &PythonTimeProvider,
        &propagator.spice,
    )
    .map_err(|err| err.to_string())
}

fn parse_observation_selection_method(method: &str) -> PyResult<ObservationSelectionMethod> {
    match method {
        "combinations" => Ok(ObservationSelectionMethod::Combinations),
        "first+middle+last" => Ok(ObservationSelectionMethod::FirstMiddleLast),
        "thirds" => Ok(ObservationSelectionMethod::Thirds),
        _ => Err(PyValueError::new_err(
            "method should be one of {'first+middle+last', 'thirds'}",
        )),
    }
}

fn parse_od_method(method: &str) -> PyResult<OdMethod> {
    match method {
        "central" => Ok(OdMethod::Central),
        "finite" => Ok(OdMethod::Finite),
        _ => Err(PyValueError::new_err(
            "method should be one of 'central' or 'finite'.",
        )),
    }
}

#[allow(clippy::too_many_arguments)]
fn build_observation_problem(
    observed_values: PyReadonlyArray2<'_, f64>,
    observed_covariances: PyReadonlyArray2<'_, f64>,
    observed_origin_codes: Vec<String>,
    observed_frame: &str,
    observed_time_scale: &str,
    observed_time_days: Vec<i64>,
    observed_time_nanos: Vec<i64>,
    observer_codes: Vec<String>,
    observer_states: PyReadonlyArray2<'_, f64>,
    observer_origin_codes: Vec<String>,
    observer_frame: &str,
    observer_time_scale: &str,
    observer_time_days: Vec<i64>,
    observer_time_nanos: Vec<i64>,
    lt_tol: f64,
    eph_max_iter: usize,
    eph_tol: f64,
    stellar_aberration: bool,
    max_lt_iter: usize,
) -> PyResult<(CoordinateBatch, ObserverBatch, EphemerisOptions)> {
    let observed_rows = states_from_pyarray(observed_values)?;
    let n = observed_rows.len();
    let observed_cov = observed_covariances.as_array();
    if observed_cov.nrows() != n || observed_cov.ncols() != 36 {
        return Err(PyValueError::new_err(
            "observed_covariances must have shape (N, 36)",
        ));
    }
    let observed = CoordinateBatch::new(
        CoordinateValues::Spherical(observed_rows),
        Frame::parse(observed_frame).map_err(py_value_error)?,
        OriginArray::new(
            observed_origin_codes
                .into_iter()
                .map(OriginId::from_code)
                .collect(),
        ),
        Some(time_array(
            observed_time_scale,
            observed_time_days,
            observed_time_nanos,
        )?),
        Some(
            CovarianceBatch::new(
                n,
                6,
                observed_cov.iter().copied().collect(),
                CovarianceUnits::Coordinate(CoordinateRepresentation::Spherical),
            )
            .map_err(py_value_error)?,
        ),
    )
    .map_err(py_value_error)?;
    let observer_times = time_array(observer_time_scale, observer_time_days, observer_time_nanos)?;
    let observer_coordinates = CoordinateBatch::cartesian(
        states_from_pyarray(observer_states)?,
        Frame::parse(observer_frame).map_err(py_value_error)?,
        OriginArray::new(
            observer_origin_codes
                .into_iter()
                .map(OriginId::from_code)
                .collect(),
        ),
        Some(observer_times),
        None,
    )
    .map_err(py_value_error)?;
    let observers = ObserverBatch::new(
        observer_codes.into_iter().map(ObservatoryCode).collect(),
        observer_coordinates,
    )
    .map_err(py_value_error)?;
    let options = EphemerisOptions {
        propagation: PropagationOptions {
            chunk_size: None,
            thread_limit: None,
            epoch_policy: EpochPolicy::CrossProduct,
            covariance: CovariancePropagation::None,
        },
        lt_tol,
        max_iter: eph_max_iter,
        tol: eph_tol,
        stellar_aberration,
        max_lt_iter,
        output_time_scale: TimeScale::parse(observer_time_scale).map_err(py_value_error)?,
        include_aberrated_coordinates: false,
        photometry: EphemerisPhotometryOptions::default(),
    };
    Ok((observed, observers, options))
}

#[allow(clippy::too_many_arguments)]
fn build_iod_problem(
    observed_values: PyReadonlyArray2<'_, f64>,
    observed_covariances: PyReadonlyArray2<'_, f64>,
    observed_origin_codes: Vec<String>,
    observed_frame: &str,
    observed_time_scale: &str,
    observed_time_days: Vec<i64>,
    observed_time_nanos: Vec<i64>,
    observer_codes: Vec<String>,
    observer_states: PyReadonlyArray2<'_, f64>,
    observer_origin_codes: Vec<String>,
    observer_frame: &str,
    observer_time_scale: &str,
    observer_time_days: Vec<i64>,
    observer_time_nanos: Vec<i64>,
    lt_tol: f64,
    eph_max_iter: usize,
    eph_tol: f64,
    stellar_aberration: bool,
    max_lt_iter: usize,
) -> PyResult<IodProblem> {
    let (observed, observers, options) = build_observation_problem(
        observed_values,
        observed_covariances,
        observed_origin_codes,
        observed_frame,
        observed_time_scale,
        observed_time_days,
        observed_time_nanos,
        observer_codes,
        observer_states,
        observer_origin_codes,
        observer_frame,
        observer_time_scale,
        observer_time_days,
        observer_time_nanos,
        lt_tol,
        eph_max_iter,
        eph_tol,
        stellar_aberration,
        max_lt_iter,
    )?;
    Ok(IodProblem {
        observed,
        observers,
        options,
    })
}

/// Marshal the shared raw OD-problem arguments into typed batches. This is
/// the exact input contract of `fit_orbit_least_squares`, factored so the
/// fused fit/od/Vallado work units share one builder.
#[allow(clippy::too_many_arguments)]
fn build_od_problem(
    orbit_ids: Vec<String>,
    object_ids: Vec<Option<String>>,
    orbit_states: PyReadonlyArray2<'_, f64>,
    orbit_origin_codes: Vec<String>,
    orbit_frame: &str,
    orbit_time_scale: &str,
    orbit_time_days: Vec<i64>,
    orbit_time_nanos: Vec<i64>,
    observed_values: PyReadonlyArray2<'_, f64>,
    observed_covariances: PyReadonlyArray2<'_, f64>,
    observer_codes: Vec<String>,
    observer_states: PyReadonlyArray2<'_, f64>,
    observer_origin_codes: Vec<String>,
    observer_frame: &str,
    observer_time_scale: &str,
    observer_time_days: Vec<i64>,
    observer_time_nanos: Vec<i64>,
    lt_tol: f64,
    eph_max_iter: usize,
    eph_tol: f64,
    stellar_aberration: bool,
    max_lt_iter: usize,
) -> PyResult<OdProblem> {
    let orbit_coordinates = CoordinateBatch::cartesian(
        states_from_pyarray(orbit_states)?,
        Frame::parse(orbit_frame).map_err(py_value_error)?,
        OriginArray::new(
            orbit_origin_codes
                .into_iter()
                .map(OriginId::from_code)
                .collect(),
        ),
        Some(time_array(
            orbit_time_scale,
            orbit_time_days,
            orbit_time_nanos,
        )?),
        None,
    )
    .map_err(py_value_error)?;
    let orbit = OrbitBatch::new(
        orbit_ids.into_iter().map(OrbitId).collect(),
        object_ids
            .into_iter()
            .map(|value| value.map(ObjectId))
            .collect(),
        orbit_coordinates,
    )
    .map_err(py_value_error)?;

    // The existing OD contract labels observed rows with the observer frame,
    // origins, and times; preserve that established behavior while sharing
    // the batch/options builder with the IOD contract (which supplies the
    // observed metadata explicitly).
    let (observed, observers, options) = build_observation_problem(
        observed_values,
        observed_covariances,
        observer_origin_codes.clone(),
        observer_frame,
        observer_time_scale,
        observer_time_days.clone(),
        observer_time_nanos.clone(),
        observer_codes,
        observer_states,
        observer_origin_codes,
        observer_frame,
        observer_time_scale,
        observer_time_days,
        observer_time_nanos,
        lt_tol,
        eph_max_iter,
        eph_tol,
        stellar_aberration,
        max_lt_iter,
    )?;
    Ok(OdProblem {
        orbit,
        observed,
        observers,
        options,
    })
}

fn shaped_matrix_array<'py>(
    py: Python<'py>,
    values: &[f64],
    rows: usize,
    cols: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let shaped = ndarray::Array2::from_shape_vec((rows, cols), values.to_vec())
        .map_err(|err| PyRuntimeError::new_err(format!("failed to shape output: {err}")))?;
    Ok(shaped.into_pyarray_bound(py))
}

fn evaluation_into_dict(
    dict: &Bound<'_, PyDict>,
    py: Python<'_>,
    evaluation: &adam_core_rs_coords::propagation::FitEvaluation,
) -> PyResult<()> {
    let n = evaluation.chi2.len();
    dict.set_item(
        "residual_values",
        shaped_matrix_array(py, &evaluation.residuals, n, 6)?,
    )?;
    dict.set_item("residual_chi2", evaluation.chi2.clone())?;
    dict.set_item("residual_dof", evaluation.dof.clone())?;
    dict.set_item("residual_probability", evaluation.probability.clone())?;
    dict.set_item("chi2", evaluation.orbit_chi2)?;
    dict.set_item("reduced_chi2", evaluation.reduced_chi2)?;
    dict.set_item("arc_length", evaluation.arc_length)?;
    dict.set_item("num_obs", evaluation.num_obs)?;
    dict.set_item("outlier", evaluation.outlier.clone())?;
    Ok(())
}

fn parse_covariance_method(value: &str) -> PyResult<OrbitVariantSamplingMethod> {
    match value {
        "auto" => Ok(OrbitVariantSamplingMethod::Auto),
        "sigma-point" => Ok(OrbitVariantSamplingMethod::SigmaPoint),
        "monte-carlo" => Ok(OrbitVariantSamplingMethod::MonteCarlo),
        other => Err(PyValueError::new_err(format!(
            "covariance_method must be one of 'auto', 'sigma-point', or 'monte-carlo'; got {other:?}"
        ))),
    }
}

fn parse_adaptive_mode(value: i32) -> PyResult<Ias15AdaptiveMode> {
    match value {
        0 => Ok(Ias15AdaptiveMode::Individual),
        1 => Ok(Ias15AdaptiveMode::Global),
        2 => Ok(Ias15AdaptiveMode::Prs23),
        3 => Ok(Ias15AdaptiveMode::Aarseth85),
        _ => Err(PyValueError::new_err(format!(
            "adaptive_mode must be one of 0, 1, 2, or 3; got {value}"
        ))),
    }
}

fn shaped_states_array<'py>(
    py: Python<'py>,
    states: &[[f64; 6]],
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let flat: Vec<f64> = states
        .iter()
        .flat_map(|state| state.iter().copied())
        .collect();
    let shaped = ndarray::Array2::from_shape_vec((states.len(), 6), flat)
        .map_err(|err| PyRuntimeError::new_err(format!("failed to shape states: {err}")))?;
    Ok(shaped.into_pyarray_bound(py))
}

fn states_from_pyarray(states: PyReadonlyArray2<'_, f64>) -> PyResult<Vec<[f64; 6]>> {
    let array = states.as_array();
    if array.ncols() != 6 {
        return Err(PyValueError::new_err(format!(
            "states must have shape (N, 6); got ({}, {})",
            array.nrows(),
            array.ncols()
        )));
    }
    let mut rows = Vec::with_capacity(array.nrows());
    for row in array.rows() {
        rows.push([row[0], row[1], row[2], row[3], row[4], row[5]]);
    }
    Ok(rows)
}

fn time_array(scale: &str, days: Vec<i64>, nanos: Vec<i64>) -> PyResult<TimeArray> {
    TimeArray::from_parts(
        TimeScale::parse(scale).map_err(py_value_error)?,
        days,
        nanos,
    )
    .map_err(py_value_error)
}

fn covariance_from_pyarray(
    covariances: PyReadonlyArray2<'_, f64>,
    rows: usize,
) -> PyResult<CovarianceBatch> {
    let array = covariances.as_array();
    if array.nrows() != rows || array.ncols() != 36 {
        return Err(PyValueError::new_err(format!(
            "covariances must have shape ({rows}, 36); got ({}, {})",
            array.nrows(),
            array.ncols()
        )));
    }
    CovarianceBatch::new(
        rows,
        6,
        array.iter().copied().collect(),
        CovarianceUnits::Coordinate(CoordinateRepresentation::Cartesian),
    )
    .map_err(py_value_error)
}

fn propagation_result_to_dict<'py>(
    py: Python<'py>,
    result: &adam_core_rs_coords::propagation::PropagationResult,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new_bound(py);
    let (orbit_ids, object_ids, variant_ids, weights, weights_cov, coordinates) =
        if let Some(variants) = &result.variants {
            (
                variants
                    .orbit_id
                    .iter()
                    .map(|value| value.0.clone())
                    .collect::<Vec<_>>(),
                variants
                    .object_id
                    .iter()
                    .map(|value| value.as_ref().map(|item| item.0.clone()))
                    .collect::<Vec<_>>(),
                Some(
                    variants
                        .variant_id
                        .iter()
                        .map(|value| value.as_ref().map(|item| item.0.clone()))
                        .collect::<Vec<_>>(),
                ),
                Some(variants.weights.clone()),
                Some(variants.weights_cov.clone()),
                &variants.coordinates,
            )
        } else {
            (
                result
                    .orbits
                    .orbit_id
                    .iter()
                    .map(|value| value.0.clone())
                    .collect::<Vec<_>>(),
                result
                    .orbits
                    .object_id
                    .iter()
                    .map(|value| value.as_ref().map(|item| item.0.clone()))
                    .collect::<Vec<_>>(),
                None,
                None,
                None,
                &result.orbits.coordinates,
            )
        };

    let states = coordinates.values.cartesian().ok_or_else(|| {
        PyRuntimeError::new_err("assist propagation returned non-Cartesian coordinates")
    })?;
    let flat_states = states
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect::<Vec<_>>();
    let shaped_states = ndarray::Array2::from_shape_vec((states.len(), 6), flat_states)
        .map_err(|err| PyRuntimeError::new_err(format!("failed to shape states: {err}")))?;
    let times = coordinates.times.as_ref().ok_or_else(|| {
        PyRuntimeError::new_err("assist propagation returned coordinates without times")
    })?;
    let input_orbit_indices = result
        .diagnostics
        .convergence
        .iter()
        .map(|row| row.input_orbit_index)
        .collect::<Vec<_>>();
    let validity = (0..result.validity.len())
        .map(|index| result.validity.is_valid(index))
        .collect::<Vec<_>>();
    let messages = result
        .diagnostics
        .convergence
        .iter()
        .map(|row| row.message.clone())
        .collect::<Vec<_>>();

    dict.set_item("orbit_id", orbit_ids)?;
    dict.set_item("object_id", object_ids)?;
    dict.set_item("variant_id", variant_ids)?;
    dict.set_item("weights", weights)?;
    dict.set_item("weights_cov", weights_cov)?;
    dict.set_item("states", shaped_states.into_pyarray_bound(py))?;
    match &coordinates.covariance {
        Some(covariance) => {
            let shaped_covariance = ndarray::Array2::from_shape_vec(
                (covariance.rows, covariance.dimension * covariance.dimension),
                covariance.values_row_major.clone(),
            )
            .map_err(|err| PyRuntimeError::new_err(format!("failed to shape covariance: {err}")))?;
            dict.set_item("covariances", shaped_covariance.into_pyarray_bound(py))?;
        }
        None => dict.set_item("covariances", py.None())?,
    }
    dict.set_item(
        "origin_codes",
        coordinates
            .origins
            .origins
            .iter()
            .map(OriginId::code)
            .collect::<Vec<_>>(),
    )?;
    dict.set_item("frame", coordinates.frame.as_str())?;
    dict.set_item("time_scale", times.scale.as_str())?;
    dict.set_item(
        "time_days",
        times
            .epochs
            .iter()
            .map(|epoch| epoch.days)
            .collect::<Vec<_>>(),
    )?;
    dict.set_item(
        "time_nanos",
        times
            .epochs
            .iter()
            .map(|epoch| epoch.nanos)
            .collect::<Vec<_>>(),
    )?;
    dict.set_item("input_orbit_indices", input_orbit_indices)?;
    dict.set_item("validity", validity)?;
    dict.set_item("messages", messages)?;
    Ok(dict)
}

fn ephemeris_result_to_dict<'py>(
    py: Python<'py>,
    result: &EphemerisResult,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new_bound(py);
    let ephemeris = &result.ephemeris;
    dict.set_item(
        "orbit_id",
        ephemeris
            .orbit_id
            .iter()
            .map(|value| value.0.clone())
            .collect::<Vec<_>>(),
    )?;
    dict.set_item(
        "object_id",
        ephemeris
            .object_id
            .iter()
            .map(|value| value.as_ref().map(|item| item.0.clone()))
            .collect::<Vec<_>>(),
    )?;
    let coordinates = &ephemeris.coordinates;
    let states = coordinates
        .values
        .spherical()
        .ok_or_else(|| PyRuntimeError::new_err("ephemeris coordinates must be spherical"))?;
    let flat = states
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect::<Vec<_>>();
    let shaped = ndarray::Array2::from_shape_vec((states.len(), 6), flat).map_err(|err| {
        PyRuntimeError::new_err(format!("failed to shape ephemeris states: {err}"))
    })?;
    dict.set_item("states", shaped.into_pyarray_bound(py))?;
    match &coordinates.covariance {
        Some(cov) => {
            let cov_shaped = ndarray::Array2::from_shape_vec(
                (coordinates.len(), 36),
                cov.values_row_major.clone(),
            )
            .map_err(|err| {
                PyRuntimeError::new_err(format!("failed to shape ephemeris covariance: {err}"))
            })?;
            dict.set_item("covariance", cov_shaped.into_pyarray_bound(py))?;
        }
        None => {
            dict.set_item("covariance", py.None())?;
        }
    }
    dict.set_item(
        "origin_codes",
        coordinates
            .origins
            .origins
            .iter()
            .map(OriginId::code)
            .collect::<Vec<_>>(),
    )?;
    dict.set_item("frame", coordinates.frame.as_str())?;
    let times = coordinates
        .times
        .as_ref()
        .ok_or_else(|| PyRuntimeError::new_err("ephemeris coordinates missing times"))?;
    dict.set_item("time_scale", times.scale.as_str())?;
    dict.set_item(
        "time_days",
        times
            .epochs
            .iter()
            .map(|epoch| epoch.days)
            .collect::<Vec<_>>(),
    )?;
    dict.set_item(
        "time_nanos",
        times
            .epochs
            .iter()
            .map(|epoch| epoch.nanos)
            .collect::<Vec<_>>(),
    )?;
    dict.set_item("light_time", ephemeris.light_time_days.clone())?;
    match &ephemeris.alpha_deg {
        Some(values) => dict.set_item("alpha", values.clone())?,
        None => dict.set_item("alpha", py.None())?,
    }
    match &ephemeris.predicted_magnitude_v {
        Some(values) => dict.set_item("predicted_magnitude_v", values.clone())?,
        None => dict.set_item("predicted_magnitude_v", py.None())?,
    }
    match &ephemeris.aberrated_coordinates {
        Some(aberrated) => {
            let aberrated_states = aberrated.values.cartesian().ok_or_else(|| {
                PyRuntimeError::new_err("aberrated coordinates must be Cartesian")
            })?;
            let aberrated_flat = aberrated_states
                .iter()
                .flat_map(|row| row.iter().copied())
                .collect::<Vec<_>>();
            let aberrated_shaped =
                ndarray::Array2::from_shape_vec((aberrated_states.len(), 6), aberrated_flat)
                    .map_err(|err| {
                        PyRuntimeError::new_err(format!("failed to shape aberrated states: {err}"))
                    })?;
            dict.set_item("aberrated_states", aberrated_shaped.into_pyarray_bound(py))?;
            dict.set_item(
                "aberrated_origin_codes",
                aberrated
                    .origins
                    .origins
                    .iter()
                    .map(OriginId::code)
                    .collect::<Vec<_>>(),
            )?;
            let aberrated_times = aberrated
                .times
                .as_ref()
                .ok_or_else(|| PyRuntimeError::new_err("aberrated coordinates missing times"))?;
            dict.set_item("aberrated_time_scale", aberrated_times.scale.as_str())?;
            dict.set_item(
                "aberrated_time_days",
                aberrated_times
                    .epochs
                    .iter()
                    .map(|epoch| epoch.days)
                    .collect::<Vec<_>>(),
            )?;
            dict.set_item(
                "aberrated_time_nanos",
                aberrated_times
                    .epochs
                    .iter()
                    .map(|epoch| epoch.nanos)
                    .collect::<Vec<_>>(),
            )?;
            match &aberrated.covariance {
                Some(cov) => {
                    let cov_shaped = ndarray::Array2::from_shape_vec(
                        (aberrated.len(), 36),
                        cov.values_row_major.clone(),
                    )
                    .map_err(|err| {
                        PyRuntimeError::new_err(format!(
                            "failed to shape aberrated covariance: {err}"
                        ))
                    })?;
                    dict.set_item("aberrated_covariance", cov_shaped.into_pyarray_bound(py))?;
                }
                None => {
                    dict.set_item("aberrated_covariance", py.None())?;
                }
            }
        }
        None => {
            dict.set_item("aberrated_states", py.None())?;
            dict.set_item("aberrated_origin_codes", py.None())?;
            dict.set_item("aberrated_time_scale", py.None())?;
            dict.set_item("aberrated_time_days", py.None())?;
            dict.set_item("aberrated_time_nanos", py.None())?;
            dict.set_item("aberrated_covariance", py.None())?;
        }
    }
    let validity = (0..ephemeris.validity.len())
        .map(|index| ephemeris.validity.is_valid(index))
        .collect::<Vec<_>>();
    dict.set_item("validity", validity)?;
    Ok(dict)
}

fn py_value_error(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn py_runtime_error(error: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NativeAssistPropagator>()?;
    Ok(())
}
