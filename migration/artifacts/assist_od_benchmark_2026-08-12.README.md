# Frozen historical ASSIST OD benchmark

`assist_od_benchmark_2026-08-12.json` is immutable migration evidence produced
before Core 0.7 moved IOD, fitting, rejection, and workflow orchestration out of
adam-assist. It records a legacy two-runtime comparison and is not a current benchmark,
release result, or compatibility claim.

The former generator `migration/scripts/benchmark_assist_od.py` was retired
because it called Assist-owned OD methods removed in 0.6.0. Historical oracle
request handling remains isolated under `migration/parity` so old evidence can
be audited; it is not imported by the shipped wheel or crate. Current benchmark
CI uses `migration.scripts.benchmark_current` and measures only the public
propagation, ephemeris, covariance, collision, non-gravitational, and native
adapter surfaces that adam-assist still owns.
