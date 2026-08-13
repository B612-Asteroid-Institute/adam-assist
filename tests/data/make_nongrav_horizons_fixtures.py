"""
Build the frozen JPL Horizons vectors for tests/test_nongrav_horizons.py.

Queries heliocentric ecliptic state vectors for the three non-gravitational
regression targets at their central epoch +/- [300, 150, 30, 0] days and
writes them to nongrav_horizons_vectors.parquet next to this script. Run it
only to refresh the fixture (e.g. after a JPL solution update), then update
the retrieval date, solution ids, and SBDB parameters recorded in the test.

    python tests/data/make_nongrav_horizons_fixtures.py

The central epochs put the +/-300 d window over each object's perihelion
(or, for Apophis, a quiet stretch matching the existing live-test epoch).
"""

import os

import pyarrow as pa
import pyarrow.compute as pc
from adam_core.orbits.query.horizons import query_horizons
from adam_core.time import Timestamp
from quivr.concat import concatenate

OFFSETS = [-300, -150, -30, 0, 30, 150, 300]

# (Horizons target id, central epoch MJD TDB)
CASES = [
    ("99942", 60000.0),  # Apophis: matches the live suite's epoch
    ("C/2022 E3", 59956.0),  # perihelion 2023-01-12
    ("1I", 58005.0),  # 'Oumuamua: perihelion 2017-09-09
]


def main() -> None:
    results = []
    for target, t0 in CASES:
        times = Timestamp.from_mjd(
            pc.add(pa.scalar(t0), pa.array(OFFSETS, type=pa.float64())),
            scale="tdb",
        )
        vectors = query_horizons([target], times)
        print(f"{target}: {len(vectors)} states at {t0} + {OFFSETS}")
        results.append(vectors)

    fixture = concatenate(results)
    path = os.path.join(os.path.dirname(__file__), "nongrav_horizons_vectors.parquet")
    fixture.to_parquet(path)
    print(f"Wrote {len(fixture)} rows to {path}")


if __name__ == "__main__":
    main()
