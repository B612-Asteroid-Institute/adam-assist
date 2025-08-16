window.BENCHMARK_DATA = {
  "lastUpdate": 1755303310127,
  "repoUrl": "https://github.com/B612-Asteroid-Institute/adam-assist",
  "entries": {
    "Python Benchmark": [
      {
        "commit": {
          "author": {
            "email": "akoumjian@users.noreply.github.com",
            "name": "Alec Koumjian",
            "username": "akoumjian"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "8d911ba8e384a3ef4dab20622e91771ddd18ad7e",
          "message": "Performance improvements",
          "timestamp": "2025-08-15T19:43:43-04:00",
          "tree_id": "50ddfc5bf10f5b2af9540de436999b8cc5a2a059",
          "url": "https://github.com/B612-Asteroid-Institute/adam-assist/commit/8d911ba8e384a3ef4dab20622e91771ddd18ad7e"
        },
        "date": 1755303309872,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_benchmark_propagation_vs_raw",
            "value": 0.2818389929739133,
            "unit": "iter/sec",
            "range": "stddev: 0.014786022342721463",
            "extra": "mean: 3.548125081800015 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_benchmark_ephemeris_generation",
            "value": 0.17616769319902667,
            "unit": "iter/sec",
            "range": "stddev: 0.01640986277978188",
            "extra": "mean: 5.676409685800013 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_benchmark_impact_detection",
            "value": 0.36027815088447257,
            "unit": "iter/sec",
            "range": "stddev: 0.018273979154180282",
            "extra": "mean: 2.775633208800002 sec\nrounds: 5"
          },
          {
            "name": "tests/test_impacts.py::test_calculate_impacts_benchmark_some_impacts[1]",
            "value": 0.36277763085948805,
            "unit": "iter/sec",
            "range": "stddev: 0.015800206603244608",
            "extra": "mean: 2.756509539000001 sec\nrounds: 5"
          },
          {
            "name": "tests/test_impacts.py::test_calculate_impacts_benchmark_some_impacts[2]",
            "value": 0.4980123857552879,
            "unit": "iter/sec",
            "range": "stddev: 0.01850533257249888",
            "extra": "mean: 2.007982187999994 sec\nrounds: 5"
          },
          {
            "name": "tests/test_impacts.py::test_calculate_impacts_benchmark_impacts[1]",
            "value": 1.7275020709502553,
            "unit": "iter/sec",
            "range": "stddev: 0.005846902268758495",
            "extra": "mean: 578.8705071999857 msec\nrounds: 5"
          },
          {
            "name": "tests/test_impacts.py::test_calculate_impacts_benchmark_impacts[2]",
            "value": 2.2804134144330175,
            "unit": "iter/sec",
            "range": "stddev: 0.0038318849500051967",
            "extra": "mean: 438.5169784000027 msec\nrounds: 5"
          },
          {
            "name": "tests/test_impacts.py::test_calculate_impacts_benchmark_no_impacts[1]",
            "value": 0.018696714766740462,
            "unit": "iter/sec",
            "range": "stddev: 0.09714939491964657",
            "extra": "mean: 53.48533218139998 sec\nrounds: 5"
          },
          {
            "name": "tests/test_impacts.py::test_calculate_impacts_benchmark_no_impacts[2]",
            "value": 0.035855722995466785,
            "unit": "iter/sec",
            "range": "stddev: 0.070699692202225",
            "extra": "mean: 27.88955057820001 sec\nrounds: 5"
          }
        ]
      }
    ]
  }
}