window.BENCHMARK_DATA = {
  "lastUpdate": 1759944129540,
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
      },
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
          "id": "3b36a9c2c8294d7dc89577974ac4e33685a16a14",
          "message": "Performance Improvements (#26)\n\n* Removing uneeded dataframes\n\n* adding optimized single orbit function\n\n* Changing hashing to removing string splitting\n\n\n---------\n\nCo-authored-by: Kathleen Kiker <kathleen@b612foundation.org>\nCo-authored-by: Kathleen Kiker <72056544+KatKiker@users.noreply.github.com>",
          "timestamp": "2025-09-22T15:09:50-04:00",
          "tree_id": "182c4537d41834adfbcbb9ea5dd66691e82f5f2a",
          "url": "https://github.com/B612-Asteroid-Institute/adam-assist/commit/3b36a9c2c8294d7dc89577974ac4e33685a16a14"
        },
        "date": 1758569906324,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_benchmark_propagation_vs_raw",
            "value": 0.2639526857390578,
            "unit": "iter/sec",
            "range": "stddev: 0.033222090867365314",
            "extra": "mean: 3.7885577757999953 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_benchmark_ephemeris_generation",
            "value": 0.27017977850432223,
            "unit": "iter/sec",
            "range": "stddev: 0.03430061126546064",
            "extra": "mean: 3.701239247199999 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_benchmark_impact_detection",
            "value": 0.4725287505463856,
            "unit": "iter/sec",
            "range": "stddev: 0.0024985787416158213",
            "extra": "mean: 2.1162733460000025 sec\nrounds: 5"
          },
          {
            "name": "tests/test_impacts.py::test_calculate_impacts_benchmark_some_impacts[1]",
            "value": 0.4694849722539746,
            "unit": "iter/sec",
            "range": "stddev: 0.025112879088201347",
            "extra": "mean: 2.1299936293999964 sec\nrounds: 5"
          },
          {
            "name": "tests/test_impacts.py::test_calculate_impacts_benchmark_some_impacts[2]",
            "value": 0.6774940569069519,
            "unit": "iter/sec",
            "range": "stddev: 0.003925756385831314",
            "extra": "mean: 1.476027708000015 sec\nrounds: 5"
          },
          {
            "name": "tests/test_impacts.py::test_calculate_impacts_benchmark_impacts[1]",
            "value": 2.004664673473516,
            "unit": "iter/sec",
            "range": "stddev: 0.003333829316969218",
            "extra": "mean: 498.83654519999254 msec\nrounds: 5"
          },
          {
            "name": "tests/test_impacts.py::test_calculate_impacts_benchmark_impacts[2]",
            "value": 2.7883046160022933,
            "unit": "iter/sec",
            "range": "stddev: 0.0052677427633582435",
            "extra": "mean: 358.6408723999966 msec\nrounds: 5"
          },
          {
            "name": "tests/test_impacts.py::test_calculate_impacts_benchmark_no_impacts[1]",
            "value": 0.021252097055554326,
            "unit": "iter/sec",
            "range": "stddev: 0.029170416624289083",
            "extra": "mean: 47.05417998919997 sec\nrounds: 5"
          },
          {
            "name": "tests/test_impacts.py::test_calculate_impacts_benchmark_no_impacts[2]",
            "value": 0.043282291880129435,
            "unit": "iter/sec",
            "range": "stddev: 0.04212338607365955",
            "extra": "mean: 23.10413697060003 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "moeyensj@users.noreply.github.com",
            "name": "Joachim Moeyens",
            "username": "moeyensj"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "04014c6145521d6599c838e99ba854dd1974879f",
          "message": "Hot fix: Correctly stack weights for variants in single orbit optimized case (#27)",
          "timestamp": "2025-10-08T09:52:45-07:00",
          "tree_id": "f6168cfb36e8b8708c44dfa231425cf96d99d97b",
          "url": "https://github.com/B612-Asteroid-Institute/adam-assist/commit/04014c6145521d6599c838e99ba854dd1974879f"
        },
        "date": 1759944128794,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_benchmark_propagation_vs_raw",
            "value": 0.2609378525444893,
            "unit": "iter/sec",
            "range": "stddev: 0.00591067424722134",
            "extra": "mean: 3.832330151599996 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_benchmark_ephemeris_generation",
            "value": 0.2605506514058546,
            "unit": "iter/sec",
            "range": "stddev: 0.024783881424884253",
            "extra": "mean: 3.8380253305999985 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_benchmark_impact_detection",
            "value": 0.4572806345960656,
            "unit": "iter/sec",
            "range": "stddev: 0.015465702680704687",
            "extra": "mean: 2.1868409120000023 sec\nrounds: 5"
          },
          {
            "name": "tests/test_impacts.py::test_calculate_impacts_benchmark_some_impacts[1]",
            "value": 0.4519280527925962,
            "unit": "iter/sec",
            "range": "stddev: 0.009620669746837889",
            "extra": "mean: 2.2127415942000197 sec\nrounds: 5"
          },
          {
            "name": "tests/test_impacts.py::test_calculate_impacts_benchmark_some_impacts[2]",
            "value": 0.6559902426646864,
            "unit": "iter/sec",
            "range": "stddev: 0.011978843906961405",
            "extra": "mean: 1.524412917999996 sec\nrounds: 5"
          },
          {
            "name": "tests/test_impacts.py::test_calculate_impacts_benchmark_impacts[1]",
            "value": 1.9841516946891786,
            "unit": "iter/sec",
            "range": "stddev: 0.0015677325841989742",
            "extra": "mean: 503.99372320000566 msec\nrounds: 5"
          },
          {
            "name": "tests/test_impacts.py::test_calculate_impacts_benchmark_impacts[2]",
            "value": 2.720076769618902,
            "unit": "iter/sec",
            "range": "stddev: 0.007618752348717668",
            "extra": "mean: 367.6366825999935 msec\nrounds: 5"
          },
          {
            "name": "tests/test_impacts.py::test_calculate_impacts_benchmark_no_impacts[1]",
            "value": 0.02079553304060629,
            "unit": "iter/sec",
            "range": "stddev: 0.04050047723717871",
            "extra": "mean: 48.0872501824 sec\nrounds: 5"
          },
          {
            "name": "tests/test_impacts.py::test_calculate_impacts_benchmark_no_impacts[2]",
            "value": 0.04238862064987924,
            "unit": "iter/sec",
            "range": "stddev: 0.12772508891503104",
            "extra": "mean: 23.591237097799944 sec\nrounds: 5"
          }
        ]
      }
    ]
  }
}