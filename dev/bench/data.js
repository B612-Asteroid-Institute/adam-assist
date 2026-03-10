window.BENCHMARK_DATA = {
  "lastUpdate": 1773168647791,
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
          "id": "82519c45f42b7f000b84b33b3a7d13705b75f37d",
          "message": "Updates for physical parameters (#29)\n\n* Updates for physical parameters\n\n* match python requirements",
          "timestamp": "2026-02-09T11:55:16-05:00",
          "tree_id": "8a52afdede19de4152c130308f564743999ecd13",
          "url": "https://github.com/B612-Asteroid-Institute/adam-assist/commit/82519c45f42b7f000b84b33b3a7d13705b75f37d"
        },
        "date": 1770738931722,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_benchmark_propagation_vs_raw",
            "value": 0.26459478734114167,
            "unit": "iter/sec",
            "range": "stddev: 0.007905790338419597",
            "extra": "mean: 3.779363947600001 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_benchmark_ephemeris_generation",
            "value": 11.026815492890405,
            "unit": "iter/sec",
            "range": "stddev: 0.0005918404777347898",
            "extra": "mean: 90.68801419999772 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_benchmark_impact_detection",
            "value": 0.4580732562566025,
            "unit": "iter/sec",
            "range": "stddev: 0.005886419911694673",
            "extra": "mean: 2.1830569376000026 sec\nrounds: 5"
          },
          {
            "name": "tests/test_impacts.py::test_calculate_impacts_benchmark_some_impacts[1]",
            "value": 0.4572171756784997,
            "unit": "iter/sec",
            "range": "stddev: 0.011856849991387579",
            "extra": "mean: 2.187144432000008 sec\nrounds: 5"
          },
          {
            "name": "tests/test_impacts.py::test_calculate_impacts_benchmark_some_impacts[2]",
            "value": 0.6550525038482036,
            "unit": "iter/sec",
            "range": "stddev: 0.01783919613774613",
            "extra": "mean: 1.526595187600003 sec\nrounds: 5"
          },
          {
            "name": "tests/test_impacts.py::test_calculate_impacts_benchmark_impacts[1]",
            "value": 2.012351951508372,
            "unit": "iter/sec",
            "range": "stddev: 0.0030746122579690897",
            "extra": "mean: 496.9309664000093 msec\nrounds: 5"
          },
          {
            "name": "tests/test_impacts.py::test_calculate_impacts_benchmark_impacts[2]",
            "value": 2.7229843348901346,
            "unit": "iter/sec",
            "range": "stddev: 0.004604785079638328",
            "extra": "mean: 367.2441251999885 msec\nrounds: 5"
          },
          {
            "name": "tests/test_impacts.py::test_calculate_impacts_benchmark_no_impacts[1]",
            "value": 0.0208977512753504,
            "unit": "iter/sec",
            "range": "stddev: 0.10638253401747454",
            "extra": "mean: 47.85203856740001 sec\nrounds: 5"
          },
          {
            "name": "tests/test_impacts.py::test_calculate_impacts_benchmark_no_impacts[2]",
            "value": 0.04193404410609062,
            "unit": "iter/sec",
            "range": "stddev: 0.12273139868794902",
            "extra": "mean: 23.846972580799978 sec\nrounds: 5"
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
          "id": "2d92fb4408d074e8f954d1be4873ede9e2948a3c",
          "message": "Add ASSIST perturber matching utilities (#30)",
          "timestamp": "2026-03-10T14:26:34-04:00",
          "tree_id": "4089b6e596be385b1aa90717b74b239b87686bc7",
          "url": "https://github.com/B612-Asteroid-Institute/adam-assist/commit/2d92fb4408d074e8f954d1be4873ede9e2948a3c"
        },
        "date": 1773168647526,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_benchmark_propagation_vs_raw",
            "value": 0.2736623156041147,
            "unit": "iter/sec",
            "range": "stddev: 0.013239537411544355",
            "extra": "mean: 3.654138487399996 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_benchmark_ephemeris_generation",
            "value": 12.930319748185006,
            "unit": "iter/sec",
            "range": "stddev: 0.00150480892395251",
            "extra": "mean: 77.33760800001619 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_benchmark_impact_detection",
            "value": 0.5192673938530772,
            "unit": "iter/sec",
            "range": "stddev: 0.002842279362662687",
            "extra": "mean: 1.92579008780001 sec\nrounds: 5"
          },
          {
            "name": "tests/test_impacts.py::test_calculate_impacts_benchmark_some_impacts[1]",
            "value": 0.5179347788369499,
            "unit": "iter/sec",
            "range": "stddev: 0.003957762656902414",
            "extra": "mean: 1.9307450297999935 sec\nrounds: 5"
          },
          {
            "name": "tests/test_impacts.py::test_calculate_impacts_benchmark_some_impacts[2]",
            "value": 0.7455829497528665,
            "unit": "iter/sec",
            "range": "stddev: 0.02035945643498696",
            "extra": "mean: 1.3412323877999939 sec\nrounds: 5"
          },
          {
            "name": "tests/test_impacts.py::test_calculate_impacts_benchmark_impacts[1]",
            "value": 2.1974820637854897,
            "unit": "iter/sec",
            "range": "stddev: 0.003333189570091175",
            "extra": "mean: 455.0662854000052 msec\nrounds: 5"
          },
          {
            "name": "tests/test_impacts.py::test_calculate_impacts_benchmark_impacts[2]",
            "value": 2.9941973670670925,
            "unit": "iter/sec",
            "range": "stddev: 0.004578086236686925",
            "extra": "mean: 333.97931979999385 msec\nrounds: 5"
          },
          {
            "name": "tests/test_impacts.py::test_calculate_impacts_benchmark_no_impacts[1]",
            "value": 0.022791391541279744,
            "unit": "iter/sec",
            "range": "stddev: 0.04799276114740024",
            "extra": "mean: 43.876215201200026 sec\nrounds: 5"
          },
          {
            "name": "tests/test_impacts.py::test_calculate_impacts_benchmark_no_impacts[2]",
            "value": 0.0467388709667549,
            "unit": "iter/sec",
            "range": "stddev: 0.11497080193893268",
            "extra": "mean: 21.395467612200015 sec\nrounds: 5"
          }
        ]
      }
    ]
  }
}