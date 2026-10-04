# cube-single task 1 online bridge pilot

Seeds [0, 1, 2]; shared SGCRL warm start to 100,000 environment steps, then three branches to 1,000,000.
Points at or below the branch step are the shared warm-up run, so the three rows are identical there by construction.

## Success rate (%, mean ± std over seeds)

| variant | 100k | 200k | 300k | 500k | 800k | 1000k |
| --- | --- | --- | --- | --- | --- | --- |
| SGCRL | 0.3 ± 0.5 (1/100, 0/100, 0/100) | 0.7 ± 0.9 (0/100, 0/100, 2/100) | 1.7 ± 1.7 (4/100, 0/100, 1/100) | 3.7 ± 2.6 (6/100, 0/100, 5/100) | 7.7 ± 3.9 (13/100, 6/100, 4/100) | 9.3 ± 3.4 (14/100, 6/100, 8/100) |
| SGCRL + deterministic bridge | 0.3 ± 0.5 (1/100, 0/100, 0/100) | 1.3 ± 1.9 (4/100, 0/100, 0/100) | 1.7 ± 1.2 (2/100, 0/100, 3/100) | 10.7 ± 5.8 (17/100, 3/100, 12/100) | 13.7 ± 7.6 (24/100, 6/100, 11/100) | 18.0 ± 7.1 (22/100, 8/100, 24/100) |
| SGCRL + RF bridge | 0.3 ± 0.5 (1/100, 0/100, 0/100) | 1.3 ± 1.2 (3/100, 1/100, 0/100) | 2.7 ± 1.7 (2/100, 1/100, 5/100) | 9.0 ± 4.9 (15/100, 3/100, 9/100) | 13.7 ± 0.9 (15/100, 13/100, 13/100) | 22.0 ± 2.4 (22/100, 25/100, 19/100) |

## Bridge quality at 1M environment steps

| variant | bridge error (MSE) | final-goal critic score | waypoint critic score |
| --- | --- | --- | --- |
| SGCRL | - | 0.421 | - |
| SGCRL + deterministic bridge | 0.1371 ± 0.0087 | -0.001421 ± 0.12 | -1.08 ± 0.68 |
| SGCRL + RF bridge | 0.3397 ± 0.012 | -0.3996 ± 0.056 | -0.9241 ± 0.12 |
