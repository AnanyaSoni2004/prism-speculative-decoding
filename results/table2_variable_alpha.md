| section | setting | landmark_pred | fixed_true_at_landmark_gamma | best_fixed | adaptive | gain_pct | landmark_error_pct |
|---|---|---|---|---|---|---|---|
| A persistent | kappa=inf, SD=0.000 | 2.353 (6) | 2.353 | 2.353 (6) | 2.353 | 0.0 | 0.0 |
| A persistent | kappa=50, SD=0.064 | 2.353 (6) | 2.394 | 2.394 (6) | 2.437 | 1.8 | 1.7 |
| A persistent | kappa=20, SD=0.100 | 2.353 (6) | 2.452 | 2.452 (6) | 2.579 | 5.2 | 4.2 |
| A persistent | kappa=10, SD=0.138 | 2.353 (6) | 2.538 | 2.556 (7) | 2.877 | 12.6 | 7.9 |
| A persistent | kappa=5, SD=0.187 | 2.353 (6) | 2.684 | 2.783 (10) | 3.619 | 30.0 | 14.1 |
| B per-token iid | kappa=50, SD=0.064 | 2.353 (6) | 2.353 ± 0.003 | 2.353 ± 0.003 (6) | history 2.353; look-ahead 2.363 | history -0.00; look-ahead 0.40 | 0.02 |
| B per-token iid | kappa=20, SD=0.100 | 2.353 (6) | 2.353 ± 0.001 | 2.353 ± 0.001 (6) | history 2.352; look-ahead 2.372 | history -0.02; look-ahead 0.81 | -0.01 |
| B per-token iid | kappa=10, SD=0.138 | 2.353 (6) | 2.353 ± 0.003 | 2.353 ± 0.003 (6) | history 2.351; look-ahead 2.388 | history -0.09; look-ahead 1.49 | -0.01 |
| B per-token iid | kappa=5, SD=0.187 | 2.353 (6) | 2.352 ± 0.002 | 2.351 ± 0.002 (5) | history 2.346; look-ahead 2.418 | history -0.23; look-ahead 2.82 | -0.06 |
| C Markov regimes | switch s=0.01 | 2.229 (5) | 2.043 ± 0.037 | 2.043 ± 0.037 (5) | state 2.325 ± 0.062; per-token 2.303 ± 0.058 | state 13.8; per-token 12.7 | -8.3 |
| C Markov regimes | switch s=0.05 | 2.229 (5) | 2.062 ± 0.012 | 2.062 ± 0.012 (5) | state 2.305 ± 0.023; per-token 2.328 ± 0.024 | state 11.8; per-token 12.9 | -7.5 |
| C Markov regimes | switch s=0.2 | 2.229 (5) | 2.121 ± 0.012 | 2.121 ± 0.012 (5) | state 2.233 ± 0.021; per-token 2.327 ± 0.021 | state 5.3; per-token 9.7 | -4.8 |
| D noisy System 1 | kappa=10, sigma=0 |  |  | 2.556 | 2.878 ± 0.002 | 12.6 (survival 100% of oracle gain) |  |
| D noisy System 1 | kappa=10, sigma=0.05 |  |  | 2.556 | 2.786 ± 0.002 | 9.0 (survival 72% of oracle gain) |  |
| D noisy System 1 | kappa=10, sigma=0.1 |  |  | 2.556 | 2.615 ± 0.003 | 2.3 (survival 19% of oracle gain) |  |
| D noisy System 1 | kappa=10, sigma=0.2 |  |  | 2.556 | 2.300 ± 0.003 | -10.0 (survival -79% of oracle gain) |  |
