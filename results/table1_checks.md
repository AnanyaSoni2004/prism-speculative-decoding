| check | formula_value | simulated_value | abs_error | pass |
|---|---|---|---|---|
| 1a exact output law: max|out - p| over 1000 pairs | 0 | 5.551e-17 | 5.551e-17 | True |
| 1b sampled draws (200k): L1(empirical, p); formula = expected L1 of exact sampling | 0.01071 | 0.01048 | 0.0002294 | True |
| 2 beta = 1 - 0.5||p-q||_1: max error over 1000 pairs | 0 | 5.551e-16 | 5.551e-16 | True |
| 3 E[N] alpha=0.5 gamma=1 (200k rounds) | 1.5 | 1.502 | 0.001835 | True |
| 3 E[N] alpha=0.5 gamma=3 (200k rounds) | 1.875 | 1.874 | 0.001175 | True |
| 3 E[N] alpha=0.5 gamma=5 (200k rounds) | 1.969 | 1.975 | 0.005985 | True |
| 3 E[N] alpha=0.5 gamma=10 (200k rounds) | 1.999 | 1.998 | 0.0006334 | True |
| 3 E[N] alpha=0.7 gamma=1 (200k rounds) | 1.7 | 1.7 | 0.00041 | True |
| 3 E[N] alpha=0.7 gamma=3 (200k rounds) | 2.533 | 2.534 | 0.001125 | True |
| 3 E[N] alpha=0.7 gamma=5 (200k rounds) | 2.941 | 2.943 | 0.001735 | True |
| 3 E[N] alpha=0.7 gamma=10 (200k rounds) | 3.267 | 3.264 | 0.003447 | True |
| 3 E[N] alpha=0.8 gamma=1 (200k rounds) | 1.8 | 1.799 | 0.000735 | True |
| 3 E[N] alpha=0.8 gamma=3 (200k rounds) | 2.952 | 2.949 | 0.00307 | True |
| 3 E[N] alpha=0.8 gamma=5 (200k rounds) | 3.689 | 3.686 | 0.003495 | True |
| 3 E[N] alpha=0.8 gamma=10 (200k rounds) | 4.571 | 4.569 | 0.001413 | True |
| 3 E[N] alpha=0.9 gamma=1 (200k rounds) | 1.9 | 1.9 | 0.000235 | True |
| 3 E[N] alpha=0.9 gamma=3 (200k rounds) | 3.439 | 3.435 | 0.003745 | True |
| 3 E[N] alpha=0.9 gamma=5 (200k rounds) | 4.686 | 4.689 | 0.003315 | True |
| 3 E[N] alpha=0.9 gamma=10 (200k rounds) | 6.862 | 6.851 | 0.01131 | True |
| 4 gamma* alpha=0.6 c=0.05: Lambert W vs real grid (step 1e-4) | 4.065 | 4.065 | 3.552e-06 | True |
| 4 best integer gamma alpha=0.6 c=0.05: floor/ceil of gamma* vs integer search | 4 | 4 | 0 | True |
| 4 best integer speedup alpha=0.6 c=0.05 | 1.921 | 1.921 | 0 | True |
| 4 gamma* alpha=0.8 c=0.05: Lambert W vs real grid (step 1e-4) | 7.857 | 7.857 | 4.554e-05 | True |
| 4 best integer gamma alpha=0.8 c=0.05: floor/ceil of gamma* vs integer search | 8 | 8 | 0 | True |
| 4 best integer speedup alpha=0.8 c=0.05 | 3.092 | 3.092 | 0 | True |
| 4 gamma* alpha=0.8 c=0.2: Lambert W vs real grid (step 1e-4) | 3.904 | 3.904 | 2.909e-05 | True |
| 4 best integer gamma alpha=0.8 c=0.2: floor/ceil of gamma* vs integer search | 4 | 4 | 0 | True |
| 4 best integer speedup alpha=0.8 c=0.2 | 1.868 | 1.868 | 0 | True |
| 4 gamma* alpha=0.9 c=0.02: Lambert W vs real grid (step 1e-4) | 19.06 | 19.06 | 1.592e-06 | True |
| 4 best integer gamma alpha=0.9 c=0.02: floor/ceil of gamma* vs integer search | 19 | 19 | 0 | True |
| 4 best integer speedup alpha=0.9 c=0.02 | 6.365 | 6.365 | 0 | True |
| 4 known value: gamma*(0.8, 0.05) = 7.8567 | 7.857 | 7.857 | 4.554e-05 | True |
| 4 known value: S at best integer (0.8, 0.05) = 3.092 | 3.092 | 3.092 | 7.954e-05 | True |
| 5 max_{gamma>=1} S > 1 iff alpha > c: disagreeing points on 99x99 grid (S>1+1e-9) | 0 | 0 | 0 | True |
| 6 known value: exact E[N] m=0.7 kappa=5 gamma=6 = 3.4894 | 3.489 | 3.489 | 4.453e-05 | True |
| 6i persistent alpha: simulation (2M rounds) vs exact sum E[alpha^k] | 3.489 | 3.485 | 0.004008 | True |
| 6ii exact E[N] >= landmark at alpha=m (formula=landmark 3.0588) | 3.059 | 3.489 | 0.4305 | True |
| 6iii per-token independent alpha: simulation (2M) vs landmark 3.0588 | 3.059 | 3.058 | 0.0008665 | True |
