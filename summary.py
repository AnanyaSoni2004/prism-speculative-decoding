"""Print a five-line summary from the results files."""
import pandas as pd

from prism.util import RESULTS as R

e0 = pd.read_csv(R / "e0_checks.csv")
A = pd.read_csv(R / "e2_modelA_main.csv").set_index("kappa")
C = pd.read_csv(R / "e2_modelC.csv")
D = pd.read_csv(R / "e2_modelD.csv").set_index("sigma")
print(f"1. E0: {int(e0['pass'].sum())}/{len(e0)} theory checks passed.")
print(f"2. E2 Model A (m=0.7, c=0.05): adaptive oracle beats best fixed gamma by {A.loc[10, 'gain_pct']:.1f}% at kappa=10 "
      f"and {A.loc[5, 'gain_pct']:.1f}% at kappa=5; landmark underestimates by {A.loc[5, 'landmark_error_pct']:.1f}% at kappa=5.")
print(f"3. E2 Model C/D: state-aware adaptive gains {C.state_gain_pct.min():.1f}-{C.state_gain_pct.max():.1f}%; "
      f"noisy System 1 keeps {D.loc[0.05, 'gain_survival_pct']:.0f}% of the gain at sigma=0.05, {D.loc[0.2, 'gain_survival_pct']:.0f}% at 0.2.")
if (R / "e3_replay.csv").exists():
    t = pd.read_csv(R / "timing.csv").dropna(subset=["c"])
    r = pd.read_csv(R / "e3_replay.csv")
    m = r[r.c_label == "measured"].pivot_table(index=["pair", "type"], columns="policy", values="thr_mean")
    g = 100 * (m.adaptive_oracle / m.best_fixed - 1)
    rec = r[(r.c_label == "measured") & (r.policy == "adaptive_realistic")].recovery_pct
    print("4. E3 measured c: " + ", ".join(f"{a.pair} {a.c:.3f}" for _, a in t.iterrows()) +
          f"; best-fixed speedup {m.best_fixed.min():.2f}-{m.best_fixed.max():.2f}x on the test half.")
    print(f"5. E3 oracle gain over best fixed: {g.min():.1f}-{g.max():.1f}%; realistic System 1 recovers "
          f"median {rec.median():.0f}% of it (range {rec.min():.0f}% to {rec.max():.0f}%).")
else:
    print("4. E3 skipped (--skip-models).")
    print("5. See results/REPORT.md for details.")
