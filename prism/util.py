"""Small shared helpers: paths, seeds, logging."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

# Every random step in the project derives from these seeds.
SEEDS = {
    "e0": 20261110,
    "e2_fit": 7001,
    "e2_eval": [101, 202, 303, 404, 505],
    "e3_split": 42,
    "e3_data": 1234,
    "e3_replay": [11, 22, 33, 44, 55],
}


def log_seeds(stage: str, seeds) -> None:
    """Append the seeds a stage used to results/seeds.json."""
    path = RESULTS / "seeds.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data[stage] = seeds
    path.write_text(json.dumps(data, indent=2))
    print(f"[{stage}] seeds: {seeds}")


def record_runtime(stage: str, seconds: float, extra: dict | None = None) -> None:
    path = RESULTS / "runtime.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data[stage] = {"seconds": round(seconds, 2), **(extra or {})}
    path.write_text(json.dumps(data, indent=2))


class Timer:
    def __init__(self, stage: str):
        self.stage = stage

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        dt = time.time() - self.t0
        record_runtime(self.stage, dt)
        print(f"[{self.stage}] finished in {dt:.1f} s")


def write_md_table(df, path, floatfmt=".4g") -> None:
    """Write a DataFrame as a GitHub markdown table without needing `tabulate`."""
    def fmt(v):
        if isinstance(v, float):
            return format(v, floatfmt)
        return str(v)

    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in cols) + " |")
    Path(path).write_text("\n".join(lines) + "\n")
