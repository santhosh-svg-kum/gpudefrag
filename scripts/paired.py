#!/usr/bin/env python3
"""Paired comparison from a suite's runs.json: per window/seed, variant minus
baseline, then mean and 95% CI (t) of the differences.

    scripts/paired.py results/suite2-x1.25/runs.json Binpack alloc_pct P95 jct_mean_h
"""
import json, math, sys
from collections import defaultdict

T = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45, 8: 2.36, 9: 2.31, 10: 2.26}

path, base, *metrics = sys.argv[1:]
runs = json.load(open(path))
key = "window" if "window" in runs[0] else "seed"
by = defaultdict(dict)
for r in runs:
    by[r["variant"]][r[key]] = r
print(f"paired vs {base} ({key}s matched): mean diff ± 95% CI")
for v in by:
    if v == base:
        continue
    cells = []
    for m in metrics:
        d = [by[v][k][m] - by[base][k][m] for k in by[v] if k in by[base]]
        n = len(d)
        mu = sum(d) / n
        sd = math.sqrt(sum((x - mu) ** 2 for x in d) / (n - 1)) if n > 1 else 0
        ci = T.get(n, 1.96) * sd / math.sqrt(n) if n > 1 else 0
        cells.append(f"{m} {mu:+.2f} ± {ci:.2f}")
    print(f"  {v:16s} " + " | ".join(cells))
