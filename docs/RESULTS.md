# gpupack v1 results

Every number here is copied from a generated report (`results/*/report.md`,
produced by the `make` targets in the README). Reproduce on your own machine
before citing anything. All results are simulation results.

## 0. Calibration gate (baselines are faithful)

`make calib`: our ports of FGD's six baselines on FGD's own protocol
(openb default trace, 130% inflation, seeds 42–51), compared with FGD's
published per-seed curves. Gate: seed-mean allocation within ±1.0 pp at
50, 80, 100, 110, 120, and 130% arrived load.

```
calibration vs FGD reference (openb default, tune 1.3, seeds 42-51, 11s)
policy         ours @ 50,80,100,110,120,130                 ref                                            maxΔ exact 
Random         49.95 78.69 86.48 87.06 87.44 87.62          49.97 78.66 86.30 86.88 87.28 87.47            0.18  0/10 PASS
DotProd        50.01 80.02 90.56 90.64 90.71 90.79          50.01 80.02 90.62 90.70 90.76 90.85            0.06  1/10 PASS
GpuClustering  50.01 80.02 91.06 91.14 91.21 91.29          50.01 80.02 91.65 91.73 91.80 91.88            0.59  0/10 PASS
GpuPacking     50.01 80.02 91.11 91.19 91.26 91.34          50.01 80.02 91.78 91.85 91.92 92.00            0.67  0/10 PASS
BestFit        50.01 80.02 92.98 93.05 93.12 93.20          50.01 80.02 92.86 92.94 93.00 93.08            0.12  0/10 PASS
FGD            50.01 80.02 95.12 95.19 95.25 95.30          50.01 80.02 95.23 95.30 95.35 95.39            0.10  1/10 PASS
calibration gate PASSED
```

FGD reproduces within 0.10 pp. GpuPacking and GpuClustering run about 0.6 pp
low. That is inside the gate, but not yet explained.

## 1. Suite 1a: FGD protocol (openb, arrival-only, 130% load)

Seeds 42–51, mean ± 95% CI. Allocation = allocated GPU / cluster GPU at the given arrived load.

| variant | alloc@100% | alloc@120% | alloc@130% | final alloc % | unplaced GPU demand % of cap | improved batches | solve p50 / p99 ms | fallbacks |
|---|---|---|---|---|---|---|---|---|
| FGD | 95.12 ± 0.10 | 95.25 ± 0.10 | 95.30 ± 0.10 | 95.30 ± 0.10 | 34.70 ± 0.09 | – | – | – |
| BestFit | 92.98 ± 0.16 | 93.12 ± 0.16 | 93.19 ± 0.15 | 93.19 ± 0.15 | 36.81 ± 0.15 | – | – | – |
| gpupack:b=1:t=250ms | 94.87 ± 0.12 | 95.01 ± 0.12 | 95.07 ± 0.11 | 95.07 ± 0.11 | 34.93 ± 0.11 | 23630 / 108245 | 3 / 56 | map[empty:24806] |
| gpupack:b=16:t=500ms | 95.45 ± 0.11 | 95.55 ± 0.11 | 95.58 ± 0.11 | 95.58 ± 0.11 | 34.42 ± 0.10 | 3012 / 6769 | 612 / 1050 | map[empty:142 status:346] |
| gpupack:b=64:t=2s | 95.10 ± 0.10 | 95.22 ± 0.10 | 95.27 ± 0.09 | 95.27 ± 0.09 | 34.73 ± 0.10 | 472 / 1695 | 3585 / 5002 | map[rpc:52 status:615] |

Reading the table:
- **gpupack with batches of 16 (500 ms budget) beats FGD by +0.28 pp** at
  130% load (95.58 ± 0.11 vs 95.30 ± 0.10). The confidence intervals do not
  overlap. The solver found a strictly better placement than FGD's in 44% of
  batches.
- **Batches of 1 lose to FGD (−0.23 pp).** Optimizing FGD's exact metric one
  pod at a time drops an implicit packing rule that FGD gets from its integer
  score truncation plus node-name tie-break. The gain comes from deciding a
  batch jointly, not from per-pod precision.
- **Batches of 64 (2 s budget) show no gain.** The solve often misses the
  budget: there are 615 status fallbacks and 52 timeouts, and solve p99 is
  5 s. Those batches fall back to FGD.
- **Headroom is small.** FGD already allocates 95.3%, so the ceiling on this
  benchmark is about 4.7 pp.
- These runs use the M2 objective (no idle-node term, count-weighted
  classes). The later defaults (§2) were not re-run here.
