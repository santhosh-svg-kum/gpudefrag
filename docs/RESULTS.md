# gpupack v1 results

Every number here is copied from a generated report (`results/*/report.md`,
produced by the `make` targets in the README). Reproduce on your own machine
before citing anything. All results are simulation results.


## Summary (read this first)

| Claim | Evidence | Strength |
|---|---|---|
| Our baselines are faithful to FGD | all 6 within ±1.0 pp of FGD's published curves; FGD within 0.10 pp | strong |
| gpupack batch placement beats FGD on FGD's own benchmark | suite 1a: +0.28 pp at 130% load, 10 seeds, non-overlapping CIs | strong |
| gpupack beats FGD under timed overload | suite 1b load 1.1, held-out seeds: +0.31 ± 0.17 pp allocation | moderate (5 seeds) |
| gpupack is not better with slack capacity | suite 1b load 0.9 held-out: −0.56 ± 0.53 pp | moderate; diagnosed (§2) |
| Defrag recovers capacity under overload | suite 2 ×1.5: Binpack+defrag +2.64 ± 2.12 pp allocation (paired by week) | moderate (4 weeks) |
| Defrag helps elsewhere | +0.3 to +1.8 pp in suites 1b and 2 at a few tens of seconds of p95 | directional only |
| gpupack placement beats Volcano-style binpack on gangs | not supported; slightly worse, not significant | negative result, diagnosed (§3) |

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

## 2. Suite 1b: timed, SYNTHETIC workload from openb (with defrag)

151 of 1213 openb GPU nodes (stratified by GPU type and count); Poisson arrivals at the given offered GPU load; (pod, duration) pairs resampled from openb with durations capped at 2h0m0s; 2h0m0s warm-up excluded, metrics over the next 6h0m0s. Seeds 47–51, mean ± 95% CI. Latency = pending time (s) of GPU pods that arrived in the window and were bound. Alloc = time-averaged allocated GPU. "Pending at end" = backlog when the window closes.

Two seed sets. Seeds 42–46 ran first and were used to diagnose large-job starvation and to choose gpupack's defaults (idle-node penalty, GPU-weighted fragmentation). Seeds 47–51 are **held out** and ran once with those defaults. Draw conclusions from the held-out set. Note that the original-seed gpupack rows predate the defaults change.

### Offered load 0.9

**Held-out seeds 47–51:**

| variant | p50 s | p95 s | p99 s | alloc % | pending at end | migrations | lost GPU-h | solve p99 ms |
|---|---|---|---|---|---|---|---|---|
| FGD | 0.0 ± 0.0 | 0 ± 0 | 4 ± 7 | 88.36 ± 1.09 | 12.2 ± 13.1 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0 ± 0 |
| BestFit | 0.0 ± 0.0 | 0 ± 0 | 17 ± 28 | 88.23 ± 1.16 | 13.4 ± 13.4 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0 ± 0 |
| gpupack | 0.0 ± 0.0 | 0 ± 0 | 0 ± 0 | 87.80 ± 1.30 | 18.4 ± 11.3 | 0.0 ± 0.0 | 0.0 ± 0.0 | 11 ± 0 |
| FGD+defrag | 0.0 ± 0.0 | 1 ± 3 | 20 ± 27 | 89.39 ± 2.57 | 5.2 ± 6.5 | 44.6 ± 21.1 | 11.5 ± 3.9 | 0 ± 0 |
| gpupack+defrag | 0.0 ± 0.0 | 0 ± 1 | 9 ± 20 | 89.09 ± 2.42 | 10.4 ± 13.3 | 31.0 ± 23.4 | 6.8 ± 5.5 | 106 ± 32 |

Paired vs FGD (held-out):

```
paired vs FGD (seeds matched): mean diff ± 95% CI
  BestFit          alloc_pct -0.13 ± 0.42 | P95 +0.00 ± 0.00 | unplaced +1.20 ± 9.44
  gpupack          alloc_pct -0.56 ± 0.53 | P95 +0.01 ± 0.00 | unplaced +6.20 ± 6.66
  FGD+defrag       alloc_pct +1.03 ± 1.61 | P95 +0.93 ± 2.58 | unplaced -7.00 ± 15.63
  gpupack+defrag   alloc_pct +0.73 ± 1.74 | P95 +0.24 ± 0.50 | unplaced -1.80 ± 19.55
```

Paired vs FGD (original seeds 42–46, pre-defaults gpupack):

```
paired vs FGD (seeds matched): mean diff ± 95% CI
  BestFit          alloc_pct +0.03 ± 0.81 | P95 +13.81 ± 37.04 | unplaced -1.00 ± 8.61
  gpupack          alloc_pct -1.65 ± 0.96 | P95 -4.92 ± 13.70 | unplaced +17.00 ± 8.34
  FGD+defrag       alloc_pct +1.73 ± 1.79 | P95 +7.33 ± 15.65 | unplaced -12.80 ± 11.71
  gpupack+defrag   alloc_pct +1.64 ± 1.72 | P95 +6.22 ± 8.97 | unplaced -11.40 ± 14.54
```

### Offered load 1.0

**Held-out seeds 47–51:**

| variant | p50 s | p95 s | p99 s | alloc % | pending at end | migrations | lost GPU-h | solve p99 ms |
|---|---|---|---|---|---|---|---|---|
| FGD | 40.1 ± 109.7 | 212 ± 153 | 261 ± 165 | 94.82 ± 1.46 | 78.6 ± 48.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0 ± 0 |
| BestFit | 45.5 ± 102.5 | 250 ± 140 | 314 ± 154 | 94.16 ± 1.61 | 75.4 ± 29.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0 ± 0 |
| gpupack | 11.9 ± 33.0 | 197 ± 126 | 345 ± 179 | 94.57 ± 1.76 | 75.4 ± 46.1 | 0.0 ± 0.0 | 0.0 ± 0.0 | 11 ± 2 |
| FGD+defrag | 57.5 ± 106.7 | 265 ± 129 | 322 ± 141 | 95.53 ± 0.62 | 90.6 ± 78.2 | 13.2 ± 8.0 | 2.1 ± 2.3 | 0 ± 0 |
| gpupack+defrag | 22.8 ± 32.9 | 248 ± 122 | 396 ± 162 | 95.68 ± 0.65 | 78.8 ± 56.9 | 18.0 ± 14.6 | 4.3 ± 3.6 | 154 ± 36 |

Paired vs FGD (held-out):

```
paired vs FGD (seeds matched): mean diff ± 95% CI
  BestFit          alloc_pct -0.66 ± 1.08 | P95 +37.56 ± 60.11 | unplaced -3.20 ± 24.92
  gpupack          alloc_pct -0.25 ± 0.57 | P95 -15.28 ± 47.13 | unplaced -3.20 ± 10.00
  FGD+defrag       alloc_pct +0.70 ± 1.28 | P95 +53.10 ± 57.33 | unplaced +12.00 ± 31.88
  gpupack+defrag   alloc_pct +0.86 ± 0.92 | P95 +35.65 ± 67.32 | unplaced +0.20 ± 18.09
```

Paired vs FGD (original seeds 42–46, pre-defaults gpupack):

```
paired vs FGD (seeds matched): mean diff ± 95% CI
  BestFit          alloc_pct -0.39 ± 0.32 | P95 +28.05 ± 56.98 | unplaced +18.40 ± 16.92
  gpupack          alloc_pct -0.46 ± 0.44 | P95 -67.38 ± 82.42 | unplaced -28.60 ± 66.92
  FGD+defrag       alloc_pct +0.51 ± 0.97 | P95 +24.62 ± 100.59 | unplaced +36.60 ± 65.95
  gpupack+defrag   alloc_pct +0.40 ± 0.87 | P95 +4.42 ± 44.50 | unplaced +1.80 ± 67.87
```

### Offered load 1.1

**Held-out seeds 47–51:**

| variant | p50 s | p95 s | p99 s | alloc % | pending at end | migrations | lost GPU-h | solve p99 ms |
|---|---|---|---|---|---|---|---|---|
| FGD | 826.9 ± 354.0 | 1691 ± 780 | 1765 ± 791 | 96.76 ± 0.19 | 864.6 ± 342.7 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0 ± 0 |
| BestFit | 858.7 ± 295.0 | 1593 ± 600 | 1667 ± 589 | 96.64 ± 0.18 | 816.0 ± 256.9 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0 ± 0 |
| gpupack | 826.0 ± 335.1 | 1569 ± 688 | 1708 ± 673 | 97.07 ± 0.10 | 787.6 ± 324.6 | 0.0 ± 0.0 | 0.0 ± 0.0 | 14 ± 5 |
| FGD+defrag | 890.4 ± 264.7 | 1749 ± 543 | 1829 ± 534 | 96.83 ± 0.24 | 893.2 ± 258.0 | 12.6 ± 13.4 | 0.9 ± 0.7 | 0 ± 0 |
| gpupack+defrag | 841.2 ± 322.4 | 1517 ± 611 | 1661 ± 591 | 97.02 ± 0.19 | 778.8 ± 296.7 | 9.6 ± 12.8 | 0.5 ± 0.8 | 163 ± 75 |

Paired vs FGD (held-out):

```
paired vs FGD (seeds matched): mean diff ± 95% CI
  BestFit          alloc_pct -0.13 ± 0.12 | P95 -97.98 ± 328.70 | unplaced -48.60 ± 128.38
  gpupack          alloc_pct +0.31 ± 0.17 | P95 -121.86 ± 242.18 | unplaced -77.00 ± 87.13
  FGD+defrag       alloc_pct +0.07 ± 0.13 | P95 +57.83 ± 251.18 | unplaced +28.60 ± 100.86
  gpupack+defrag   alloc_pct +0.26 ± 0.17 | P95 -173.93 ± 288.15 | unplaced -85.80 ± 104.13
```

Paired vs FGD (original seeds 42–46, pre-defaults gpupack):

```
paired vs FGD (seeds matched): mean diff ± 95% CI
  BestFit          alloc_pct -0.04 ± 0.09 | P95 -101.43 ± 237.82 | unplaced -45.20 ± 101.82
  gpupack          alloc_pct +0.25 ± 0.03 | P95 -301.00 ± 283.65 | unplaced -155.60 ± 174.88
  FGD+defrag       alloc_pct -0.02 ± 0.11 | P95 -16.02 ± 278.54 | unplaced +5.40 ± 144.11
  gpupack+defrag   alloc_pct +0.28 ± 0.14 | P95 -253.40 ± 171.18 | unplaced -125.20 ± 101.42
```

Reading the held-out results (seeds 47–51):
- **Overload (1.1): gpupack beats FGD on allocation by +0.31 ± 0.17 pp**
  (significant), and gpupack+defrag by +0.26 ± 0.17 pp. p95 wait also trends
  lower (−122 s and −174 s), but not significantly.
- **At 1.0, nothing differs significantly.** gpupack+defrag trends highest
  (+0.86 ± 0.92 pp allocation) at about +36 s p95.
- **At 0.9, gpupack alone is worse than FGD (−0.56 ± 0.53 pp).** Even with
  the large-job fixes, its batch objective under-protects whole nodes when
  the cluster has slack. Defrag trends positive (+0.7 to +1.0 pp), but not
  significantly with 5 seeds.
- **Pattern:** batch optimization pays off when the cluster is saturated and
  there is a real choice of which pods to fit. With slack, FGD's simple
  packing tie-break is hard to beat.

## 3. Suite 2: gang training backtest (Helios Venus)

Helios Venus backtest: 135 nodes x 8 GPUs, synthetic topology domains of 4 nodes (gangs of <= 4 pods must stay in one domain). Real submit times and durations; windows = the 4 busiest weeks by submitted GPU-hours (starting 2020-07-31, 2020-08-28, 2020-09-09, 2020-09-21), each after 3 days of warm-up replay. Dropped rows: 121405 CPU-only, 0 uneven gangs, 0 zero-duration. Mean ± 95% CI across windows. Wait = submit → bind (s) for GPU jobs arriving in the window.

### Time compression ×1.0

| variant | wait p50 s | wait p95 s | wait p99 s | gang wait p95 s | JCT mean h | alloc % | pending at end | migrations | lost GPU-h | solve p99 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| Binpack | 0 ± 0 | 16 ± 49 | 368 ± 604 | 999 ± 3004 | 1.93 ± 0.58 | 82.59 ± 3.36 | 1.5 ± 2.8 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0 ± 0 |
| FGD | 0 ± 0 | 68 ± 215 | 382 ± 606 | 864 ± 2161 | 1.93 ± 0.58 | 82.61 ± 3.36 | 0.8 ± 2.4 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0 ± 0 |
| gpupack | 0 ± 0 | 14 ± 43 | 340 ± 513 | 1715 ± 5319 | 1.93 ± 0.58 | 82.53 ± 3.45 | 1.0 ± 2.2 | 0.0 ± 0.0 | 0.0 ± 0.0 | 283 ± 395 |
| Binpack+defrag | 0 ± 0 | 13 ± 41 | 288 ± 431 | 1013 ± 2991 | 1.93 ± 0.58 | 82.61 ± 3.35 | 0.8 ± 2.4 | 11.8 ± 23.7 | 2.9 ± 6.1 | 0 ± 0 |
| gpupack+defrag | 0 ± 0 | 13 ± 41 | 286 ± 404 | 1729 ± 5306 | 1.93 ± 0.58 | 82.55 ± 3.45 | 1.0 ± 2.2 | 10.5 ± 25.1 | 2.8 ± 7.2 | 186 ± 331 |

Paired vs Binpack (same week):

```
paired vs Binpack (windows matched): mean diff ± 95% CI
  FGD              alloc_pct +0.02 ± 0.04 | P95 +52.00 ± 165.38 | gang_p95 -134.97 ± 1026.74 | jct_mean_h +0.00 ± 0.01
  gpupack          alloc_pct -0.06 ± 0.10 | P95 -1.85 ± 6.53 | gang_p95 +716.95 ± 2317.02 | jct_mean_h -0.00 ± 0.01
  Binpack+defrag   alloc_pct +0.01 ± 0.02 | P95 -2.75 ± 8.74 | gang_p95 +14.05 ± 46.85 | jct_mean_h -0.00 ± 0.01
  gpupack+defrag   alloc_pct -0.05 ± 0.12 | P95 -2.38 ± 8.19 | gang_p95 +730.00 ± 2303.00 | jct_mean_h -0.00 ± 0.01
```

### Time compression ×1.25

| variant | wait p50 s | wait p95 s | wait p99 s | gang wait p95 s | JCT mean h | alloc % | pending at end | migrations | lost GPU-h | solve p99 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| Binpack | 2547 ± 4866 | 47330 ± 45395 | 61739 ± 45337 | 72067 ± 40028 | 4.22 ± 2.28 | 90.48 ± 2.16 | 593.8 ± 986.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 1 ± 0 |
| FGD | 1705 ± 3655 | 55298 ± 53331 | 67908 ± 59194 | 83695 ± 48122 | 4.38 ± 2.31 | 89.67 ± 3.35 | 560.0 ± 1104.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 1 ± 0 |
| gpupack | 3817 ± 9137 | 46583 ± 50682 | 63877 ± 59054 | 85628 ± 45640 | 4.43 ± 2.72 | 89.52 ± 3.26 | 727.0 ± 906.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 406 ± 354 |
| Binpack+defrag | 1819 ± 3466 | 27847 ± 25794 | 39557 ± 29741 | 62657 ± 30737 | 3.51 ± 1.64 | 92.32 ± 1.47 | 542.5 ± 781.8 | 121.5 ± 6.9 | 29.1 ± 8.0 | 0 ± 0 |
| gpupack+defrag | 2252 ± 4402 | 56718 ± 56251 | 72813 ± 69062 | 78796 ± 59930 | 4.77 ± 2.85 | 89.00 ± 3.74 | 781.8 ± 1064.6 | 96.5 ± 31.5 | 23.4 ± 7.1 | 333 ± 371 |

Paired vs Binpack (same week):

```
paired vs Binpack (windows matched): mean diff ± 95% CI
  FGD              alloc_pct -0.81 ± 1.93 | P95 +7968.15 ± 18278.87 | gang_p95 +11628.37 ± 22584.26 | jct_mean_h +0.16 ± 0.42
  gpupack          alloc_pct -0.97 ± 2.21 | P95 -746.64 ± 28317.53 | gang_p95 +13561.10 ± 36355.60 | jct_mean_h +0.22 ± 0.89
  Binpack+defrag   alloc_pct +1.84 ± 2.69 | P95 -19482.42 ± 25574.42 | gang_p95 -9410.29 ± 32346.32 | jct_mean_h -0.71 ± 1.57
  gpupack+defrag   alloc_pct -1.48 ± 3.81 | P95 +9388.28 ± 36582.14 | gang_p95 +6729.23 ± 27635.27 | jct_mean_h +0.55 ± 1.66
```

### Time compression ×1.5

| variant | wait p50 s | wait p95 s | wait p99 s | gang wait p95 s | JCT mean h | alloc % | pending at end | migrations | lost GPU-h | solve p99 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| Binpack | 6699 ± 7162 | 71210 ± 22875 | 88390 ± 35116 | 105491 ± 39443 | 6.55 ± 2.91 | 91.44 ± 2.63 | 1267.0 ± 1193.3 | 0.0 ± 0.0 | 0.0 ± 0.0 | 1 ± 0 |
| FGD | 9739 ± 15316 | 66016 ± 19991 | 87910 ± 30724 | 112804 ± 34658 | 6.52 ± 3.16 | 92.83 ± 1.22 | 868.2 ± 983.7 | 0.0 ± 0.0 | 0.0 ± 0.0 | 1 ± 0 |
| gpupack | 12225 ± 19578 | 79471 ± 20826 | 99449 ± 28454 | 121219 ± 60504 | 7.85 ± 4.36 | 90.09 ± 2.40 | 1251.2 ± 1015.4 | 0.0 ± 0.0 | 0.0 ± 0.0 | 382 ± 300 |
| Binpack+defrag | 14566 ± 11992 | 68036 ± 22386 | 81489 ± 32560 | 102914 ± 38206 | 7.55 ± 2.22 | 94.09 ± 0.65 | 1374.2 ± 888.5 | 96.0 ± 15.0 | 23.9 ± 4.8 | 1 ± 0 |
| gpupack+defrag | 13061 ± 14714 | 71770 ± 39384 | 88025 ± 34688 | 103458 ± 53871 | 7.29 ± 2.56 | 92.44 ± 2.89 | 1113.8 ± 915.5 | 99.0 ± 24.1 | 23.2 ± 7.2 | 347 ± 328 |

Paired vs Binpack (same week):

```
paired vs Binpack (windows matched): mean diff ± 95% CI
  FGD              alloc_pct +1.38 ± 3.06 | P95 -5193.45 ± 12376.20 | gang_p95 +7312.51 ± 17325.10 | jct_mean_h -0.03 ± 2.00
  gpupack          alloc_pct -1.36 ± 2.42 | P95 +8260.80 ± 9311.97 | gang_p95 +15727.51 ± 22113.26 | jct_mean_h +1.30 ± 1.47
  Binpack+defrag   alloc_pct +2.64 ± 2.12 | P95 -3174.42 ± 20199.14 | gang_p95 -2577.30 ± 13220.84 | jct_mean_h +1.00 ± 1.41
  gpupack+defrag   alloc_pct +1.00 ± 4.56 | P95 +560.44 ± 26425.91 | gang_p95 -2033.09 ± 17548.72 | jct_mean_h +0.74 ± 1.67
```

Reading the tables:
- **At normal load (×1.0) no variant differs significantly** (paired
  allocation differences are within ±0.1 pp). The cluster is
  demand-limited.
- **Binpack (Volcano-style) + defrag is the only variant that significantly
  beats Binpack**, and only under overload: **+2.64 ± 2.12 pp allocation at
  ×1.5**. At ×1.25 it is +1.84 ± 2.69 pp and −19.5k s p95 wait, which is
  directional only.
- **gpupack's placement does not beat Binpack on gang workloads.** It is
  slightly worse at every load, though not significantly. Diagnosis: the
  solver works (at most 36 fallbacks in about 7.9k batches per week; it
  improves on its hint in about 3% of batches), but it optimizes FGD's
  fragmentation measure. Whole-node packing, which is what gangs need, is
  better served by aggressive most-allocated binpacking. FGD was designed
  for GPU-sharing workloads.
- **Only 4 weeks.** The intervals are wide; more windows would tighten them.
- **Next steps (v1.1):** warm-start from the better of the Binpack and FGD
  placements, and add a gang-aware term (count of free whole nodes per
  domain) to the objective.

## Future work: decision latency

Benchmark solve times are pessimistic: CP-SAT already runs a parallel
portfolio search, but the benchmarks capped it at `NumCPU / concurrent runs`
(2 workers) because several simulations shared one machine. Planned
reductions, in expected payoff order:

1. **Parallel pattern generation (Go).** Each node's patterns are
   independent: one goroutine per node.
2. **Remove the Python model-building bottleneck.** Model construction runs
   under the GIL. Short term: run a pool of solver processes instead of one
   threaded server. Long term: build the CP-SAT model in Go and call the C++
   solver directly. The proto stays as the pluggable solver contract.
3. **Decomposition.** Pods with disjoint candidate nodes (often separate
   topology domains) form independent subproblems. Solve them concurrently
   and merge.
4. **Pipelining and caching.** Solve batch N+1 while batch N binds, and reuse
   patterns for nodes unchanged since the last batch.

The first step is profiling the split between pattern generation, model
build, RPC, and search, so effort goes to the measured bottleneck.

## Known simplifications in the defrag model

- **No eviction grace period.** The blocked pod binds at the instant the plan
  is made. In Kubernetes, evicted pods first terminate gracefully (30 s by
  default), so real waits for the unblocked job are about that much longer.
- **Fixed 60 s restart.** Migrated pods pay restart time plus the work lost
  since their last checkpoint. Lost work is modeled per pod; restart time is
  a constant. Large training jobs reloading tens of GB of checkpoint may take
  2–10 minutes. v1.1: model the grace period, and sweep restart time (60 s,
  5 min, 15 min) to find where defrag stops paying off. The hysteresis rule
  already refuses plans whose cost exceeds the benefit.
- **Defrag never provisions nodes.** It only rearranges pods across running,
  already-initialized nodes. Node provisioning (minutes) is the alternative
  it avoids. Merging the two decisions is v2 (elastic mode).
