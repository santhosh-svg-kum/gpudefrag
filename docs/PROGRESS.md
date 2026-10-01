# gpupack progress

Ledger for the autonomous build loop. Newest notes at the bottom of each section.

## Milestones
- [x] M1 foundation + FGD-faithful baselines + calibration gate — plan: docs/superpowers/plans/2026-09-30-m1-foundation-calibration.md
- [ ] M2 CP-SAT solver service + MIP placement + suite 1a report
- [ ] M3 timed DES + busy-server latency + defrag + suite 1b
- [ ] M4 gangs/topology + Helios/Philly loader + Volcano gang baseline + suite 2 + README

## M1 tasks
- [x] 1 model  - [x] 2 trace  - [x] 3 frag  - [x] 4 workload
- [x] 5 sim cluster  - [x] 6 sched  - [x] 7 calib  - [x] 8 CLI + gate run

## Notes

### 2026-09-30 — M1 done: calibration gate PASSES for all six baselines
`make calib` (60 runs, ~10 s on 10 cores). Seed-mean allocation % at 50/80/100/110/120/130% arrived:

```
calibration vs FGD reference (openb default, tune 1.3, seeds 42-51, 9s)
policy         ours @ 50,80,100,110,120,130                 ref                                            maxΔ exact 
Random         49.95 78.69 86.48 87.06 87.44 87.62          49.97 78.66 86.30 86.88 87.28 87.47            0.18  0/10 PASS
DotProd        50.01 80.02 90.56 90.64 90.71 90.79          50.01 80.02 90.62 90.70 90.76 90.85            0.06  1/10 PASS
GpuClustering  50.01 80.02 91.06 91.14 91.21 91.29          50.01 80.02 91.65 91.73 91.80 91.88            0.59  0/10 PASS
GpuPacking     50.01 80.02 91.11 91.19 91.26 91.34          50.01 80.02 91.78 91.85 91.92 92.00            0.67  0/10 PASS
BestFit        50.01 80.02 92.98 93.05 93.12 93.20          50.01 80.02 92.86 92.94 93.00 93.08            0.12  0/10 PASS
FGD            50.01 80.02 95.12 95.19 95.25 95.30          50.01 80.02 95.23 95.30 95.35 95.39            0.10  1/10 PASS
calibration gate PASSED
```
- FGD reproduced within 0.10 pp (ref 95.39 @130%). FGD frag unit tests match FGD's own expected values exactly.
- Known residual: GpuPacking (-0.67 pp) and GpuClustering (-0.59 pp) run slightly low, inside tolerance. Not yet explained; candidates: kube default score plugins left enabled in FGD's profile, or GPU device choice inside multi-GPU pods. Revisit if M2 comparisons hinge on them (they don't: headline compares vs FGD).
- Deviation: baselines live in `sched/` (not `sched/baselines/`); M1 has no event heap (arrival-only), DES arrives in M3.
- Headroom note: FGD already allocates 95.4% at 130% load, so the max possible gain on suite 1a is ~4.6 pp. M2 should also report fragmentation and failed-pod counts, not only allocation.
