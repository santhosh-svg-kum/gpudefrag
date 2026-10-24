# Weekly plan

## Week 1

- [x] 1. CI hardening: in .github/workflows/ci.yml pin runs-on to ubuntu-24.04 (ubuntu-latest moves to Ubuntu 26 on 2026-10-19), add timeout-minutes (test: 15, calibration: 20), and a concurrency group that cancels superseded runs on the same ref.
- [x] 2. Docs wording (README: 'constraint solver'; FGD warm start guarantee reworded as per-decision). Done except the RESULTS.md evidence cell, split out as 2b.
- [x] 2b. Suite 1a report gains a "Paired by seed vs FGD at 130% load" section (mean diff +/- 95% CI and seeds better), with tests.
- [ ] 2c. In docs/RESULTS.md summary table change the suite 1a evidence cell to the paired-by-seed figure, only with numbers from a `make suite1a` report that reproduces the published table (a 4-core re-run gave b=16 +0.16 pp with 1200 status fallbacks vs the published +0.28 pp: the 500 ms solver budget is hardware-sensitive, so re-run on the original hardware or document the discrepancy first).
- [x] 3. Solve-time breakdown, Go side: Stats records pattern generation, RPC, and solver-reported wall time per solved batch; suite 1a report gains a "Solve-time breakdown" section (p50/p99). Tests.
- [x] 3b. Split solver build vs CP-SAT search time: add an optional `search_wall_s` field to PlaceResponse (regenerate Go and Python code), fill it in solver/src/gpudefrag_solver/place.py from the CP-SAT wall time, record it in mip.Stats, add a `build` (wall_s - search) and `search` column to the breakdown section. Tests on both sides.
- [x] 4. Add BenchmarkGenPatterns (mip/patterns.go genPatterns) and parallelize per-node pattern generation with a bounded goroutine pool. A test must prove the output is identical (same patterns, same order) to the serial version. Put before/after ns/op in the PR body.
- [x] 5. Make defrag restart time configurable: Restart is hard-coded to 60 in cmd/gpudefrag-sim/suite1b.go and suite2.go; add a -restart flag (default 60, unchanged behavior) and a -restart-sweep option (60,300,900) wiring into timed.DefragConfig. Tests.
- [x] 6. Model the eviction grace period in timed defrag (pods evicted terminate over a grace period, default 0 = current behavior; flag -grace, e.g. 30s, delays the blocked pod's bind). Tests.
- [ ] 7. Gang warm-start: option for the MIP placer to warm-start from the better (by the batch objective) of Binpack and FGD placements, behind a flag defaulting off. Tests. Then write next week's 7 tasks into docs/WEEKLY_PLAN.md (new week heading).
