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
- [x] 7. Gang warm-start: option for the MIP placer to warm-start from the better (by the batch objective) of Binpack and FGD placements, behind a flag defaulting off. Tests. Then write next week's 7 tasks into docs/WEEKLY_PLAN.md (new week heading).

## Week 2

- [x] 1. Gang-aware objective term: count free whole nodes per topology domain in the batch objective (`mip.evaluate`), behind a `Placer` flag defaulting off. Unit tests on a cluster where two placements tie on fragmentation but differ in free whole nodes.
- [x] 2. Decomposition, part 1: a pure function that splits a batch into groups of pods with disjoint candidate-node sets (union-find over `cands`). Tests: disjoint, chained and gang-linked cases. No behavior change yet.
- [x] 3. Decomposition, part 2: solve the groups from task 2 as separate solver requests concurrently, behind a flag defaulting off; merge answers and validate. Tests with a fake solver proving the merged result equals the single-request result.
- [x] 4. Solver process pool: `StartLocal` can start N solver processes and `Dial` round-robins requests across them (flag `-solver-procs`, default 1). Tests for the round-robin and shutdown.
- [x] 5. Pattern caching: memoize `genPatterns` per (node state, pod-resource multiset) within a batch so repeated pod shapes reuse results. Test identical output to uncached; benchmark ns/op in the PR body.
- [x] 6a. Cross-batch pattern cache: `Placer.CrossBatchCache` (default off, bounded by `CacheMaxEntries`) keeps the pattern cache across batches. Tests for identical output, cross-batch hits and the bound.
- [x] 6b. Pipeline pattern generation with the previous batch's solve: while the solver runs, speculatively generate (into the cross-batch cache) patterns for the next batch against the cluster state after the hint is applied, behind a flag defaulting off. Test that decisions are identical to the non-pipelined run.
- [x] 7. Investigate the ~0.6 pp GpuPacking/GpuClustering gap: `docs/design/notes/2026-11-02-packing-clustering-gap.md` records what is ruled out (score/bind device mismatch, pinned by a test) and what is unchecked. Week 3 tasks written below.

## Week 3

- [x] 1. Score normalization experiment: add an opt-in `-normalize-baselines` flag that min-max normalizes GpuPacking/GpuClustering scores (default off); report calibration deltas in the PR body without changing published numbers. Tests for the normalized ordering.
- [x] 2. GpuClustering truncation audit: unit tests that pin `used` and the four affinity tiers for hand-computed nodes, so any later change to integer arithmetic is deliberate.
- [x] 3. Calibration report: print per-seed max delta per policy in `make calib` output (currently only the worst), with a test on the formatting.
- [x] 4. First-divergence tool: `cmd/gpudefrag-sim calib -trace-placements FILE` writes (pod, node) per placement as CSV for one seed and policy; test that output is deterministic.
- [x] 5. Gang-free option in suite 2: `GangFree` is a bool tie-breaker (no weight exists), so expose it as `-gang-free` (default off) and note it in the suite 2 report header when on. Tests for the header note.
- [x] 5b. Turn the gang-free tie-breaker into a weighted objective term (`GangWeight` milli-GPU per free-whole-node-domain unit) in `mip.evaluate`, default 0 = current behavior, and expose `-gang-weight`. Tests on a cluster where the weight flips a non-tied choice.
- [x] 6. Solver budget sensitivity: record the fraction of batches that hit the time limit in `mip.Stats` and show it in the suite reports. Tests.
- [x] 7. Warm-start comparison: in the suite 1a report, add the share of batches where the solver beat its hint, split by hint source (FGD vs Binpack-or-FGD). Tests.

## Week 4

- [x] 1. Defrag sweep driver (`-grace-sweep` crossed with `-restart-sweep`, `make sweep-defrag`; one row per grid cell in the suite 1b report): a `make sweep-defrag` target (or `-sweep-grid` flag) that runs suite 1b over `-restart-sweep` x `-grace` values and writes one combined table to `results/`. Test the grid expansion and table formatting with a fake runner; do not publish numbers.
- [ ] 2. Break-even report: from a sweep table, compute the smallest restart time at which defrag stops beating its no-defrag baseline on mean allocation (linear interpolation, `n/a` if never). Pure function with table-driven tests.
- [ ] 3. Hysteresis visibility: count plans refused by the benefit-vs-cost rule in `timed` defrag and show the count in the suite 1b report. Tests with a cluster where the rule refuses and one where it accepts.
- [ ] 4. Solver budget plumbing: expose the per-batch CP-SAT time limit as `-solver-budget-ms` in all suites (default unchanged at 500) and print the value in report headers. Tests for flag parsing and header text.
- [ ] 5. Fallback accounting: in suite reports, separate "solver status fallback" from "time limit hit" batch counts and show both. Tests on `mip.Stats`.
- [ ] 6. Pattern-gen reuse across defrag planning: let the defrag planner share the placer's pattern cache for its what-if placements, behind a flag defaulting off. Test identical plans with and without it.
- [ ] 7. Elastic-mode design note: `docs/design/notes/` entry sketching how defrag and node provisioning would be merged into one decision (inputs, objective, what the proto would need), listing open questions. Docs only; no code.
