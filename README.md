# gpudefrag

[![ci](https://github.com/santhosh-svg-kum/gpudefrag/actions/workflows/ci.yml/badge.svg)](https://github.com/santhosh-svg-kum/gpudefrag/actions/workflows/ci.yml)

**Solver-backed GPU scheduling, with reproducible benchmarks against published baselines.**

gpudefrag places GPU pods with a mixed-integer solver (OR-Tools CP-SAT). Three
pieces do the work:

- **Batch placement.** gpudefrag optimizes FGD's own fragmentation measure
  ([Weng et al., USENIX ATC '23](https://www.usenix.org/conference/atc23/presentation/weng))
  jointly over a batch of pending pods, instead of greedily one pod at a time.
- **FGD warm start.** Each solve starts from FGD's answer. gpudefrag validates
  every solver result and keeps whichever of the two is better, so it is
  **never worse than FGD on a batch**.
- **Demand-driven defragmentation.** When a pending pod is blocked only
  because free GPUs are scattered, gpudefrag finds the cheapest set of
  migrations that frees room for it. A migration's cost is the work lost
  since the pod's last checkpoint plus restart time.

It also handles gang (all-or-nothing) jobs with topology domains, and comes
with a discrete-event simulator for comparing schedulers on real traces.

> Status: v1 is a research prototype and simulator. A Kubernetes
> integration (scheduler plugin plus descheduler-style controller on a KWOK
> shadow cluster) is planned for v2.

## Results

Full tables, paired statistics, and caveats are in
[`docs/RESULTS.md`](docs/RESULTS.md). Numbers there are copied from the
generated reports.

- **Faithful baselines.** All six FGD baselines reproduce FGD's published
  curves within ±1.0 pp (FGD itself within 0.10 pp).
- **Beats FGD on FGD's own benchmark.** Batches of 16 reach +0.28 pp GPU
  allocation at 130% load (10 seeds, non-overlapping CIs).
- **Beats FGD under timed overload.** +0.31 ± 0.17 pp allocation at offered
  load 1.1, on held-out seeds.
- **Defrag recovers capacity under overload.** +2.64 ± 2.12 pp allocation
  over Volcano-style binpack in the Helios gang backtest at 1.5× load.
- **Negative results, reported.** gpudefrag loses to FGD with slack capacity
  (load 0.9) and does not beat binpack on gang workloads. Both are diagnosed
  in RESULTS.md, with planned fixes.

## Reproduce

Requirements: Go ≥ 1.25, [uv](https://docs.astral.sh/uv/), about 10 cores.

```bash
make data      # fetch traces (checksum-pinned): Alibaba openb, FGD reference curves, Helios
make test      # Go + Python tests
make calib     # calibration gate: our baselines vs FGD's published curves (±1.0 pp)
make suite1a   # FGD protocol on openb (arrival-only, 130% load), 10 seeds
make suite1b   # timed synthetic workload from openb, with defrag
make suite2    # Helios Venus gang-training backtest with topology domains
```

## How it works

```
                 ┌───────────────── Go ──────────────────────────────┐
 trace ──► workload ──► timed DES (event heap, virtual clock)        │
 (openb,   (FGD order,   │  busy-server scheduler: decide on a snapshot,
  Helios)   synthetic)   │  bind at t+latency, revalidate, back off on conflict
                         ▼                                           │
                 Decider ── baselines: Random DotProd GpuClustering  │
                    │                  GpuPacking BestFit FGD Binpack│
                    └─ mip.Placer:  FGD hint ─► candidates ─► patterns
                         │            (exact frag per node subset)   │
                         ▼                                           │
                 gRPC (proto/gpudefrag/v1/solver.proto) ───────────────┘
                         ▼
                 Python solver service (OR-Tools CP-SAT)
                   Place:  pick one pattern per node, each pod at most once,
                           gangs all-or-nothing, domain-local gangs in one domain;
                           maximize placed GPU, then minimize fragmentation
                   Defrag: min-cost migrations (<= K) that let a blocked pod fit
```

The design decisions that matter:

- **Calibration before comparison.** The baselines are ports of FGD's
  Apache-2.0 simulator, including integer score truncation and its
  lexicographic tie-break. `make calib` must reproduce FGD's published curves
  within ±1.0 pp before any comparison is published. The tolerance is never
  loosened to make a run pass.
- **Pattern (column) formulation.** A direct CP-SAT model of FGD
  fragmentation (reified per-GPU thresholds) could not beat its own warm
  start within 2 s. Go instead enumerates, per node, every subset of up to 3
  batch pods with its exact fragmentation. CP-SAT then solves a small set
  packing over those subsets, which is how cutting-stock problems are usually
  solved.
- **Trust nothing the solver says.** Go replays every answer against the
  cluster (device capacity, GPU count, candidates, gang wholeness, domains)
  and falls back to FGD on errors, timeouts, or invalid output. Every
  fallback is counted by reason.
- **Latency is modeled, not hidden.** The simulator charges measured decision
  time as virtual time (the busy-server model). A slow solver therefore shows
  up as queueing delay.
- **Defragmentation is demand-driven.** It runs only when a pod is blocked by
  fragmentation. It is rate limited, capped at K moves, priced by lost work
  since checkpoint, and guarded by hysteresis. Gang pods are never moved.

## Limitations

- These are simulations. Absolute numbers will differ in production; use them
  to rank configurations, not to predict exact behavior.
- Suite 1b is synthetic. openb's timestamps are snapshot-like (peak
  concurrency is about 1% of capacity), so the timed workload resamples
  openb's (pod, duration) pairs under Poisson arrivals.
- Suite 2's topology (4-node domains) is synthetic, because Helios does not
  publish rack layout.
- GPU sharing is modeled as milli-GPU, as in FGD. MIG and MPS are not modeled.
- Two baselines (GpuPacking, GpuClustering) reproduce about 0.6 pp below
  FGD's published curves. That is inside the gate, but not yet explained.

## Layout

| path | what |
|---|---|
| `model/`, `frag/` | resources, FGD fragmentation measure |
| `trace/`, `workload/` | openb and Helios loaders, FGD submission order, synthetic generator |
| `sched/` | baseline policies, gang placement |
| `sim/`, `timed/` | cluster state, event engine, timed simulator, defrag execution |
| `mip/` | solver client: placer, pattern generation, defragmenter, validation |
| `calib/` | FGD-protocol runner and calibration gate |
| `solver/` | Python CP-SAT service (`uv run pytest` for its tests) |
| `cmd/gpudefrag-sim` | `calib`, `suite1a`, `suite1b`, `suite2` |
| `docs/` | `RESULTS.md` (benchmarks), `design/` (spec and per-milestone plans), `PROGRESS.md` (build log) |

## License and attribution

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE). The baseline and
fragmentation logic is ported from
[hkust-adsl/kubernetes-scheduler-simulator](https://github.com/hkust-adsl/kubernetes-scheduler-simulator)
(Apache-2.0). The traces are
[Alibaba clusterdata](https://github.com/alibaba/clusterdata) (openb) and
[HeliosData](https://github.com/S-Lab-System-Group/HeliosData) (CC-BY-4.0).
They are downloaded on demand and never redistributed.
