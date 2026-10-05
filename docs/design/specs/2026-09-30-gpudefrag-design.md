# gpudefrag v1 — Design Spec

- **Date:** 2026-09-30
- **Status:** Draft, awaiting review
- **License:** Apache-2.0
- **Origin:** A personal open-source project, unrelated to any employer.
  It implements the modeled tier (phases 2–4, scoped down) of the author's
  own draft design for simulating Kubernetes scheduling and autoscaling, and
  adds a solver-based GPU scheduler and defragmenter on top of it. All
  results use public traces (Alibaba openb, SenseTime Helios).

## 1. Purpose and success criteria

gpudefrag is an open-source, fragmentation-aware GPU scheduler that uses a
constraint solver (OR-Tools CP-SAT), plus a discrete-event simulator (DES)
to benchmark it reproducibly against published baselines.

v1 is successful when one command per suite produces a report that supports a
claim of this form, with confidence intervals:

> On the Alibaba openb trace, gpudefrag allocates X pp more GPUs than FGD at
> 100–130% arrived load; with defrag enabled it unblocks Y% of
> fragmentation-blocked pods at Z GPU-seconds of lost work; p99 solve time is
> W ms at a 500 ms budget. On gang training traces it cuts p95 queueing delay
> by V% versus Volcano gang + binpack.

We publish whatever the numbers turn out to be, including no improvement.
Comparisons are reported only after the calibration gate (§7.3) passes.

## 2. Scope

**In v1**
- A Go DES engine with a virtual clock, deterministic ordering, invariants, and
  a busy-server scheduler latency model.
- A fixed-size cluster with fractional (GPU-sharing) and whole-GPU pods, gang
  jobs, and topology domains.
- Baseline schedulers: Random, DotProd, BestFit, GpuPacking, GpuClustering, FGD,
  FGD-gang, and VolcanoGangBinpack (modeled).
- gpudefrag placement: FGD warm start, then CP-SAT batch improvement.
- gpudefrag defrag: demand-driven, minimum-cost migration on the fixed cluster.
- Benchmark suites 1a, 1b, and 2, with a calibration gate and reports.

**Out of v1 (later)**
- KWOK shadow twin with real kube-scheduler, Volcano, Karpenter, and KEDA (v2).
- Elastic cluster: provisioning and consolidation with prices (v2), which is
  where defrag and Karpenter disruption merge into one optimization.
- Load-driven inference autoscaling with HPA or KEDA (v3).
- MIG, MPS, and time-slicing representation (open question).
- Embedded real kube-scheduler or Karpenter Go packages (doc phase 5).

## 3. Architecture

```
gpudefrag/
  go.mod                  module github.com/<owner>/gpudefrag
  cmd/gpudefrag-sim/        CLI: run an experiment spec → results
  sim/                    DES: event heap, virtual clock, cluster state, invariants, metrics
  model/                  Node, GPU, Pod, Gang, Domain types; integer units
  trace/                  loaders: openb (suite 1), helios/philly (suite 2) → common schema
  frag/                   FGD fragmentation measure + task-size distribution
  sched/                  interfaces (Scheduler, Defragmenter)
    baselines/            random, dotprod, bestfit, gpudefraging, gpuclustering, fgd, fgdgang, volcanogang
    mip/                  Go client: build request, call solver, validate, fall back to FGD
  proto/gpudefrag/v1/       solver.proto (the swap boundary)
  solver/                 Python service (uv project): grpc server + CP-SAT models
    gpudefrag_solver/place.py, defrag.py, server.py
    tests/                brute-force optimality + hypothesis property tests
  bench/                  experiment YAML specs, runner, report generator (plots + markdown)
  calib/                  FGD reproduction (calibration gate)
  Makefile                make test, make bench-suite1, make bench-suite2, make calib
```

Units are integers everywhere: GPU in milli-GPU (1 GPU = 1000), CPU in
millicores, memory in MiB. These match the openb columns `gpu_milli`,
`cpu_milli`, and `memory_mib`.

### 3.1 Plugin interfaces (Go)

```go
type Scheduler interface {
    Name() string
    // Session decides on a snapshot. The engine applies binds at now+latency.
    Session(pending []*model.Pod, s *sim.ClusterState, now sim.Time) []sim.BindAttempt
}
type Defragmenter interface {
    // Plan is called only when blocked pods exist. It returns ordered moves.
    Plan(blocked []*model.Pod, s *sim.ClusterState, now sim.Time) *sim.DefragPlan
}
```

Interfaces for v2 (`NodeAutoscaler.Provision/Disrupt`, `CloudProvider`) are
reserved for v2, as the author's simulation design draft describes. v1 neither defines nor stubs them.

## 4. Simulation engine

This follows the author's simulation design draft, restricted to v1 events.

- **Events:** `PodArrived`, `PodCompleted`, `SchedulerSession`, `BindAttempt`,
  `DefragTick`, `EvictionDone`, and `PodRestarted`.
- **Queue key:** `(time, classRank, sourceSeq, insertionSeq)` in a min-heap.
  Arrivals rank before controller ticks, and an event scheduled in the past
  triggers a panic.
- **Busy-server latency:** a session starts only when the scheduler is idle.
  Its binds are applied at `now + L`, where L is one of:
  - `constant`
  - `measured`: the wall-clock duration of `Session()`, times a scale factor
  - `deterministic`: the solver's reported deterministic time, times a factor
- **Bind revalidation:** at bind time the engine rechecks capacity, gang
  completeness, and topology. A failed check requeues the pod and increments
  `bind_conflicts`. Gang binds are all-or-nothing.
- **Idle skipping:** a session is scheduled only while pods are pending.
- **Invariants checked after every event:**
  - No over-capacity node or GPU.
  - No partial gang bound.
  - No defrag move violating a job's disruption budget.
  - No event scheduled in the past.
  - A violation fails the run.
- **Determinism:** all randomness comes from one seeded RNG per run. Given the
  same trace, config, and seed, the metrics hash must be identical, except
  under `measured` latency, which is documented as non-deterministic.

## 5. Fragmentation measure (shared by FGD and the MIP)

This follows FGD (Weng et al., USENIX ATC '23). The target workload is a
discrete distribution over task-size classes. Each class *m* has a popularity
`π_m` and a demand (GPU milli, whole-GPU count, CPU, memory). Classes are built
from the trace by bucketing into at most 8 classes, for example {≤250m, ≤500m,
<1000m, 1, 2, 4, 8 GPUs, CPU-only}.

Node fragmentation is `F(n) = Σ_m π_m · U(n, m)`, where `U(n, m)` is the GPU
capacity on node n that is unusable by a task of class m:
- If the task can't fit on the node at all (GPU, CPU, or memory), U is all of
  the node's free GPU.
- Otherwise, U is the free GPU on GPUs that individually can't host the task's
  per-GPU demand.

FGD's score for a candidate node is `F(n after placing p) − F(n before)`, and
FGD picks the minimum. The exact definition is copied from FGD's open-source
implementation (`hkust-adsl/kubernetes-scheduler-simulator`, Apache-2.0) and
verified by the calibration gate.

## 6. gpudefrag algorithms

### 6.1 Placement (each session)

1. **Batch:** up to `B` pending pods (default 200), ordered by priority and
   then age. Gangs are included whole or not at all.
2. **Candidates:** for each pod, the top-`K` feasible nodes by FGD score
   (default 32). For a gang, every node in its feasible domains. Nodes outside
   the candidate set keep their fragmentation unchanged, so they are left out
   of the model.
3. **Hint:** run FGD sequentially over the batch to get an initial assignment.
4. **Solve:** send `PlaceRequest` to the solver with time limit `T` (default
   500 ms).
5. **Validate:** the Go client checks every returned assignment against the
   snapshot. On RPC error, timeout (`T` plus 200 ms), infeasible or invalid
   output, or a failed validation, it uses the FGD hint and increments
   `fallback_total{reason}`.

**CP-SAT model (`solver/place.py`)**
- `x[p,n] ∈ {0,1}` for candidate pairs. `s[p] = Σ_n x[p,n] ≤ 1`.
- Fractional pods (`0 < gpu_milli < 1000`): `y[p,n,g]`, with
  `Σ_g y[p,n,g] = x[p,n]` and per-GPU capacity
  `Σ_p r_p·y[p,n,g] ≤ free[n,g]`.
- Whole-GPU pods: fully free GPUs on a node are interchangeable, so they are
  modeled as a count. Let `F_n` be the number of fully free GPUs before the
  session and `t[n,g]` mark a fully free GPU g taken by a fractional pod. Then
  `Σ_p k_p·x[p,n] + Σ_g t[n,g] ≤ F_n`. Symmetry breaking: fully free GPUs are
  used by fractional pods in index order.
- CPU and memory: linear capacity constraint per node.
- Gangs: `s[p] = a[j]` for every p in gang j. Topology:
  `Σ_D d[j,D] = a[j]` and `x[p,n] ≤ d[j, dom(n)]`.
- Fragmentation after placement for each touched node and class m: an
  indicator `blocked[n,m]` ("class m no longer fits on n"), plus per-GPU
  indicators for fractional classes, linked by reified constraints on the
  node's post-placement free resources. `Frag(n)` is the π-weighted linear
  expression from §5.
- Objective (single weighted sum; weights are chosen so each term strictly
  dominates the next, computed from instance bounds):
  1. maximize `Σ_p prio_p · gpu_p · s[p]`
  2. minimize `Σ_n Frag(n)` over touched nodes
  3. minimize the number of previously empty nodes that become used
- The FGD assignment is passed via `AddHint`. Solver parameters:
  `max_time_in_seconds = T`, `num_workers = 8`. For reproducible benchmarks,
  `max_deterministic_time` is used instead.

**Guarantee:** the returned objective is ≥ the hint's objective, and the
fallback is the hint itself. So gpudefrag's placement is never worse than FGD's
on the session objective. This is checked by property tests (§9).

### 6.2 Defrag (demand-driven, fixed cluster)

- **Blocked pod:** a pending pod (or gang) that doesn't fit on any node (or
  domain) while the cluster's total free resources would cover it. The engine
  checks for blocked pods after each session that leaves pods pending, rate
  limited to one `DefragTick` per `defrag_interval` (default 60 s virtual).
- **Movable pods:** running, non-gang, not do-not-disrupt, under their job's
  disruption budget, and on candidate source nodes (nodes in the blocked pod's
  feasible set).
- **Variables:** placement variables for blocked pods (as in §6.1), plus
  `m[q,n']` for each movable pod q and target n' (including "stay").
- **Constraints:**
  - Capacity after all moves.
  - Every pod placed exactly once.
  - At most `K_mig` moves per tick (default 4).
- **Migration cost:** `restart_s · gpu_q + lost_work_q`, where
  `lost_work_q = gpu_q · (now − last_checkpoint_q)`. The checkpoint interval
  comes from the pod's class (configurable; suite 1 default 30 min, phase
  randomized by seed).
- **Objective:** maximize `Σ value(blocked placed) − Σ cost(moves)`. A plan is
  executed only if `benefit ≥ (1 + margin) · cost` (hysteresis, default
  margin 0.2).
- **Execution in the DES:**
  - The freed capacity is reserved for the target blocked pod (a nominated
    node), so other pods can't take it.
  - Moves run in order. Each is an eviction followed by restart on the target
    after `restart_s`, with progress rolled back to the last checkpoint.
  - If the target disappears or a move fails revalidation, the remaining moves
    are aborted.
- **Baseline for defrag:** none in the literature matches this exactly. We
  compare gpudefrag with and without defrag, and against FGD plus a modeled
  descheduler `HighNodeUtilization` rule at equal migration budget.

## 7. Benchmarks

### 7.1 Suite 1a — FGD protocol replication (headline)
- **Data:** Alibaba `cluster-trace-gpu-v2023`: `openb_node_list_gpu_node.csv`
  and the openb pod list variants, checksum-pinned and downloaded by
  `make data`.
- **Protocol:** as in the FGD repo's `experiments/`:
  - Workload is inflated to 130% of cluster GPU capacity.
  - Pods are submitted in sampled order, with no departures.
  - 10 seeds (42–51).
  - The exact submission and inflation procedure is ported from their code.
- **Metric:** GPU allocation ratio, or unallocated GPU %, against arrived
  workload (% of capacity). Also fragmentation ratio and failed pods.
- **Policies:**
  - Random, DotProd, BestFit, GpuPacking, GpuClustering, FGD
  - gpudefrag placement at T ∈ {100 ms, 500 ms, 2 s}
- **Arrivals in batches:** suite 1a has no clock. "Session batch" means the
  next B arrivals, which is a deliberate difference from FGD's one-at-a-time
  submission. We also report gpudefrag with B=1, which isolates the solver gain
  from the batching gain.

### 7.2 Suite 1b — timed replay with departures
- Uses the openb `creation_time` and `deletion_time` for arrival and duration.
- Pods that never ran get a sampled duration.
- Measures pending latency (p50/p95/p99), allocation over time, blocked pods,
  defrag moves, lost GPU-seconds, solve-time distribution, and fallback count.
- Policies: FGD, gpudefrag placement, and gpudefrag placement + defrag.

### 7.3 Calibration gate
`make calib` runs our FGD and baselines on suite 1a. It passes when each
policy's mean allocation curve is within ±1.0 pp of the published or reproduced
FGD results at load points 50, 80, 100, 110, 120, and 130%. If we can't obtain
reference numbers, we run their simulator ourselves in Docker and compare
against that. Reports refuse to render comparisons unless the gate's result
file is present and passing.

### 7.4 Suite 2 — gang training
- **Trace:** HeliosData (SenseTime), with Philly as fallback. Jobs requesting
  ≥ 8 GPUs become gangs of 8-GPU pods. Smaller jobs are single pods.
- **Cluster:** synthetic. N nodes × 8 GPUs, in domains of 4 nodes. N is sized
  so the trace reaches 70–90% utilization.
- **Policies:** VolcanoGangBinpack (modeled), FGD-gang, gpudefrag placement, and
  gpudefrag placement + defrag.
- **Metrics:** queueing delay (p50/p95/p99), JCT, allocated GPU %, stranded
  GPUs, migrations, lost GPU-seconds, solve time.

### 7.5 Rigor
- Paired runs: same trace and seed across all policies.
- At least 10 seeds. Mean ± 95% CI for every reported number.
- Hardware and solver version are stamped into every report.
- Outputs: `results/<experiment>/<run>.parquet` plus `report.md` and PNG plots.

## 8. Solver service contract (`proto/gpudefrag/v1/solver.proto`)

- `Place(PlaceRequest) → PlaceResponse`
  - Request: nodes (with per-GPU free milli, CPU, memory, domain), pods, gangs,
    candidate lists, class distribution, hint, time limit, determinism flag.
  - Response: assignments, objective, hint objective, solver status, wall
    time, deterministic time.
- `Defrag(DefragRequest) → DefragResponse`: the same pattern, returning an
  ordered list of moves.
- `Health()`.

Go stubs are generated with `buf` (run via `go run`) and checked in. Python
stubs are generated with `grpcio-tools`. The proto is the swap boundary: any
solver (HiGHS, Gurobi) can implement it.

## 9. Testing

- **Go:**
  - Event ordering and tie-break tests.
  - Invariant tests (each invariant has a test that triggers it).
  - Golden determinism test.
  - Table tests for each baseline on tiny hand-built clusters.
  - Fragmentation-measure tests on worked examples.
  - Validation and fallback tests for the mip client, using a fake solver
    that returns garbage, times out, or errors.
- **Python:**
  - Brute-force optimality: on instances with ≤ 6 pods × 3 nodes, the
    CP-SAT optimum equals exhaustive enumeration.
  - `hypothesis` property tests: every solution is feasible, and its
    objective is ≥ the hint's objective.
  - Defrag: never exceeds K_mig, never moves gang or do-not-disrupt pods,
    respects the hysteresis rule.
- **Contract test:** the Go client against the real Python server on a fixture
  request.
- **Long checks:** `make calib` and a small smoke version of each suite.

## 10. Error handling summary

| Failure | Behavior | Signal |
|---|---|---|
| Solver unreachable or timeout | Use FGD hint | `fallback_total{reason=rpc}` |
| INFEASIBLE / MODEL_INVALID | Use FGD hint | `fallback_total{reason=status}` |
| Assignment fails Go validation | Use FGD hint, log the instance to `bugs/` | `fallback_total{reason=invalid}` |
| Bind revalidation fails | Requeue the pod | `bind_conflicts` |
| Defrag move fails or target gone | Abort the rest of the plan, release the reservation | `defrag_aborts` |
| Invariant violated | Fail the run | run status `invariant_violation` |

## 11. Risks

- **Our FGD might not match theirs.** Mitigation: port its scoring code
  directly, and the calibration gate is mandatory.
- **CP-SAT may be too slow at B=200 and K=32.** Mitigation: B and K are
  tunable, the fragmentation term is restricted to touched nodes, and a
  solve-time sweep is part of the report.
- **The fragmentation linearization may diverge from FGD's exact measure.**
  Mitigation: a unit test compares the model's `Frag(n)` on the solved
  assignment with Go's `frag.F(n)`.
- **Batching adds an advantage FGD didn't get.** Mitigation: report B=1
  separately.
- **Trace licenses.** Alibaba clusterdata and HeliosData are research datasets
  that are downloaded, not redistributed.

## 12. Open questions
- Final project name and GitHub owner. `gpudefrag` is a placeholder.
- Whether suite 2 uses Helios or Philly. Decided by which one has usable
  per-job GPU counts and submit times once downloaded.
