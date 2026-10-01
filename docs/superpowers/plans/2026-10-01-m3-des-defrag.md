# M3: Timed DES + busy-server latency + demand-driven defrag + suite 1b

**Spec:** §4, §6.2, §7.2, §10. **Design doc:** §7 (engine), §10.1 (parametric
generation).

## Finding that shapes M3
openb's creation and deletion timestamps are snapshot-like. Peak concurrent
GPU demand is about 1% of capacity, so an exact timed replay never stresses
the cluster. Suite 1b therefore uses a **synthetic, labeled** workload:
- Pods are bootstrapped as (resource, duration) pairs from openb.
  `deletion − creation` gives the duration; it is capped at 7 days, and rows
  with duration ≤ 0 are dropped.
- Arrivals are Poisson with rate λ = ρ·capacity / E[gpu·dur], for offered
  loads ρ ∈ {0.9, 1.0, 1.1}.
- 20k arrivals per run, seeded.

Results are reported separately from suite 1a, never averaged with it.

## Engine (`sim/engine.go`)
- **Event heap:** keyed by `(time, rank, seq)`. Ranks run
  Completion < Arrival < BindApply < Session < DefragTick, so freed
  capacity is visible to the same-instant scheduler. Pushing into the past
  panics.
- **Pending queue:** FIFO.
- **Busy-server scheduler:** a session starts only when the scheduler is idle
  and pods are pending, and only after a state change (an arrival, a
  completion, or a bind conflict). That is the idle skipping.
- **Session latency:**
  - The session decides on the current state and applies binds at `t + L`.
  - `L = scale × measured wall time of the decision`, with a floor of 1 ms
    per pod for baselines.
  - Binds are revalidated at apply time. A failed bind is requeued at the
    head of the queue and increments `bind_conflicts`.
- **Policies:** baselines decide pod-by-pod within a session (up to B pods).
  gpudefrag decides the batch via `mip.Placer`.
- **Completion:** at `start + remaining`. An evicted pod's remaining work grows
  by the lost work since its last checkpoint, plus `restart_s`.
- **Metrics:**
  - GPU-pod pending latency p50/p95/p99.
  - Time-averaged GPU allocation.
  - Blocked-pod-seconds.
  - Pods still pending at the end.
  - Migrations and lost GPU-seconds.
  - Solve wall p50/p99.
  - bind_conflicts.

## Defrag (`mip/defrag.go`, `solver/defrag.py`)
- **Trigger:** after a session, if some pending GPU pod is **blocked by
  fragmentation**. Blocked means it fits no node, but the total free GPU milli
  on accessible nodes is at least its demand, and some accessible node could
  host it if movable pods left. Rate limited to one DefragTick per 60 s of
  virtual time.
- **Scope:** one target per tick, the oldest blocked pod.
  - Targets are the 16 candidate nodes needing the least movable mass.
  - Movable pods are running non-gang GPU pods on target nodes.
  - Receivers are the top-64 nodes by free GPU.
- **CP-SAT model (direct, device level):**
  - Choose exactly one target node, with devices for the blocked pod.
  - Each movable pod either stays or moves to one receiver (devices
    included).
  - Device, CPU, and memory capacity constraints, with at most `K_mig` moves.
  - Minimize Σ cost.
  - Cost = gpuMilli × (now − lastCheckpoint) + gpuMilli × restart_s.
    Checkpoint interval 30 min, with a per-pod phase drawn from the seed.
- **Hysteresis (Go):** execute only if
  `cost ≤ benefit / (1 + margin)`, where
  `benefit = gpuMilli_blocked × BenefitHorizon` (default 1 h, margin 0.2).
- **Execution:**
  - Moves happen at t: unbind from the source, then bind on the destination
    (reserved, so no race).
  - The blocked pod binds on the target at t. Moved pods pay restart time and
    lost work.
- **Go validation:** replay the moves on a scratch cluster. Invalid output or
  failed hysteresis means no plan, and is counted.

## Suite 1b (`gpudefrag-sim suite1b`)
- **Variants:** FGD, BestFit, gpudefrag (B=16), FGD+defrag, gpudefrag+defrag.
- 10 seeds per load; mean ± 95% CI; `report.md`.

## Tests
- **Engine:**
  - Ordering and tie ranks.
  - Panic on scheduling into the past.
  - Busy-server example (design doc §7.5: job2 waits 4 s).
  - Completion frees capacity.
  - A bind conflict requeues the pod.
  - Determinism (same seed gives the same metrics).
- **Workload generator:** the offered load is within 5% of target on a long
  run; durations are bootstrapped from the trace.
- **Defrag:**
  - Python: brute-force minimum cost on tiny cases. `max_moves`, gang, and
    do-not-disrupt are respected. The infeasible case returns no plan.
  - Go: validation rejects bad plans, hysteresis blocks expensive plans, and
    the two-node textbook case moves one pod to unblock an 8-GPU-style job.
