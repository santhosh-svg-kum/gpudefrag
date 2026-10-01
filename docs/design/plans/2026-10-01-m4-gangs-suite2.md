# M4: Gangs + topology domains + Helios backtest (suite 2) + README

**Spec:** §6.1 (gang and topology constraints), §7.4, §12.

## Data: HeliosData (SenseTime, SC '21, CC-BY-4.0)
- `data.zip` comes from GitHub (LFS media URL), sha256 `3d22a5f6…32ac`. It is
  fetched by `scripts/fetch-data.sh`.
- We use the Venus cluster:
  - 1,080 GPUs (135 nodes × 8 GPUs).
  - About 125k GPU jobs.
  - Jobs larger than 8 GPUs are 58% of GPU-time.
- Job → pods:
  - `node_num == 1`: a single pod with `gpu_num` whole GPUs.
  - `node_num > 1`: a gang of `node_num` pods with `gpu_num / node_num` GPUs
    each. Jobs that don't divide evenly are dropped and counted.
  - CPU per pod is `cpu_num / node_num` (milli). Memory is not modeled.
- CPU-only jobs are dropped, because suite 2 is about GPUs.

## Topology
- Nodes are grouped into domains of 4 consecutive nodes (an NVLink or IB
  "rack"). This synthetic topology is labeled as such.
- A gang with ≤ 4 pods must land in one domain. Larger gangs may span
  domains.
- The constraint is hard and identical for every policy.

## Timed backtest (suite 2)
- Real submit times and durations from Venus.
- Measurement windows: the 7 busiest days by submitted GPU-hours, preceded by
  3 days of warm-up replay.
- Time compression ×c (c ∈ {1.0, 1.25}) scales inter-arrival gaps to raise
  load: a uniform-growth scenario.
- Metrics:
  - Queueing delay (submit → bind) p50/p95/p99, overall and for gangs.
  - JCT mean.
  - Time-averaged allocation.
  - Pending at end.
  - Migrations and lost GPU-hours.
  - Solve p99.

## Engine changes
- `trace.Job` gains `Gang []model.Pod`: one pod for singles, n pods for a gang.
- `model.NodeRes` gains `Domain string`.
- Decider: `Decide(c, jobs []Request) []*Decision`. A Request is a job's pods;
  a Decision holds a choice per pod or is nil. Binding is all-or-nothing, and
  each pod is revalidated.
- Single-pod deciders (FGD, BestFit) and `mip.Placer` adapt to it.

## Policies
- **VolcanoGangBinpack** (modeled):
  - FIFO jobs, all-or-nothing, with backfill (blocked jobs don't stop
    later ones).
  - Binpack node order (most allocated first).
  - A gang tries each feasible domain in binpack order.
- **FGD-gang:** the same loop, with nodes chosen by FGD score.
- **gpudefrag:**
  - The pattern model gains gangs. A pod is covered exactly `a[j]` times.
    Pattern `u` containing a gang pod on node n implies `d[j, dom(n)]`, and
    `Σ_D d[j,D] = a[j]`.
  - The objective places weight first, then fragmentation.
  - The hint is FGD-gang.
- **+defrag:** the existing single-pod defragmenter. For example, it frees an
  8-GPU node for a blocked 8-GPU job by moving 1-GPU jobs. Gang pods are
  never moved.

## Tests
- Helios loader: gang splitting, uneven jobs dropped, sums.
- Gang binding: all-or-nothing on conflict; domain constraint honored by
  every policy.
- Solver: gangs with brute force on tiny cases; never a partial gang; domain
  respected.
- Placer: gang hint passes validation, and partial-gang solver output is
  rejected.

## README
- What gpudefrag is and the claims, with numbers taken from the reports.
- How to reproduce: `make data calib suite1a suite1b suite2`.
- Architecture diagram (ASCII), design notes, limitations, license and
  attribution.
