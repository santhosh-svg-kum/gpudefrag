# M1: Foundation + FGD-faithful baselines + calibration gate

> Executed natively (user authorized autonomous execution, 2026-09-30).

**Goal:** Reproduce FGD's (ATC '23) published allocation curves for six
policies on the openb default trace within ±1.0 pp, using our own Go code.

**Spec:** `docs/superpowers/specs/2026-09-30-gpupack-design.md` (§5, §7.1, §7.3)

**Reference:** `hkust-adsl/kubernetes-scheduler-simulator` @
`8f3d6417353c5083c6d56617f255fadd8dc306bc` (Apache-2.0).

**Deviation from spec:** M1 replays arrivals in order with no clock, because
suite 1a has no time dimension. The event-heap DES lands in M3, where timed
replay first needs it.

## Global constraints
- Go module `gpupack`, stdlib only in M1. Integer units: milli-CPU, MiB, milli-GPU.
- Gate: workload `openb_pod_list_default`, tune 1.3, seeds 42–51. Policies:
  Random, DotProd, GpuClustering, GpuPacking, BestFit, FGD. Points:
  50, 80, 100, 110, 120, 130. Tolerance ±1.0 pp on the seed mean.
  **Never loosen it.**
- Data is fetched by `scripts/fetch-data.sh` with sha256 pins and is never
  committed.

## FGD semantics to replicate (from reading the reference code)
1. **Trace → pod:**
   - `gpu_milli` is clamped to 1000. `num_gpu=0` means CPU-only (milli 0).
   - CPU 0 becomes 100m and memory 0 becomes 200 MiB (kube nonzero defaults).
2. **Ordering:**
   - `rng := rand.New(rand.NewSource(seed))`, then one `rng.Int()` (FGD logs a
     draw).
   - Sort pods by name, then `rng.Shuffle`.
   - Tune to 1.3 × cluster milli-GPU. Tune-up appends random resamples of the
     name-sorted originals, named `<name>-tuned-<i>`. It breaks when
     `total + pod.GpuMilli` (per-GPU, a quirk) exceeds the target, and
     otherwise adds the pod's total milli. Tune-down removes random pods.
3. **Typical pods:**
   - Key = (cpu, gpuMilli, gpuNum, gpuType), memory ignored. CPU pods are
     included.
   - Sort by count descending; ties broken by reverse of `Less`.
   - Take the top classes until the cumulative count reaches ≥ 95%, then
     renormalize.
4. **Filter:**
   - CPU and memory fit.
   - GPU pods: the node must have GPUs and an accessible type (`a|b`
     alternatives), and at least `gpuNum` GPUs must each have
     `left ≥ gpuMilli`.
5. **Scheduling:**
   - If exactly one feasible node exists, take it without scoring.
   - Otherwise score every feasible node and pick the max. **Ties go to the
     lexicographically smallest node name** (patched `selectHost`).
6. **GPU selection:**
   - Random uses random-fit (reservoir). DotProd, GpuClustering, GpuPacking,
     and BestFit use best-fit (smallest left ≥ milli, first index).
   - FGD uses the argmax of its own score over GPUs.
   - Whole-GPU pods take the first fully free GPUs by index.
7. **Scores:**
   - Ported verbatim from `pkg/simulator/plugin/*_score.go`.
   - Only BestFit normalizes (min-max to 0–100).
   - FGD = `int64(sigmoid((before−after)/1000)·100)`. For a share pod, after
     is computed per GPU, taking the max. Otherwise after = `Sub` (pack into
     the smallest-left GPUs).
8. **Curve:**
   - Take a sample (arrived total milli, used milli) initially and after
     every pod, including failures.
   - `arrive% = roundHalfEven(arrived/totalGpus/10)` and
     `alloc% = round2(used/totalGpus/10)`.
   - For each k in 0..130: the mean of samples at k, or failing that the
     ±1 window. Round to 2 decimals.

## Tasks
1. `model/` resources: `PodRes`, `NodeRes`, `Sub`, `Accessible`, affinity tags.
2. `trace/` openb loader, plus `scripts/fetch-data.sh` and `make data`.
3. `frag/` fragmentation measure. Tests use FGD's own unit-test values.
4. `workload/` `Prepare` (shuffle and tune) and `TypicalPods`.
5. `sim/` cluster state: `Fits`, `Feasible`, `Bind`, used-milli accounting,
   invariants.
6. `sched/`: policies, GPU selectors, and `Place`.
7. `calib/`: arrival runner, discretizer, reference loader, gate.
8. `cmd/gpupack-sim calib`, the Makefile, and a real gate run.

## Review focus
- Pods with a GPU type no node has: counted as arrived, fail cleanly.
- Multi-GPU pods larger than every node: filtered out, no panic.
- Malformed trace rows (share multi-GPU, gpu>0 with milli 0): loader errors
  naming the row.
- Missing reference seed or policy, or a NaN gate point: the gate fails and
  never passes vacuously.
- Parallel runs must not share mutable node state. Repeated calib runs must
  give identical curves.
