# GpuPacking / GpuClustering ~0.6 pp calibration gap: what is ruled out

Context: RESULTS.md section 0 shows our GpuClustering and GpuPacking ports
0.59 and 0.67 pp below FGD's published curves, while Random, DotProd, BestFit
and FGD are within 0.18 pp. The gap is inside the +/-1.0 pp gate and is not
yet explained. This note records what was checked in `sched/baselines.go`
and `sched/sched.go`. It does not claim a fix.

## Ruled out (by reading and by test)

- **Score/bind device mismatch in GpuPacking.** `Score` simulates the
  devices it would use with `SortedGpuIdx(true)` (ascending milli-left,
  stable); binding uses `BestFitGPUs` (least milli-left, first index on
  ties). For share pods these always choose the same device. Pinned by
  `TestGpuPackingScoredDeviceMatchesBoundDevice` (2000 random nodes). Whole-GPU
  pods use `ExclusiveGPUs` in both paths.
- **Tie-break among equal scores.** `selectHost` takes the lexicographically
  smallest node name, as in FGD's patched kube-scheduler. Both policies emit
  coarse integer scores, so ties are common and this rule decides many
  placements; it is shared with the policies that do match, so it is an
  unlikely cause on its own.

## Not checked (needs upstream source or the reference run)

- Whether upstream normalizes GpuPacking/GpuClustering scores. Ours returns
  raw scores in [0,100]; BestFit is min-max normalized. Normalization changes
  which scores tie.
- Integer truncation order in `GpuClustering` (`MaxScore/4*(...)/MaxScore`
  evaluates left to right) and the idle-node constant in GpuPacking
  (`max(MaxScore/3-free, free)`).
- Whether upstream's Affinity tags count pods or GPUs.
- Per-seed trace order differences: the gap is 0/10 exact on both policies,
  but also 0/10 on Random, so exactness is not informative.

## Suggested next step

Diff per-pod placements for a single seed against a reference run of the
upstream simulator (needs its source and the reference trace), starting with
the first pod where the chosen node differs.
