# M2: CP-SAT solver service + MIP placement + suite 1a report

**Goal:** Batch MIP placement that optimizes FGD's own fragmentation measure
over a batch of pending pods, warm-started from FGD. It is never worse than
FGD on the exact Go-evaluated objective, and is benchmarked on suite 1a
against the calibrated baselines.

**Spec:** §6.1, §7.1, §8, §9, §10.

## Exact linearization of FGD's node fragmentation
For a node with GPU free milli L_g, CPU left C, and typical classes m (Pct_m):

- frag(n) = Σ_m Pct_m · Total − Σ_{m ∈ GPU classes accessible on n} Pct_m · ok_m · U_μ(m)
- Total = Σ_g L_g
- U_μ = Σ_g [L_g ≥ μ] · L_g
- ok_m = [#{g : L_g ≥ μ_m} ≥ num_m] ∧ [C ≥ cpu_m]

Derivation: Q3 contributes nothing, Q2-within-Q3 is Total − U, and every
other kind contributes Total. CPU-only classes and inaccessible classes are
constant (Pct · Total). In CP-SAT:
- `b[g,μ] ⇔ L_g ≥ μ` (reified).
- `w[g,μ] = b·L_g` (via OnlyEnforceIf).
- `host[m] ⇔ Σ_g b[g,μ] ≥ num`, and `cpu[m] ⇔ C ≥ cpu_m`.
- `ok = host ∧ cpu`, and `z_m = ok·U_μ`.
- Weights are `round(Pct · 1e6)`.

## Lexicographic solve
1. Phase 1: maximize Σ (gpuTotalMilli + 1) · placed.
2. Phase 2: fix placed ≥ the phase-1 value, then minimize Σ frag over touched
   nodes.

The time budget is split 40/60. Both phases are hinted (FGD, then the phase-1
solution).

## Go side (`mip/`)
- Hint: sequential FGD over the batch on a scratch copy of the cluster.
- Candidates: the top-K feasible nodes by FGD score on the pre-batch state,
  plus the hint node.
- Validation: apply the solver's assignment to a scratch copy (device-level
  capacity, GPU count, CPU and memory).
- Choice: compare against the hint on (placed weight, exact `frag.NodeScore`
  sum) and keep the better one.
- Fallback: RPC error, timeout, invalid output, or a worse result means the
  hint is used. `fallback{reason}` is counted.
- `StartLocal()` spawns the Python server via `uv run`, on a free port.

## Suite 1a harness
- `gpupack-sim suite1a` runs the same seeds and protocol as calib.
- Policy `gpupack` places in batches of B consecutive arrivals.
- Samples are taken at the end of each batch.
- Outputs per-seed curves, seed-mean ± 95% CI, failed pods, and solve-time
  p50/p99, plus `report.md`. Reported variants:
  - B=1, which isolates "FGD without score quantization".
  - B=16 and B=64.
  - Time limits of 100 ms, 500 ms, and 2 s.

## Tests
- Python:
  - Brute-force optimality on ≤ 5 pods × 3 nodes × 2 GPUs.
  - The frag expression equals the Python reference frag on random states
    (hypothesis).
  - The solution is always feasible.
  - Placed ≥ the hint's placed count.
- Go:
  - The validation rejects bad assignments: over-commit, wrong GPU count, a
    node not in the candidates.
  - Fallback is used when the solver is unreachable.
  - Contract test against the real server (skipped if `uv` is missing).
