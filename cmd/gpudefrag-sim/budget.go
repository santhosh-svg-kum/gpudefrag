package main

import (
	"fmt"
	"strconv"
	"strings"
)

// defaultBudgetMs is the per-batch CP-SAT time limit used when
// -solver-budget-ms is not given.
const defaultBudgetMs = 500

// budgetNote is the report-header sentence recording the solver budget, so a
// report says which budget produced its numbers.
func budgetNote(ms int) string {
	return fmt.Sprintf(". Solver budget: %d ms per batch", ms)
}

// budgetVariants expands every gpudefrag variant without an explicit t= option
// into one variant per sweep value ("gpudefrag:b=16" -> "gpudefrag:b=16:t=250ms").
// Baselines and variants with an explicit t= are kept as is; an empty sweep is
// a no-op.
func budgetVariants(vs []string, sweep string) ([]string, error) {
	if sweep == "" {
		return vs, nil
	}
	var ms []int
	for _, s := range strings.Split(sweep, ",") {
		n, err := strconv.Atoi(strings.TrimSpace(s))
		if err != nil || n <= 0 {
			return nil, fmt.Errorf("bad -budget-sweep value %q", s)
		}
		ms = append(ms, n)
	}
	var out []string
	for _, v := range vs {
		if !strings.HasPrefix(v, "gpudefrag") || strings.Contains(v, ":t=") {
			out = append(out, v)
			continue
		}
		for _, n := range ms {
			out = append(out, fmt.Sprintf("%s:t=%dms", v, n))
		}
	}
	return out, nil
}

// budgetSweepSection is one row per solver variant: its time limit, mean
// alloc@130%, and the shares of batches that hit the limit or fell back on
// solver status, over all seeds. Empty when no variant ran the solver.
func budgetSweepSection(vs []variant, results map[string][]runSummary) string {
	var b strings.Builder
	for _, v := range vs {
		if v.Limit == 0 {
			continue
		}
		var a []float64
		solved, hit, fall := 0, 0, 0
		for _, r := range results[v.Name] {
			a = append(a, r.Curve[130])
			solved += r.Solved
			hit += r.BudgetHit
			fall += r.NoSolution + r.StatusOther
		}
		if b.Len() == 0 {
			fmt.Fprintf(&b, "\n## Solver budget sweep\n\n")
			fmt.Fprintf(&b, "Per time limit: mean alloc@130%% ± 95%% CI over seeds, batches stopped by the limit (FEASIBLE) of solved, and status fallbacks (timeout without solution or other) as a share of all solver batches.\n\n")
			fmt.Fprintf(&b, "| variant | budget ms | alloc@130%% | time limit hit / solved | share | fallback share |\n|---|---|---|---|---|---|\n")
		}
		m, ci := meanCI(a)
		hitS, fallS := "–", "–"
		if solved > 0 {
			hitS = fmt.Sprintf("%.1f%%", 100*float64(hit)/float64(solved))
		}
		if solved+fall > 0 {
			fallS = fmt.Sprintf("%.1f%%", 100*float64(fall)/float64(solved+fall))
		}
		fmt.Fprintf(&b, "| %s | %d | %.2f ± %.2f | %d / %d | %s | %s |\n", v.Name, v.Limit.Milliseconds(), m, ci, hit, solved, hitS, fallS)
	}
	return b.String()
}
