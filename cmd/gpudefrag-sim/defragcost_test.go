package main

import (
	"strings"
	"testing"
	"time"
)

func TestDefragRPCSummary(t *testing.T) {
	w := []time.Duration{10 * time.Millisecond, 20 * time.Millisecond, 30 * time.Millisecond, 40 * time.Millisecond}
	p50, p99 := pctl(w, 0.5), pctl(w, 0.99)
	if p50 != 30 || p99 != 40 {
		t.Fatalf("p50=%v p99=%v", p50, p99)
	}
}

func TestSuite1bReportShowsDefragPlanningCost(t *testing.T) {
	runs := []run1b{
		{Variant: "defrag", Load: 1.0, Seed: 1, DefragRPCP50: 100, DefragRPCP99: 900, DefragStatus: map[string]int{"OPTIMAL": 3, "FEASIBLE": 1}},
		{Variant: "defrag", Load: 1.0, Seed: 2, DefragRPCP50: 200, DefragRPCP99: 1100, DefragStatus: map[string]int{"OPTIMAL": 1, "UNKNOWN": 2, "rpc_error": 1}},
	}
	out := suite1bReport([]string{"defrag"}, []float64{1.0}, runs, 1, 2, "test")
	for _, want := range []string{"Defrag planning cost", "| defrag | 150.0 ± ", "OPTIMAL 4", "FEASIBLE 1", "UNKNOWN 2", "rpc_error 1"} {
		if !strings.Contains(out, want) {
			t.Fatalf("missing %q:\n%s", want, out)
		}
	}
}

func TestSuite1bReportDefragCostEmptyForNonDefrag(t *testing.T) {
	runs := []run1b{{Variant: "base", Load: 1.0, Seed: 1}}
	out := suite1bReport([]string{"base"}, []float64{1.0}, runs, 1, 1, "test")
	if !strings.Contains(out, "| base | 0.0 ± 0.0 | 0.0 ± 0.0 | - |") {
		t.Fatalf("want dash for no statuses:\n%s", out)
	}
}
