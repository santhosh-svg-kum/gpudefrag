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

func TestDefragCostPerPlan(t *testing.T) {
	cases := []struct {
		name          string
		rpcMs, gpus   float64
		plans         int
		wantMs, wantG float64
		wantOK        bool
	}{
		{"typical", 900, 6, 3, 300, 2, true},
		{"fractional", 100, 1.5, 4, 25, 0.375, true},
		{"no plans accepted", 500, 0, 0, 0, 0, false},
	}
	for _, c := range cases {
		ms, g, ok := defragCostPerPlan(c.rpcMs, c.plans, c.gpus)
		if ok != c.wantOK || ms != c.wantMs || g != c.wantG {
			t.Errorf("%s: got %v %v %v want %v %v %v", c.name, ms, g, ok, c.wantMs, c.wantG, c.wantOK)
		}
	}
}

func TestSuite1bReportShowsCostVsBenefit(t *testing.T) {
	runs := []run1b{
		{Variant: "defrag", Load: 1.0, Seed: 1, Plans: 2, DefragRPCTotalMs: 600, UnblockedGpu: 3},
		{Variant: "defrag", Load: 1.0, Seed: 2, Plans: 4, DefragRPCTotalMs: 600, UnblockedGpu: 9},
		{Variant: "base", Load: 1.0, Seed: 1},
	}
	out := suite1bReport([]string{"defrag", "base"}, []float64{1.0}, runs, 1, 2, "test")
	for _, want := range []string{"mean Defrag RPC ms per executed plan", "| defrag | 200.0 | 2.00 |", "| base | n/a | n/a |"} {
		if !strings.Contains(out, want) {
			t.Fatalf("missing %q:\n%s", want, out)
		}
	}
}
