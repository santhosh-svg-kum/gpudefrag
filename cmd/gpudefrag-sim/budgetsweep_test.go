package main

import (
	"reflect"
	"strings"
	"testing"
	"time"
)

func TestBudgetVariants(t *testing.T) {
	vs := []string{"FGD", "gpudefrag:b=16", "gpudefrag:b=1:t=100ms"}
	got, err := budgetVariants(vs, "")
	if err != nil || !reflect.DeepEqual(got, vs) {
		t.Fatalf("empty sweep must be a no-op, got %v %v", got, err)
	}
	got, err = budgetVariants(vs, "250,1000")
	want := []string{"FGD", "gpudefrag:b=16:t=250ms", "gpudefrag:b=16:t=1000ms", "gpudefrag:b=1:t=100ms"}
	if err != nil || !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v %v, want %v", got, err, want)
	}
	for _, bad := range []string{"x", "250,0", "-5"} {
		if _, err := budgetVariants(vs, bad); err == nil {
			t.Fatalf("sweep %q should fail", bad)
		}
	}
}

func TestBudgetSweepSection(t *testing.T) {
	vs := []variant{{Name: "FGD"}, {Name: "gpudefrag:b=16:t=250ms", Limit: 250 * time.Millisecond}, {Name: "gpudefrag:b=16:t=1000ms", Limit: time.Second}}
	curve := func(a float64) []float64 {
		c := make([]float64, 131)
		c[130] = a
		return c
	}
	res := map[string][]runSummary{
		"FGD": {{Seed: 1, Curve: curve(80)}},
		"gpudefrag:b=16:t=250ms": {
			{Seed: 1, Curve: curve(80), Solved: 8, BudgetHit: 6, NoSolution: 1, StatusOther: 1},
			{Seed: 2, Curve: curve(82), Solved: 10, BudgetHit: 4},
		},
		"gpudefrag:b=16:t=1000ms": {{Seed: 1, Curve: curve(83), Solved: 10, BudgetHit: 1}},
	}
	got := budgetSweepSection(vs, res)
	for _, s := range []string{
		"## Solver budget sweep",
		"| gpudefrag:b=16:t=250ms | 250 | 81.00 ± ",
		"| 10 / 18 | 55.6% | 10.0% |",
		"| gpudefrag:b=16:t=1000ms | 1000 | 83.00 ± 0.00 | 1 / 10 | 10.0% | 0.0% |",
	} {
		if !strings.Contains(got, s) {
			t.Errorf("missing %q in:\n%s", s, got)
		}
	}
	if strings.Contains(got, "| FGD |") {
		t.Errorf("baselines must not appear:\n%s", got)
	}
	if budgetSweepSection(vs[:1], res) != "" {
		t.Error("no solver variants: want empty section")
	}
}
