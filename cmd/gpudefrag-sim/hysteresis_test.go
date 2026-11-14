package main

import (
	"strings"
	"testing"
)

func TestSuite1bReportShowsHysteresisRefusals(t *testing.T) {
	runs := []run1b{
		{Variant: "defrag", Load: 1.0, Seed: 1, Plans: 2, Rejected: map[string]int{"hysteresis": 4, "grace-aborted": 1}},
		{Variant: "defrag", Load: 1.0, Seed: 2, Plans: 3, Rejected: map[string]int{"hysteresis": 6}},
	}
	out := suite1bReport([]string{"defrag"}, []float64{1.0}, runs, 1, 2, "test")
	if !strings.Contains(out, "plans refused by hysteresis") {
		t.Fatalf("missing refusal section:\n%s", out)
	}
	if !strings.Contains(out, "| defrag | 2.5 ± ") {
		t.Fatalf("want mean plans 2.5:\n%s", out)
	}
}

func TestSuite1bReportHysteresisZeroWhenNilMap(t *testing.T) {
	runs := []run1b{{Variant: "base", Load: 1.0, Seed: 1}}
	out := suite1bReport([]string{"base"}, []float64{1.0}, runs, 1, 1, "test")
	if !strings.Contains(out, "| base | 0.0 ± 0.0 | 0.0 ± 0.0 |\n\n") {
		t.Fatalf("nil Rejected must count as 0:\n%s", out)
	}
}
