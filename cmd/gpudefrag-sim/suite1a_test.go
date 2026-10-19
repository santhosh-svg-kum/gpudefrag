package main

import (
	"strings"
	"testing"
)

func TestSuite1aReportPaired(t *testing.T) {
	mk := func(vals ...float64) []runSummary {
		var rs []runSummary
		for i, v := range vals {
			c := make([]float64, 131)
			c[130] = v
			rs = append(rs, runSummary{Seed: int64(42 + i), Curve: c, FinalAlloc: v})
		}
		return rs
	}
	vs := []variant{{Name: "FGD"}, {Name: "gpudefrag:b=16:t=500ms"}, {Name: "BestFit"}}
	res := map[string][]runSummary{
		"FGD":                    mk(95.0, 95.0, 95.0),
		"gpudefrag:b=16:t=500ms": mk(95.2, 95.4, 94.9),
		"BestFit":                mk(93, 93, 93),
	}
	rep := suite1aReport(vs, res, 42, 44)
	if !strings.Contains(rep, "Paired by seed vs FGD at 130% load") {
		t.Fatalf("missing paired section:\n%s", rep)
	}
	if !strings.Contains(rep, "| gpudefrag:b=16:t=500ms | +0.17 ± ") || !strings.Contains(rep, "| 2/3 |") {
		t.Fatalf("bad paired row:\n%s", rep)
	}
	if !strings.Contains(rep, "| BestFit | -2.00 ± 0.00 | 0/3 |") {
		t.Fatalf("bad BestFit row:\n%s", rep)
	}
}

func TestSuite1aReportNoPairedWithoutFGD(t *testing.T) {
	vs := []variant{{Name: "BestFit"}}
	res := map[string][]runSummary{"BestFit": {{Seed: 42, Curve: make([]float64, 131)}}}
	if strings.Contains(suite1aReport(vs, res, 42, 42), "Paired") {
		t.Fatal("unexpected paired section")
	}
}
