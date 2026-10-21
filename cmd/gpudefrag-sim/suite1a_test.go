package main

import (
	"strings"
	"testing"
	"time"

	"github.com/santhosh-svg-kum/gpudefrag/mip"
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

func TestStageTimesAndBreakdownSection(t *testing.T) {
	ms := func(n int) []time.Duration { return []time.Duration{time.Duration(n) * time.Millisecond} }
	st := stageTimes(mip.Stats{PatternWall: ms(7), RPCWall: ms(480), SolverWall: ms(470), SearchWall: ms(450)})
	if st["pattern gen"].P50 != 7 || st["rpc"].P99 != 480 || st["solver"].P50 != 470 || st["build"].P50 != 20 || st["search"].P99 != 450 {
		t.Fatalf("%+v", st)
	}
	vs := []variant{{Name: "FGD"}, {Name: "gpudefrag:b=16:t=500ms"}}
	res := map[string][]runSummary{
		"FGD":                    {{Seed: 42}},
		"gpudefrag:b=16:t=500ms": {{Seed: 42, Stages: st}, {Seed: 43, Stages: map[string]stageTime{"rpc": {600, 900}}}},
	}
	rep := breakdownSection(vs, res)
	if !strings.Contains(rep, "| pattern gen | rpc | solver | build | search |") || !strings.Contains(rep, "| gpudefrag:b=16:t=500ms | 7 / 7 | 600 / 900 | 470 / 470 | 20 / 20 | 450 / 450 |") {
		t.Fatalf("bad breakdown:\n%s", rep)
	}
	if strings.Contains(rep, "| FGD |") {
		t.Fatalf("FGD has no stages:\n%s", rep)
	}
	if breakdownSection(vs[:1], res) != "" {
		t.Fatal("section must be empty without stage data")
	}
}
