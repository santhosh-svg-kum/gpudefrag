package main

import (
	"testing"

	"github.com/santhosh-svg-kum/gpudefrag/calib"
)

func TestFmtSeedMaxDiffs(t *testing.T) {
	g := calib.GateResult{Policies: []calib.PolicyGate{
		{Policy: "FGD", SeedMaxDiff: []calib.SeedDiff{{Seed: 42, MaxAbsDiff: 0.5}, {Seed: 43, MaxAbsDiff: 1.25}}},
		{Policy: "BestFit"},
	}}
	want := "per-seed max |ours-ref| (pp):\n" +
		"  FGD            42=0.50 43=1.25\n" +
		"  BestFit        \n"
	if got := fmtSeedMaxDiffs(g); got != want {
		t.Fatalf("got %q want %q", got, want)
	}
}
