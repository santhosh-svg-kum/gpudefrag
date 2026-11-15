package main

import (
	"strings"
	"testing"
	"time"
)

func TestBudgetNote(t *testing.T) {
	if got := budgetNote(500); got != ". Solver budget: 500 ms per batch" {
		t.Fatalf("got %q", got)
	}
}

func TestParseVariantBudget(t *testing.T) {
	v, err := parseVariantBudget("gpudefrag:b=16", 250)
	if err != nil || v.Limit != 250*time.Millisecond {
		t.Fatalf("default budget not applied: %v %v", v.Limit, err)
	}
	v, _ = parseVariantBudget("gpudefrag:b=16:t=2s", 250)
	if v.Limit != 2*time.Second {
		t.Fatalf("explicit t= must win, got %v", v.Limit)
	}
	v, _ = parseVariant("gpudefrag")
	if v.Limit != 500*time.Millisecond {
		t.Fatalf("default changed: %v", v.Limit)
	}
}

func TestSuite1aReportShowsBudget(t *testing.T) {
	rep := suite1aReport(nil, nil, 42, 44, 750)
	if !strings.Contains(rep, "Solver budget: 750 ms per batch") {
		t.Fatalf("header missing budget:\n%s", rep)
	}
}
