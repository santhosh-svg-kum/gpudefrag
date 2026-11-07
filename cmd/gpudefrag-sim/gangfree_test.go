package main

import (
	"strings"
	"testing"
)

func TestSuite2GangNote(t *testing.T) {
	if got := suite2GangNote(false); got != "" {
		t.Fatalf("default must leave the header unchanged, got %q", got)
	}
	if got := suite2GangNote(true); !strings.Contains(got, "gang-free") {
		t.Fatalf("enabled note must mention gang-free, got %q", got)
	}
}
