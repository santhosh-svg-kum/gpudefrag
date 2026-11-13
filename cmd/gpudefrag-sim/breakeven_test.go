package main

import (
	"math"
	"testing"
)

func TestBreakEven(t *testing.T) {
	for _, c := range []struct {
		name string
		pts  []sweepPoint
		base float64
		want float64
		ok   bool
	}{
		{"never", []sweepPoint{{60, 0.9}, {300, 0.8}}, 0.7, 0, false},
		{"interpolated", []sweepPoint{{60, 0.8}, {300, 0.6}}, 0.7, 180, true},
		{"unsorted input", []sweepPoint{{900, 0.5}, {60, 0.8}, {300, 0.8}}, 0.7, 500, true},
		{"already below at first", []sweepPoint{{60, 0.6}, {300, 0.5}}, 0.7, 60, true},
		{"exact tie counts as stopped", []sweepPoint{{60, 0.8}, {300, 0.7}}, 0.7, 300, true},
		{"empty", nil, 0.7, 0, false},
	} {
		got, ok := breakEven(c.pts, c.base)
		if ok != c.ok || math.Abs(got-c.want) > 1e-9 {
			t.Errorf("%s: got %v %v, want %v %v", c.name, got, ok, c.want, c.ok)
		}
	}
}

func TestFormatBreakEven(t *testing.T) {
	if s := formatBreakEven(180, true); s != "180 s" {
		t.Errorf("got %q", s)
	}
	if s := formatBreakEven(0, false); s != "n/a" {
		t.Errorf("got %q", s)
	}
}
