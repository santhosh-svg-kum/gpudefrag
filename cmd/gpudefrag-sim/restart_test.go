package main

import (
	"reflect"
	"testing"
)

func TestRestartVariants(t *testing.T) {
	vs := []string{"FGD", "FGD+defrag", "gpudefrag+defrag"}
	got, err := restartVariants(vs, "")
	if err != nil || !reflect.DeepEqual(got, vs) {
		t.Fatalf("empty sweep must be a no-op, got %v %v", got, err)
	}
	got, err = restartVariants(vs, "60,300")
	want := []string{"FGD", "FGD+defrag@r60", "FGD+defrag@r300", "gpudefrag+defrag@r60", "gpudefrag+defrag@r300"}
	if err != nil || !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v %v, want %v", got, err, want)
	}
	for _, bad := range []string{"x", "60,-1"} {
		if _, err := restartVariants(vs, bad); err == nil {
			t.Fatalf("sweep %q should fail", bad)
		}
	}
}

func TestSplitRestart(t *testing.T) {
	for _, c := range []struct {
		in   string
		def  float64
		base string
		r    float64
	}{
		{"FGD+defrag", 60, "FGD+defrag", 60},
		{"FGD+defrag@r900", 60, "FGD+defrag", 900},
		{"FGD+defrag", 120, "FGD+defrag", 120},
	} {
		b, r := splitRestart(c.in, c.def)
		if b != c.base || r != c.r {
			t.Errorf("%s: got %s %v", c.in, b, r)
		}
	}
}
