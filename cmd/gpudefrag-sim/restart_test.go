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

func TestGridVariants(t *testing.T) {
	vs := []string{"FGD", "FGD+defrag"}
	got, err := gridVariants(vs, "60,300", "0,30")
	want := []string{"FGD", "FGD+defrag@r60@g0", "FGD+defrag@r60@g30", "FGD+defrag@r300@g0", "FGD+defrag@r300@g30"}
	if err != nil || !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v %v, want %v", got, err, want)
	}
	got, _ = gridVariants(vs, "", "30")
	if !reflect.DeepEqual(got, []string{"FGD", "FGD+defrag@g30"}) {
		t.Fatalf("grace only: %v", got)
	}
	got, _ = gridVariants(vs, "", "")
	if !reflect.DeepEqual(got, vs) {
		t.Fatalf("no sweep: %v", got)
	}
	if _, err := gridVariants(vs, "60", "-5"); err == nil {
		t.Fatal("negative grace should fail")
	}
	if _, err := gridVariants(vs, "x", ""); err == nil {
		t.Fatal("bad restart should fail")
	}
}

func TestSplitVariant(t *testing.T) {
	for _, c := range []struct {
		in   string
		base string
		r, g float64
	}{
		{"FGD+defrag", "FGD+defrag", 60, 5},
		{"FGD+defrag@r300@g30", "FGD+defrag", 300, 30},
		{"FGD+defrag@g30", "FGD+defrag", 60, 30},
		{"FGD+defrag@r900", "FGD+defrag", 900, 5},
	} {
		b, r, g := splitVariant(c.in, 60, 5)
		if b != c.base || r != c.r || g != c.g {
			t.Errorf("%s: got %s %v %v", c.in, b, r, g)
		}
	}
}
