package model

import (
	"reflect"
	"testing"
)

func TestSubPacksSmallestFittingGPUFirst(t *testing.T) {
	n := NewNode("a", 64000, 1024, 4, "V100")
	n.GpuLeft = []int64{1000, 300, 600, 1000}

	got, err := n.Sub(PodRes{MilliCPU: 1000, GpuNum: 1, GpuMilli: 250})
	if err != nil {
		t.Fatal(err)
	}
	if want := []int64{1000, 50, 600, 1000}; !reflect.DeepEqual(got.GpuLeft, want) {
		t.Fatalf("share: got %v want %v", got.GpuLeft, want)
	}
	if got.CPULeft != 63000 {
		t.Fatalf("cpu left %d", got.CPULeft)
	}
	if n.GpuLeft[1] != 300 {
		t.Fatal("Sub must not mutate the receiver")
	}

	got, err = n.Sub(PodRes{MilliCPU: 1000, GpuNum: 2, GpuMilli: 1000})
	if err != nil {
		t.Fatal(err)
	}
	if want := []int64{0, 300, 600, 0}; !reflect.DeepEqual(got.GpuLeft, want) {
		t.Fatalf("multi: got %v want %v", got.GpuLeft, want)
	}
}

func TestSubErrors(t *testing.T) {
	n := NewNode("a", 1000, 1024, 2, "")
	if _, err := n.Sub(PodRes{MilliCPU: 2000}); err == nil {
		t.Fatal("expected cpu error")
	}
	if _, err := n.Sub(PodRes{MilliCPU: 100, GpuNum: 4, GpuMilli: 1000}); err == nil {
		t.Fatal("expected gpu count error")
	}
}

func TestAccessible(t *testing.T) {
	cases := []struct {
		node, pod string
		want      bool
	}{
		{"V100", "", true},
		{"", "", true},
		{"", "V100", false},
		{"V100M16", "V100M16|V100M32", true},
		{"T4", "V100M16|V100M32", false},
		{"T4", "|", true},
	}
	for _, c := range cases {
		n := NewNode("n", 1, 1, 1, c.node)
		if got := n.Accessible(PodRes{GpuType: c.pod}); got != c.want {
			t.Errorf("node %q pod %q: got %v", c.node, c.pod, got)
		}
	}
}

func TestAffinityTags(t *testing.T) {
	if (PodRes{}).Affinity() != "no-gpu" {
		t.Fatal("cpu")
	}
	if (PodRes{GpuNum: 1, GpuMilli: 500}).Affinity() != "share-gpu" {
		t.Fatal("share")
	}
	if (PodRes{GpuNum: 4, GpuMilli: 1000}).Affinity() != "4-gpu" {
		t.Fatal("multi")
	}
}

func TestCopyIsDeep(t *testing.T) {
	n := NewNode("a", 1, 1, 2, "")
	n.Affinity["share-gpu"] = 1
	c := n.Copy()
	c.GpuLeft[0] = 0
	c.Affinity["share-gpu"] = 5
	if n.GpuLeft[0] != 1000 || n.Affinity["share-gpu"] != 1 {
		t.Fatal("copy aliased receiver")
	}
}

func TestCounters(t *testing.T) {
	n := NewNode("a", 1, 1, 3, "")
	n.GpuLeft = []int64{1000, 400, 0}
	if n.TotalGpuLeft() != 1400 || n.FullyFree() != 1 || n.GpuNum() != 3 {
		t.Fatal("counters")
	}
	if got := n.SortedGpuIdx(true); !reflect.DeepEqual(got, []int{2, 1, 0}) {
		t.Fatalf("asc %v", got)
	}
}
