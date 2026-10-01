package sim

import (
	"reflect"
	"testing"

	"gpupack/model"
)

func cluster() *Cluster {
	a := model.NewNode("a", 8000, 1000, 2, "V100")
	b := model.NewNode("b", 64000, 1000, 8, "T4")
	c := model.NewNode("c", 64000, 1000, 0, "")
	return NewCluster([]*model.NodeRes{a, b, c})
}

func TestFeasible(t *testing.T) {
	c := cluster()
	cases := []struct {
		name string
		pod  model.PodRes
		want []int
	}{
		{"cpu pod fits everywhere", model.PodRes{MilliCPU: 1000, MemMiB: 10}, []int{0, 1, 2}},
		{"cpu too large", model.PodRes{MilliCPU: 9000, MemMiB: 10}, []int{1, 2}},
		{"mem too large", model.PodRes{MilliCPU: 1, MemMiB: 2000}, nil},
		{"share pod skips gpu-less node", model.PodRes{MilliCPU: 1, GpuNum: 1, GpuMilli: 500}, []int{0, 1}},
		{"type constraint", model.PodRes{MilliCPU: 1, GpuNum: 1, GpuMilli: 1000, GpuType: "T4"}, []int{1}},
		{"type nobody has", model.PodRes{MilliCPU: 1, GpuNum: 1, GpuMilli: 1000, GpuType: "H100"}, nil},
		{"8-gpu pod only on 8-gpu node", model.PodRes{MilliCPU: 1, GpuNum: 8, GpuMilli: 1000}, []int{1}},
		{"16-gpu pod fits nowhere", model.PodRes{MilliCPU: 1, GpuNum: 16, GpuMilli: 1000}, nil},
	}
	for _, tc := range cases {
		if got := c.Feasible(tc.pod); !reflect.DeepEqual(got, tc.want) {
			t.Errorf("%s: got %v want %v", tc.name, got, tc.want)
		}
	}
}

func TestBindAndAccounting(t *testing.T) {
	c := cluster()
	if c.TotalGpus() != 10 || c.TotalGpuMilli() != 10000 {
		t.Fatal("totals")
	}
	share := model.PodRes{MilliCPU: 1000, MemMiB: 100, GpuNum: 1, GpuMilli: 300}
	c.Bind(0, share, []int{1})
	n := c.Nodes[0]
	if n.CPULeft != 7000 || n.MemLeft != 900 || n.GpuLeft[1] != 700 || n.Affinity["share-gpu"] != 1 {
		t.Fatalf("%+v", n)
	}
	c.Bind(1, model.PodRes{MilliCPU: 1, GpuNum: 2, GpuMilli: 1000}, []int{0, 1})
	c.Bind(2, model.PodRes{MilliCPU: 1}, nil)
	if c.UsedGpuMilli() != 2300 {
		t.Fatalf("used %d", c.UsedGpuMilli())
	}
	if c.Nodes[2].Affinity["no-gpu"] != 0 {
		t.Fatal("cpu pods carry no affinity")
	}
}

func TestBindPanicsOnOvercommit(t *testing.T) {
	c := cluster()
	defer func() {
		if recover() == nil {
			t.Fatal("expected invariant panic")
		}
	}()
	c.Bind(0, model.PodRes{MilliCPU: 1, GpuNum: 1, GpuMilli: 1000}, []int{0})
	c.Bind(0, model.PodRes{MilliCPU: 1, GpuNum: 1, GpuMilli: 500}, []int{0})
}

func TestNewClusterCopiesNodes(t *testing.T) {
	src := []*model.NodeRes{model.NewNode("a", 1000, 1000, 1, "")}
	c := NewCluster(src)
	c.Bind(0, model.PodRes{MilliCPU: 1, GpuNum: 1, GpuMilli: 1000}, []int{0})
	if src[0].GpuLeft[0] != 1000 {
		t.Fatal("cluster must not mutate the caller's nodes")
	}
}
