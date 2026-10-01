package sched

import (
	"math/rand"
	"testing"

	"gpupack/model"
	"gpupack/sim"
)

func domainCluster() *sim.Cluster {
	var ns []*model.NodeRes
	for i, name := range []string{"a", "b", "c", "d"} {
		n := model.NewNode(name, 96000, 1<<20, 8, "")
		n.Domain = []string{"d1", "d1", "d2", "d2"}[i]
		ns = append(ns, n)
	}
	c := sim.NewCluster(ns)
	c.Bind(0, model.PodRes{MilliCPU: 1000, GpuNum: 4, GpuMilli: 1000}, []int{0, 1, 2, 3}) // d1 partly used
	return c
}

func gang(n, gpus int) []model.Pod {
	var out []model.Pod
	for i := 0; i < n; i++ {
		out = append(out, model.Pod{Name: "g", Res: model.PodRes{MilliCPU: 1000, GpuNum: gpus, GpuMilli: 1000}})
	}
	return out
}

func TestPlaceGangDomainLocalFallsBackToNextDomain(t *testing.T) {
	c := domainCluster()
	nodes, _, ok := PlaceGang(c, Binpack{}, gang(2, 8), true, rand.New(rand.NewSource(1)))
	if !ok || c.Nodes[nodes[0]].Domain != "d2" || c.Nodes[nodes[1]].Domain != "d2" {
		t.Fatalf("ok=%v nodes=%v", ok, nodes)
	}
}

func TestPlaceGangPrefersMostAllocatedDomain(t *testing.T) {
	c := domainCluster()
	nodes, _, ok := PlaceGang(c, Binpack{}, gang(2, 2), true, rand.New(rand.NewSource(1)))
	if !ok || c.Nodes[nodes[0]].Domain != "d1" || c.Nodes[nodes[1]].Domain != "d1" {
		t.Fatalf("binpack should fill the used domain first: %v", nodes)
	}
}

func TestPlaceGangAllOrNothing(t *testing.T) {
	c := domainCluster()
	before := c.UsedGpuMilli()
	if _, _, ok := PlaceGang(c, Binpack{}, gang(3, 8), true, nil); ok {
		t.Fatal("3 full nodes never fit in a 2-node domain")
	}
	if c.UsedGpuMilli() != before || c.Nodes[1].CPULeft != 96000 {
		t.Fatal("failed gang must leave the cluster unchanged")
	}
	if _, _, ok := PlaceGang(c, Binpack{}, gang(3, 8), false, nil); !ok {
		t.Fatal("non-local gang may span domains")
	}
}

func TestBinpackPrefersFullerNode(t *testing.T) {
	c := domainCluster()
	i, _, ok := Place(c, Binpack{}, model.Pod{Res: model.PodRes{MilliCPU: 1000, GpuNum: 2, GpuMilli: 1000}}, nil)
	if !ok || i != 0 {
		t.Fatalf("got %d", i)
	}
}
