package mip

import (
	"reflect"
	"testing"

	"github.com/santhosh-svg-kum/gpudefrag/model"
	"github.com/santhosh-svg-kum/gpudefrag/sim"
)

func TestGroupBlocked(t *testing.T) {
	// Two 2-GPU nodes (a, b) of type A, one 1-GPU node (c) of type B.
	mk := func() *sim.Cluster {
		return sim.NewCluster([]*model.NodeRes{
			model.NewNode("a", 64000, 1<<20, 2, "A"),
			model.NewNode("b", 64000, 1<<20, 2, "A"),
			model.NewNode("c", 64000, 1<<20, 1, "B"),
		})
	}
	pa := model.PodRes{MilliCPU: 1000, GpuNum: 1, GpuMilli: 500, GpuType: "A"}
	pb := model.PodRes{MilliCPU: 1000, GpuNum: 1, GpuMilli: 500, GpuType: "B"}
	big := model.PodRes{MilliCPU: 1000, GpuNum: 1, GpuMilli: 500, GpuType: "A", MemMiB: 1 << 30}

	cases := []struct {
		name string
		pods []model.PodRes
		gang []int
		want [][]int
	}{
		{"empty", nil, nil, nil},
		{"shared node", []model.PodRes{pa, pa}, []int{0, 0}, [][]int{{0, 1}}},
		{"disjoint types", []model.PodRes{pa, pb, pa}, []int{0, 0, 0}, [][]int{{0, 2}, {1}}},
		{"gang links disjoint", []model.PodRes{pa, pb}, []int{7, 7}, [][]int{{0, 1}}},
		{"no candidate stays alone", []model.PodRes{big, pb}, []int{0, 0}, [][]int{{0}, {1}}},
	}
	for _, tc := range cases {
		got := GroupBlocked(mk(), tc.pods, tc.gang)
		if !reflect.DeepEqual(got, tc.want) {
			t.Errorf("%s: got %v want %v", tc.name, got, tc.want)
		}
	}
}

func TestBlockedCandidatesIgnoresFragmentedFit(t *testing.T) {
	// Node has 600 free milli-GPU total but split 300/300: a 600 milli pod does
	// not fit (blocked) yet is a defrag candidate; a 700 milli pod is not.
	n := model.NewNode("a", 64000, 1<<20, 2, "A")
	n.GpuLeft = []int64{300, 300}
	c := sim.NewCluster([]*model.NodeRes{n})
	p6 := model.PodRes{GpuNum: 1, GpuMilli: 600, GpuType: "A"}
	p7 := model.PodRes{GpuNum: 1, GpuMilli: 700, GpuType: "A"}
	got := blockedCandidates(c, []model.PodRes{p6, p7})
	if !got[0][0] || len(got[1]) != 0 {
		t.Errorf("got %v", got)
	}
}
