package mip

import (
	"testing"

	"gpupack/frag"
	"gpupack/model"
	"gpupack/sim"
)

func patternFixture() (*sim.Cluster, []model.Pod, []frag.TargetPod) {
	n := model.NewNode("n", 64000, 1<<20, 2, "")
	n.GpuLeft[1] = 500
	pods := []model.Pod{
		{Name: "share", Res: model.PodRes{MilliCPU: 1000, MemMiB: 1, GpuNum: 1, GpuMilli: 500}},
		{Name: "whole", Res: model.PodRes{MilliCPU: 1000, MemMiB: 1, GpuNum: 1, GpuMilli: 1000}},
		{Name: "cpu", Res: model.PodRes{MilliCPU: 1000, MemMiB: 1}},
	}
	typ := []frag.TargetPod{{Res: model.PodRes{MilliCPU: 1000, GpuNum: 1, GpuMilli: 1000}, Pct: 1}}
	return sim.NewCluster([]*model.NodeRes{n}), pods, typ
}

func TestGenPatternsEnumeratesFeasibleSubsetsWithBestGPU(t *testing.T) {
	c, pods, typ := patternFixture()
	pats := genPatterns(c, 0, []int{0, 1, 2}, pods, typ, 3, 1000)
	if len(pats) != 8 {
		t.Fatalf("want 8 subsets (all feasible), got %d", len(pats))
	}
	for _, p := range pats {
		// frag must equal the exact FGD measure of the resulting node
		s := sim.NewCluster(c.Nodes)
		for k, i := range p.pods {
			s.Bind(0, pods[i].Res, p.gpus[k])
		}
		if got := frag.NodeScore(s.Nodes[0], typ); got != p.frag {
			t.Fatalf("pattern %v frag %v, exact %v", p.pods, p.frag, got)
		}
		if len(p.pods) == 1 && p.pods[0] == 0 && p.gpus[0][0] != 1 {
			t.Fatalf("share pod alone should fill the half-used GPU (frag 0), got GPU %v", p.gpus[0])
		}
	}
}

func TestGenPatternsRespectsMaxSize(t *testing.T) {
	c, pods, typ := patternFixture()
	for _, p := range genPatterns(c, 0, []int{0, 1, 2}, pods, typ, 1, 1000) {
		if len(p.pods) > 1 {
			t.Fatalf("pattern larger than max size: %v", p.pods)
		}
	}
	if n := len(genPatterns(c, 0, []int{0, 1, 2}, pods, typ, 3, 2)); n > 2 {
		t.Fatalf("cap not respected: %d", n)
	}
}

func TestGenPatternsSkipsInfeasible(t *testing.T) {
	c, pods, typ := patternFixture()
	pods = append(pods, model.Pod{Name: "big", Res: model.PodRes{MilliCPU: 1, MemMiB: 1, GpuNum: 2, GpuMilli: 1000}})
	for _, p := range genPatterns(c, 0, []int{3}, pods, typ, 3, 1000) {
		if len(p.pods) != 0 {
			t.Fatal("2-GPU pod cannot fit a node with one free GPU")
		}
	}
}

func TestPatternsMarkOpeningIdleNodes(t *testing.T) {
	idle := model.NewNode("idle", 64000, 1<<20, 2, "")
	used := model.NewNode("used", 64000, 1<<20, 2, "")
	used.GpuLeft[0] = 500
	c := sim.NewCluster([]*model.NodeRes{idle, used})
	pods := []model.Pod{
		{Name: "g", Res: model.PodRes{MilliCPU: 1, MemMiB: 1, GpuNum: 1, GpuMilli: 500}},
		{Name: "cpu", Res: model.PodRes{MilliCPU: 1, MemMiB: 1}},
	}
	typ := []frag.TargetPod{{Res: model.PodRes{MilliCPU: 1, GpuNum: 1, GpuMilli: 1000}, Pct: 1}}
	for _, p := range genPatterns(c, 0, []int{0, 1}, pods, typ, 3, 100) {
		hasGPU := false
		for _, i := range p.pods {
			hasGPU = hasGPU || pods[i].Res.GpuNum > 0
		}
		if p.opensIdle != hasGPU {
			t.Fatalf("idle node pattern %v: opensIdle=%v", p.pods, p.opensIdle)
		}
	}
	for _, p := range genPatterns(c, 1, []int{0, 1}, pods, typ, 3, 100) {
		if p.opensIdle {
			t.Fatal("a partly used node is never 'opened'")
		}
	}
}
