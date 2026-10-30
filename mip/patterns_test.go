package mip

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/santhosh-svg-kum/gpudefrag/frag"
	"github.com/santhosh-svg-kum/gpudefrag/model"
	"github.com/santhosh-svg-kum/gpudefrag/sim"
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

func benchFixture() (*sim.Cluster, []model.Pod, []frag.TargetPod, []int) {
	n := model.NewNode("n", 64000, 1<<20, 4, "")
	var pods []model.Pod
	var cand []int
	for i := 0; i < 12; i++ {
		r := model.PodRes{MilliCPU: 1000, MemMiB: 1, GpuNum: 1, GpuMilli: 250 * int64(1+i%4)}
		pods = append(pods, model.Pod{Name: "p", Res: r})
		cand = append(cand, i)
	}
	typ := []frag.TargetPod{{Res: model.PodRes{MilliCPU: 1000, GpuNum: 1, GpuMilli: 1000}, Pct: 1}}
	return sim.NewCluster([]*model.NodeRes{n}), pods, typ, cand
}

func BenchmarkGenPatterns(b *testing.B) {
	c, pods, typ, cand := benchFixture()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		genPatterns(c, 0, cand, pods, typ, 3, 2000)
	}
}

func manyNodeFixture(nodes int) (*sim.Cluster, []model.Pod, []frag.TargetPod, []int, []map[int]bool) {
	var ns []*model.NodeRes
	for i := 0; i < nodes; i++ {
		n := model.NewNode("n", 64000, 1<<20, 4, "")
		n.GpuLeft[i%4] = 500
		ns = append(ns, n)
	}
	_, pods, typ, cand := benchFixture()
	cands := make([]map[int]bool, len(pods))
	for i := range cands {
		cands[i] = map[int]bool{}
		for n := 0; n < nodes; n++ {
			cands[i][n] = true
		}
	}
	return sim.NewCluster(ns), pods, typ, cand, cands
}

func BenchmarkPlacerPatterns(b *testing.B) {
	c, pods, typ, _, cands := manyNodeFixture(16)
	nodes := make([]int, 16)
	for i := range nodes {
		nodes[i] = i
	}
	for _, w := range []int{1, 4} {
		b.Run(fmt.Sprintf("workers=%d", w), func(b *testing.B) {
			pl := &Placer{Typical: typ}
			for i := 0; i < b.N; i++ {
				pl.patternsN(w, c, pods, nodes, cands, nil)
			}
		})
	}
}

func TestPatternsParallelMatchesSerial(t *testing.T) {
	c, pods, typ, _, cands := manyNodeFixture(24)
	nodes := make([]int, 24)
	for i := range nodes {
		nodes[i] = i
	}
	hint := make([]*choice, len(pods))
	hint[1] = &choice{node: 3, gpus: []int{3}}
	pl := &Placer{Typical: typ}
	want := pl.patternsN(1, c, pods, nodes, cands, hint)
	if len(want) == 0 {
		t.Fatal("no patterns generated")
	}
	for _, w := range []int{2, 4, 64} {
		got := pl.patternsN(w, c, pods, nodes, cands, hint)
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("workers=%d: output differs from serial (%d vs %d patterns)", w, len(got), len(want))
		}
	}
}

func TestPatternCacheMatchesUncached(t *testing.T) {
	c, pods, typ, _, cands := manyNodeFixture(12) // 4 distinct node states, repeated
	// Drop some candidates on some nodes so the same state sees different pod subsets.
	for n := 0; n < 12; n += 5 {
		delete(cands[0], n)
		delete(cands[3], n)
	}
	cache := newPatternCache()
	for n := range c.Nodes {
		var here []int
		for i := range pods {
			if cands[i][n] {
				here = append(here, i)
			}
		}
		want := genPatterns(c, n, here, pods, typ, 3, 500)
		for rep := 0; rep < 2; rep++ {
			got := cache.gen(c, n, here, pods, typ, 3, 500)
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("node %d rep %d: cached output differs from uncached", n, rep)
			}
		}
	}
	if cache.hits.Load() == 0 {
		t.Fatal("expected cache hits across identical node states")
	}
}

func BenchmarkPatternsCache(b *testing.B) {
	c, pods, typ, _, cands := manyNodeFixture(16)
	nodes := make([]int, 16)
	for i := range nodes {
		nodes[i] = i
	}
	pl := &Placer{Typical: typ}
	for i := 0; i < b.N; i++ {
		pl.patternsN(1, c, pods, nodes, cands, nil)
	}
}
