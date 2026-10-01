package sched

import (
	"math/rand"
	"sort"

	"gpupack/model"
	"gpupack/sim"
)

// Binpack is Volcano's binpack plugin with equal CPU and GPU weights: prefer
// the node that is fullest after placement.
type Binpack struct{}

func (Binpack) Name() string { return "Binpack" }
func (Binpack) Score(c *sim.Cluster, idx []int, p model.PodRes, _ *rand.Rand) []int64 {
	return scoreEach(c, idx, func(n *model.NodeRes) int64 {
		var s, w float64
		if n.CPUCap > 0 {
			s += float64(n.CPUCap-n.CPULeft+p.MilliCPU) / float64(n.CPUCap)
			w++
		}
		if g := n.GpuNum(); g > 0 {
			s += float64(int64(g)*model.Milli-n.TotalGpuLeft()+p.TotalMilliGpu()) / float64(int64(g)*model.Milli)
			w++
		}
		return int64(s / w * float64(MaxScore))
	})
}
func (Binpack) SelectGPUs(n *model.NodeRes, p model.PodRes, _ *rand.Rand) []int { return BestFitGPUs(n, p) }

// PlaceGang binds every pod of a gang or none. With local, all pods must land
// in one topology domain; domains are tried most-allocated first (Volcano
// binpack order), ties by name. On failure the cluster is left unchanged.
func PlaceGang(c *sim.Cluster, pol Policy, pods []model.Pod, local bool, rng *rand.Rand) ([]int, [][]int, bool) {
	if !local {
		all := make([]int, len(c.Nodes))
		for i := range all {
			all[i] = i
		}
		return placeWithin(c, pol, pods, all, rng)
	}
	for _, idx := range DomainsByUse(c) {
		if nodes, gpus, ok := placeWithin(c, pol, pods, idx, rng); ok {
			return nodes, gpus, true
		}
	}
	return nil, nil, false
}

// DomainsByUse groups node indices by Domain, fullest domain (by GPU) first.
func DomainsByUse(c *sim.Cluster) [][]int {
	by := map[string][]int{}
	var names []string
	for i, n := range c.Nodes {
		if _, ok := by[n.Domain]; !ok {
			names = append(names, n.Domain)
		}
		by[n.Domain] = append(by[n.Domain], i)
	}
	used := func(d string) float64 {
		var u, t int64
		for _, i := range by[d] {
			n := c.Nodes[i]
			t += int64(n.GpuNum()) * model.Milli
			u += int64(n.GpuNum())*model.Milli - n.TotalGpuLeft()
		}
		if t == 0 {
			return 0
		}
		return float64(u) / float64(t)
	}
	sort.Slice(names, func(a, b int) bool {
		ua, ub := used(names[a]), used(names[b])
		if ua != ub {
			return ua > ub
		}
		return names[a] < names[b]
	})
	out := make([][]int, len(names))
	for i, d := range names {
		out[i] = by[d]
	}
	return out
}

func placeWithin(c *sim.Cluster, pol Policy, pods []model.Pod, idx []int, rng *rand.Rand) ([]int, [][]int, bool) {
	nodes := make([]int, 0, len(pods))
	gpus := make([][]int, 0, len(pods))
	rollback := func() {
		for k := range nodes {
			c.Unbind(nodes[k], pods[k].Res, gpus[k])
		}
	}
	for _, p := range pods {
		var feas []int
		for _, i := range idx {
			if sim.Fits(c.Nodes[i], p.Res) {
				feas = append(feas, i)
			}
		}
		if len(feas) == 0 {
			rollback()
			return nil, nil, false
		}
		node := feas[0]
		if len(feas) > 1 {
			node = selectHost(c, feas, pol.Score(c, feas, p.Res, rng))
		}
		g := pol.SelectGPUs(c.Nodes[node], p.Res, rng)
		c.Bind(node, p.Res, g)
		nodes = append(nodes, node)
		gpus = append(gpus, g)
	}
	return nodes, gpus, true
}
