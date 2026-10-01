package sched

// Baseline score plugins ported from kubernetes-scheduler-simulator
// pkg/simulator/plugin/*_score.go (Apache-2.0). Integer arithmetic and
// truncation are kept exactly, because they create the score ties that
// decide placements.

import (
	"math"
	"math/rand"

	"gpupack/frag"
	"gpupack/model"
	"gpupack/sim"
)

func scoreEach(c *sim.Cluster, idx []int, f func(n *model.NodeRes) int64) []int64 {
	out := make([]int64, len(idx))
	for k, i := range idx {
		out[k] = f(c.Nodes[i])
	}
	return out
}

// Random gives MaxScore to one uniformly chosen feasible node.
type Random struct{}

func (Random) Name() string { return "Random" }
func (Random) Score(_ *sim.Cluster, idx []int, _ model.PodRes, rng *rand.Rand) []int64 {
	out := make([]int64, len(idx))
	out[rng.Intn(len(idx))] = MaxScore
	return out
}

// SelectGPUs is FGD's random-fit: reservoir sampling over fitting devices.
func (Random) SelectGPUs(n *model.NodeRes, p model.PodRes, rng *rand.Rand) []int {
	if !p.IsShare() {
		return ExclusiveGPUs(n, p)
	}
	pick, cnt := -1, 0
	for i, left := range n.GpuLeft {
		if left >= p.GpuMilli {
			cnt++
			if rng.Intn(cnt) == 0 {
				pick = i
			}
		}
	}
	return []int{pick}
}

// BestFit prefers the node with the least CPU and GPU left after placement.
type BestFit struct{}

func (BestFit) Name() string { return "BestFit" }
func (BestFit) Score(c *sim.Cluster, idx []int, p model.PodRes, _ *rand.Rand) []int64 {
	out := scoreEach(c, idx, func(n *model.NodeRes) int64 {
		free := [2]float64{float64(n.CPULeft), float64(n.TotalGpuLeft())}
		req := [2]float64{float64(p.MilliCPU), float64(p.TotalMilliGpu())}
		spec := [2]float64{model.MaxSpecCPU, model.MaxSpecGPU}
		var s float64
		for i := range free {
			if free[i] < req[i] {
				return MinScore
			}
			s += (free[i] - req[i]) / spec[i] * 0.5
		}
		return int64((1.0 - s) * float64(MaxScore))
	})
	normalize(out)
	return out
}
func (BestFit) SelectGPUs(n *model.NodeRes, p model.PodRes, _ *rand.Rand) []int {
	return BestFitGPUs(n, p)
}

// DotProd prefers nodes whose free-resource vector is most aligned with the
// pod's request ("merge" dimensions, "max" normalisation).
type DotProd struct{}

func (DotProd) Name() string { return "DotProd" }
func (DotProd) Score(c *sim.Cluster, idx []int, p model.PodRes, _ *rand.Rand) []int64 {
	return scoreEach(c, idx, func(n *model.NodeRes) int64 {
		node := [2]float64{float64(n.CPULeft) / model.MaxSpecCPU, float64(n.TotalGpuLeft()) / model.MaxSpecGPU}
		pod := [2]float64{float64(p.MilliCPU) / model.MaxSpecCPU, float64(p.TotalMilliGpu()) / model.MaxSpecGPU}
		dot := (node[0]*pod[0] + node[1]*pod[1]) / 2
		return int64(float64(MaxScore) * (1 - dot))
	})
}
func (DotProd) SelectGPUs(n *model.NodeRes, p model.PodRes, _ *rand.Rand) []int {
	return BestFitGPUs(n, p)
}

// GpuClustering co-locates pods with the same GPU-affinity tag.
type GpuClustering struct{}

func (GpuClustering) Name() string { return "GpuClustering" }
func (GpuClustering) Score(c *sim.Cluster, idx []int, p model.PodRes, _ *rand.Rand) []int64 {
	tag := p.Affinity()
	return scoreEach(c, idx, func(n *model.NodeRes) int64 {
		if p.GpuNum == 0 {
			return MinScore
		}
		used := MaxScore / 4 * (model.MaxSpecGPU - n.TotalGpuLeft()) / model.MaxSpecGPU
		switch {
		case n.Affinity[tag] > 0 && len(n.Affinity) == 1:
			return used + MaxScore*3/4
		case n.Affinity[tag] > 0:
			return used + MaxScore*2/4
		case len(n.Affinity) == 0:
			return used + MaxScore/4
		default:
			return used
		}
	})
}
func (GpuClustering) SelectGPUs(n *model.NodeRes, p model.PodRes, _ *rand.Rand) []int {
	return BestFitGPUs(n, p)
}

// GpuPacking prefers partially used GPUs, then partially used nodes, then
// idle nodes.
type GpuPacking struct{}

func (GpuPacking) Name() string { return "GpuPacking" }
func (GpuPacking) Score(c *sim.Cluster, idx []int, p model.PodRes, _ *rand.Rand) []int64 {
	return scoreEach(c, idx, func(n *model.NodeRes) int64 {
		if p.GpuMilli <= 0 {
			return MinScore
		}
		free := int64(n.FullyFree())
		if int(free) == n.GpuNum() { // all GPUs idle
			return max(MaxScore/3-free, free)
		}
		req, freeUsed := p.GpuNum, int64(0)
		var use []int
		for _, i := range n.SortedGpuIdx(true) {
			if req == 0 {
				break
			}
			if p.GpuMilli <= n.GpuLeft[i] {
				req--
				use = append(use, i)
				if n.GpuLeft[i] == model.Milli {
					freeUsed++
				}
			}
		}
		if req != 0 {
			return MinScore
		}
		if freeUsed > 0 { // must break into idle GPUs
			return max(MaxScore/2-freeUsed, MaxScore/3)
		}
		var ratio int64 // shared GPUs only
		for _, i := range use {
			ratio += n.GpuLeft[i] * 100 / model.Milli
		}
		return max(MaxScore-ratio/10, MaxScore/2)
	})
}
func (GpuPacking) SelectGPUs(n *model.NodeRes, p model.PodRes, _ *rand.Rand) []int {
	return BestFitGPUs(n, p)
}

// FGD (fragmentation gradient descent) picks the node, and for share pods the
// GPU, whose placement increases fragmentation least.
type FGD struct{ typical []frag.TargetPod }

func NewFGD(typical []frag.TargetPod) FGD { return FGD{typical: typical} }

func (FGD) Name() string { return "FGD" }

func sigmoid(x float64) float64 { return 1.0 / (1.0 + math.Exp(-x)) }

// score returns FGD's score for placing p on n and the GPU it would use
// (share pods only; -1 otherwise).
func (f FGD) score(n *model.NodeRes, p model.PodRes) (int64, int) {
	before := frag.NodeScore(n, f.typical)
	if p.IsShare() {
		best, gpu := int64(0), -1
		after := n.Copy()
		after.CPULeft -= p.MilliCPU
		for i, left := range n.GpuLeft {
			if left < p.GpuMilli {
				continue
			}
			after.GpuLeft[i] -= p.GpuMilli
			s := int64(sigmoid((before-frag.NodeScore(after, f.typical))/1000) * float64(MaxScore))
			after.GpuLeft[i] += p.GpuMilli
			if gpu == -1 || s > best {
				best, gpu = s, i
			}
		}
		return best, gpu
	}
	after, _ := n.Sub(p) // feasible nodes always fit
	return int64(sigmoid((before-frag.NodeScore(after, f.typical))/1000) * float64(MaxScore)), -1
}

func (f FGD) Score(c *sim.Cluster, idx []int, p model.PodRes, _ *rand.Rand) []int64 {
	return scoreEach(c, idx, func(n *model.NodeRes) int64 { s, _ := f.score(n, p); return s })
}

func (f FGD) SelectGPUs(n *model.NodeRes, p model.PodRes, _ *rand.Rand) []int {
	if !p.IsShare() {
		return ExclusiveGPUs(n, p)
	}
	_, g := f.score(n, p)
	return []int{g}
}
