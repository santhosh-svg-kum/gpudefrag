// Package sched contains placement policies and the one-pod-at-a-time
// scheduling cycle used by FGD's simulator (filter, score, select host,
// select GPUs, bind).
package sched

import (
	"fmt"
	"math/rand"

	"gpupack/frag"
	"gpupack/model"
	"gpupack/sim"
)

const (
	MinScore int64 = 0
	MaxScore int64 = 100
)

// Policy scores feasible nodes and picks GPU devices on the chosen node.
type Policy interface {
	Name() string
	// Score returns one score per entry of idx (indices into c.Nodes).
	Score(c *sim.Cluster, idx []int, p model.PodRes, rng *rand.Rand) []int64
	// SelectGPUs returns the device indices for p on n (nil for CPU pods).
	SelectGPUs(n *model.NodeRes, p model.PodRes, rng *rand.Rand) []int
}

// Names lists the baselines in FGD's experiment order.
var Names = []string{"Random", "DotProd", "GpuClustering", "GpuPacking", "BestFit", "FGD"}

// New builds a policy by name. typical is only used by FGD.
func New(name string, typical []frag.TargetPod) (Policy, error) {
	switch name {
	case "Random":
		return Random{}, nil
	case "DotProd":
		return DotProd{}, nil
	case "GpuClustering":
		return GpuClustering{}, nil
	case "GpuPacking":
		return GpuPacking{}, nil
	case "BestFit":
		return BestFit{}, nil
	case "FGD":
		return NewFGD(typical), nil
	case "Binpack":
		return Binpack{}, nil
	}
	return nil, fmt.Errorf("unknown policy %q", name)
}

// Place runs one scheduling cycle for p and binds it. ok is false when no
// node is feasible.
func Place(c *sim.Cluster, pol Policy, p model.Pod, rng *rand.Rand) (node int, gpus []int, ok bool) {
	idx := c.Feasible(p.Res)
	if len(idx) == 0 {
		return -1, nil, false
	}
	node = idx[0]
	if len(idx) > 1 { // kube-scheduler skips scoring when one node is feasible
		node = selectHost(c, idx, pol.Score(c, idx, p.Res, rng))
	}
	gpus = pol.SelectGPUs(c.Nodes[node], p.Res, rng)
	c.Bind(node, p.Res, gpus)
	return node, gpus, true
}

// selectHost picks the max score; ties go to the lexicographically smallest
// node name (FGD's patched kube-scheduler, for reproducibility).
func selectHost(c *sim.Cluster, idx []int, scores []int64) int {
	best := 0
	for k := 1; k < len(idx); k++ {
		if scores[k] > scores[best] ||
			(scores[k] == scores[best] && c.Nodes[idx[k]].Name < c.Nodes[idx[best]].Name) {
			best = k
		}
	}
	return idx[best]
}

// ExclusiveGPUs takes the first fully free devices by index (FGD's
// AllocateExclusiveGpuId). Returns nil for CPU pods.
func ExclusiveGPUs(n *model.NodeRes, p model.PodRes) []int {
	if p.GpuNum == 0 {
		return nil
	}
	var out []int
	need := p.TotalMilliGpu()
	for i, left := range n.GpuLeft {
		if need <= 0 {
			break
		}
		if left == model.Milli {
			out = append(out, i)
			need -= model.Milli
		}
	}
	return out
}

// BestFitGPUs puts a share pod on the fitting device with the least milli
// left (first index on ties); whole-GPU pods use ExclusiveGPUs.
func BestFitGPUs(n *model.NodeRes, p model.PodRes) []int {
	if !p.IsShare() {
		return ExclusiveGPUs(n, p)
	}
	best := -1
	for i, left := range n.GpuLeft {
		if left >= p.GpuMilli && (best == -1 || left < n.GpuLeft[best]) {
			best = i
		}
	}
	return []int{best}
}

// normalize is FGD's min-max NormalizeScore (integer arithmetic).
func normalize(scores []int64) {
	hi, lo := scores[0], scores[0]
	for _, s := range scores {
		hi, lo = max(hi, s), min(lo, s)
	}
	for i, s := range scores {
		if hi == lo {
			scores[i] = MinScore
		} else {
			scores[i] = (s-lo)*(MaxScore-MinScore)/(hi-lo) + MinScore
		}
	}
}
