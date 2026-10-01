package mip

import (
	"gpupack/frag"
	"gpupack/model"
	"gpupack/sched"
	"gpupack/sim"
)

// pattern is one way to fill a node with a subset of the batch: the pods,
// their devices, and the node's exact FGD fragmentation afterwards.
type pattern struct {
	node int
	pods []int   // indices into the batch, ascending
	gpus [][]int // devices per pod
	frag float64
}

// genPatterns enumerates subsets of cand (batch indices) of size <= maxSize
// that fit on node, keeping for each subset the device assignment with the
// lowest fragmentation. Share pods try one device per distinct free-milli
// value (devices with equal free milli are interchangeable); whole-GPU pods
// take fully free devices. At most maxPatterns are returned; the empty
// pattern is always first.
func genPatterns(c *sim.Cluster, node int, cand []int, pods []model.Pod, typical []frag.TargetPod, maxSize, maxPatterns int) []pattern {
	best := map[uint64]int{} // subset mask -> index in out
	var out []pattern
	var rec func(start int, n *model.NodeRes, mask uint64, picks []pick)
	rec = func(start int, n *model.NodeRes, mask uint64, picks []pick) {
		f := frag.NodeScore(n, typical)
		if k, ok := best[mask]; ok {
			if f < out[k].frag {
				out[k] = toPattern(node, picks, f)
			}
		} else {
			if len(out) >= maxPatterns {
				return
			}
			best[mask] = len(out)
			out = append(out, toPattern(node, picks, f))
		}
		if len(picks) >= maxSize {
			return
		}
		for i := start; i < len(cand); i++ {
			p := pods[cand[i]].Res
			if !sim.Fits(n, p) {
				continue
			}
			for _, g := range deviceOptions(n, p) {
				next := n.Copy()
				next.CPULeft -= p.MilliCPU
				next.MemLeft -= p.MemMiB
				for _, d := range g {
					next.GpuLeft[d] -= p.GpuMilli
				}
				rec(i+1, next, mask|1<<uint(i), append(append([]pick(nil), picks...), pick{cand[i], g}))
			}
		}
	}
	rec(0, c.Nodes[node], 0, nil)
	return out
}

type pick struct {
	pod  int
	gpus []int
}

func toPattern(node int, picks []pick, f float64) pattern {
	p := pattern{node: node, frag: f}
	for _, k := range picks {
		p.pods = append(p.pods, k.pod)
		p.gpus = append(p.gpus, k.gpus)
	}
	return p
}

// deviceOptions lists distinct device choices for p on n.
func deviceOptions(n *model.NodeRes, p model.PodRes) [][]int {
	if p.GpuNum == 0 {
		return [][]int{nil}
	}
	if !p.IsShare() {
		return [][]int{sched.ExclusiveGPUs(n, p)}
	}
	var out [][]int
	seen := map[int64]bool{}
	for i, left := range n.GpuLeft {
		if left >= p.GpuMilli && !seen[left] {
			seen[left] = true
			out = append(out, []int{i})
		}
	}
	return out
}
