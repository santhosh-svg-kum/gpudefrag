package mip

import (
	"fmt"
	"sort"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/santhosh-svg-kum/gpudefrag/frag"
	"github.com/santhosh-svg-kum/gpudefrag/model"
	"github.com/santhosh-svg-kum/gpudefrag/sched"
	"github.com/santhosh-svg-kum/gpudefrag/sim"
)

// pattern is one way to fill a node with a subset of the batch: the pods,
// their devices, and the node's exact FGD fragmentation afterwards.
type pattern struct {
	node int
	pods []int   // indices into the batch, ascending
	gpus [][]int // devices per pod
	frag float64
	// opensIdle: the node had every GPU free and this pattern uses one.
	opensIdle bool
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
	if n := c.Nodes[node]; n.GpuNum() > 0 && n.FullyFree() == n.GpuNum() {
		for i := range out {
			out[i].opensIdle = usesGPU(out[i].gpus)
		}
	}
	return out
}

func usesGPU(gpus [][]int) bool {
	for _, g := range gpus {
		if len(g) > 0 {
			return true
		}
	}
	return false
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

// patternCache memoizes genPatterns within one batch. Two nodes with the same
// resource state and the same sequence of candidate pod shapes yield the same
// patterns up to node and pod indices, which are remapped on a hit.
type patternCache struct {
	m    sync.Map // string -> []pattern, with pods as positions in cand
	hits atomic.Int64
	size atomic.Int64 // entries stored
}

func newPatternCache() *patternCache { return &patternCache{} }

func (pc *patternCache) key(n *model.NodeRes, cand []int, pods []model.Pod, maxSize, maxPatterns int) string {
	var b strings.Builder
	fmt.Fprintf(&b, "%d,%d|%d,%d|%s|%v|", maxSize, maxPatterns, n.CPULeft, n.MemLeft, n.GpuType, n.GpuLeft)
	tags := make([]string, 0, len(n.Affinity))
	for k, v := range n.Affinity {
		tags = append(tags, fmt.Sprintf("%s=%d", k, v))
	}
	sort.Strings(tags)
	fmt.Fprintf(&b, "%v|", tags)
	for _, i := range cand {
		fmt.Fprintf(&b, "%v;", pods[i].Res)
	}
	return b.String()
}

// gen returns exactly what genPatterns would.
func (pc *patternCache) gen(c *sim.Cluster, node int, cand []int, pods []model.Pod, typical []frag.TargetPod, maxSize, maxPatterns int) []pattern {
	k := pc.key(c.Nodes[node], cand, pods, maxSize, maxPatterns)
	pos := make(map[int]int, len(cand))
	for p, i := range cand {
		pos[i] = p
	}
	if v, ok := pc.m.Load(k); ok {
		pc.hits.Add(1)
		return remapPatterns(v.([]pattern), node, cand, nil)
	}
	out := genPatterns(c, node, cand, pods, typical, maxSize, maxPatterns)
	if _, loaded := pc.m.LoadOrStore(k, remapPatterns(out, node, nil, pos)); !loaded {
		pc.size.Add(1)
	}
	return out
}

// remapPatterns copies ps onto node. With pos set it rewrites batch indices to
// positions in cand; with cand set it maps positions back to batch indices.
func remapPatterns(ps []pattern, node int, cand []int, pos map[int]int) []pattern {
	out := make([]pattern, len(ps))
	for i, p := range ps {
		q := p
		q.node = node
		if p.pods != nil {
			q.pods = make([]int, len(p.pods))
			for j, x := range p.pods {
				if pos != nil {
					q.pods[j] = pos[x]
				} else {
					q.pods[j] = cand[x]
				}
			}
		}
		out[i] = q
	}
	return out
}
