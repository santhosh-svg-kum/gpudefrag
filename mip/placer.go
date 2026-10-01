// Package mip is gpupack's solver-backed scheduler. It warm-starts the solver
// with FGD's own placement, validates every answer against the cluster, and
// keeps the better of the two by the exact (Go-evaluated) objective, so it is
// never worse than FGD on a batch.
package mip

import (
	"context"
	"sort"
	"time"

	"google.golang.org/grpc"

	"gpupack/frag"
	"gpupack/model"
	pb "gpupack/proto/gpupack/v1"
	"gpupack/sched"
	"gpupack/sim"
)

// Solver is the subset of the gRPC client the placer needs (fakeable).
type Solver interface {
	Place(ctx context.Context, in *pb.PlaceRequest, opts ...grpc.CallOption) (*pb.PlaceResponse, error)
}

type Stats struct {
	Batches   int
	Improved  int            // batches where the solver beat FGD
	Fallbacks map[string]int // rpc, status, invalid
	SolveWall []time.Duration
}

type Placer struct {
	Solver        Solver
	Typical       []frag.TargetPod
	K             int           // candidate nodes per pod
	TimeLimit     time.Duration // solver budget per batch
	Deterministic bool
	Workers       int
	// Pattern (column) formulation, the default: per-node subsets of up to
	// MaxPatternSize batch pods (default 3), at most MaxPatterns per node
	// (default 2000). Direct=true sends the direct CP-SAT model instead.
	MaxPatternSize int
	MaxPatterns    int
	Direct         bool
	Stats          Stats
}

type choice struct {
	node int
	gpus []int
}

// Choice is a decided placement for one pod of a batch.
type Choice struct {
	Node int
	GPUs []int
}

// PlaceBatch places pods (in order) onto c and reports which were placed.
func (pl *Placer) PlaceBatch(c *sim.Cluster, pods []model.Pod) []bool {
	dec := pl.Decide(c, pods)
	placed := make([]bool, len(pods))
	for i, ch := range dec {
		if ch != nil {
			c.Bind(ch.Node, pods[i].Res, ch.GPUs)
			placed[i] = true
		}
	}
	return placed
}

// Decide chooses placements for pods without changing c (nil = not placed).
func (pl *Placer) Decide(c *sim.Cluster, pods []model.Pod) []*Choice {
	pl.Stats.Batches++
	if pl.Stats.Fallbacks == nil {
		pl.Stats.Fallbacks = map[string]int{}
	}
	fgd := sched.NewFGD(pl.Typical)

	// FGD hint on a scratch copy.
	scratch := sim.NewCluster(c.Nodes)
	hint := make([]*choice, len(pods))
	for i, p := range pods {
		if n, g, ok := sched.Place(scratch, fgd, p, nil); ok {
			hint[i] = &choice{n, g}
		}
	}

	best := hint
	if sol, reason := pl.solve(c, pods, hint, fgd); reason != "" {
		pl.Stats.Fallbacks[reason]++
	} else if better(c, pods, sol, hint, pl.Typical) {
		best = sol
		pl.Stats.Improved++
	}

	out := make([]*Choice, len(pods))
	for i, ch := range best {
		if ch != nil {
			out[i] = &Choice{Node: ch.node, GPUs: ch.gpus}
		}
	}
	return out
}

func (pl *Placer) solve(c *sim.Cluster, pods []model.Pod, hint []*choice, fgd sched.FGD) ([]*choice, string) {
	cands := make([]map[int]bool, len(pods))
	touched := map[int]bool{}
	req := &pb.PlaceRequest{
		TimeLimitS:    pl.TimeLimit.Seconds(),
		Deterministic: pl.Deterministic,
		Workers:       int32(pl.Workers),
	}
	for _, tp := range pl.Typical {
		req.Classes = append(req.Classes, &pb.TypicalClass{Cpu: tp.Res.MilliCPU, GpuMilli: tp.Res.GpuMilli, GpuNum: int32(tp.Res.GpuNum), Pct: tp.Pct})
	}
	for i, p := range pods {
		cands[i] = map[int]bool{}
		for _, n := range topK(c, fgd, p.Res, pl.K) {
			cands[i][n] = true
		}
		if hint[i] != nil {
			cands[i][hint[i].node] = true
		}
		pp := &pb.Pod{Id: int32(i), Cpu: p.Res.MilliCPU, Mem: p.Res.MemMiB, GpuNum: int32(p.Res.GpuNum),
			GpuMilli: p.Res.GpuMilli, Weight: weight(p.Res)}
		for n := range cands[i] {
			pp.Candidates = append(pp.Candidates, int32(n))
			touched[n] = true
		}
		sort.Slice(pp.Candidates, func(a, b int) bool { return pp.Candidates[a] < pp.Candidates[b] })
		req.Pods = append(req.Pods, pp)
		if hint[i] != nil {
			req.Hint = append(req.Hint, &pb.Assignment{Pod: int32(i), Node: int32(hint[i].node), Gpus: toI32(hint[i].gpus)})
		}
	}
	ids := make([]int, 0, len(touched))
	for n := range touched {
		ids = append(ids, n)
	}
	sort.Ints(ids)
	for _, id := range ids {
		n := c.Nodes[id]
		pn := &pb.Node{Id: int32(id), CpuLeft: n.CPULeft, MemLeft: n.MemLeft, GpuLeft: append([]int64(nil), n.GpuLeft...)}
		for _, tp := range pl.Typical {
			pn.ClassAccess = append(pn.ClassAccess, n.Accessible(tp.Res))
		}
		req.Nodes = append(req.Nodes, pn)
	}
	if len(req.Nodes) == 0 {
		return nil, "empty"
	}
	if !pl.Direct {
		req.Patterns = pl.patterns(c, pods, ids, cands, hint)
	}

	ctx, cancel := context.WithTimeout(context.Background(), pl.TimeLimit*3/2+2*time.Second)
	defer cancel()
	t0 := time.Now()
	resp, err := pl.Solver.Place(ctx, req)
	pl.Stats.SolveWall = append(pl.Stats.SolveWall, time.Since(t0))
	if err != nil {
		return nil, "rpc"
	}
	if resp.Status != "OPTIMAL" && resp.Status != "FEASIBLE" {
		return nil, "status"
	}
	sol, ok := validate(c, pods, cands, resp.Assignments)
	if !ok {
		return nil, "invalid"
	}
	return sol, ""
}

func (pl *Placer) patterns(c *sim.Cluster, pods []model.Pod, nodes []int, cands []map[int]bool, hint []*choice) []*pb.Pattern {
	size, limit := pl.MaxPatternSize, pl.MaxPatterns
	if size <= 0 {
		size = 3
	}
	if limit <= 0 {
		limit = 2000
	}
	var out []*pb.Pattern
	for _, n := range nodes {
		var here []int
		for i := range pods {
			if cands[i][n] && len(here) < 64 {
				here = append(here, i)
			}
		}
		for _, p := range genPatterns(c, n, here, pods, pl.Typical, size, limit) {
			out = append(out, toPB(p))
		}
		// The hint's own pattern for this node keeps FGD's answer reachable.
		h := pattern{node: n}
		scratch := sim.NewCluster([]*model.NodeRes{c.Nodes[n]})
		for i, ch := range hint {
			if ch != nil && ch.node == n {
				scratch.Bind(0, pods[i].Res, ch.gpus)
				h.pods = append(h.pods, i)
				h.gpus = append(h.gpus, ch.gpus)
			}
		}
		h.frag = frag.NodeScore(scratch.Nodes[0], pl.Typical)
		out = append(out, toPB(h))
	}
	return out
}

func toPB(p pattern) *pb.Pattern {
	pp := &pb.Pattern{Node: int32(p.node), Frag: p.frag}
	for k, i := range p.pods {
		pp.Assignments = append(pp.Assignments, &pb.Assignment{Pod: int32(i), Node: int32(p.node), Gpus: toI32(p.gpus[k])})
	}
	return pp
}

// validate applies the solver's answer to a scratch cluster, checking every
// constraint the solver was supposed to respect.
func validate(c *sim.Cluster, pods []model.Pod, cands []map[int]bool, asg []*pb.Assignment) ([]*choice, bool) {
	scratch := sim.NewCluster(c.Nodes)
	out := make([]*choice, len(pods))
	for _, a := range asg {
		i, n := int(a.Pod), int(a.Node)
		if i < 0 || i >= len(pods) || out[i] != nil || !cands[i][n] {
			return nil, false
		}
		p := pods[i].Res
		node := scratch.Nodes[n]
		if len(a.Gpus) != p.GpuNum || p.MilliCPU > node.CPULeft || p.MemMiB > node.MemLeft {
			return nil, false
		}
		seen := map[int32]bool{}
		for _, g := range a.Gpus {
			if g < 0 || int(g) >= node.GpuNum() || seen[g] || node.GpuLeft[g] < p.GpuMilli {
				return nil, false
			}
			seen[g] = true
		}
		gpus := fromI32(a.Gpus)
		scratch.Bind(n, p, gpus)
		out[i] = &choice{n, gpus}
	}
	return out, true
}

// better reports whether a beats b: more placed weight, or equal weight and
// strictly less fragmentation on the nodes either one touches.
func better(c *sim.Cluster, pods []model.Pod, a, b []*choice, typical []frag.TargetPod) bool {
	wa, fa := evaluate(c, pods, a, b, typical)
	wb, fb := evaluate(c, pods, b, a, typical)
	return wa > wb || (wa == wb && fa < fb-1e-6)
}

func evaluate(c *sim.Cluster, pods []model.Pod, a, other []*choice, typical []frag.TargetPod) (int64, float64) {
	scratch := sim.NewCluster(c.Nodes)
	touched := map[int]bool{}
	var w int64
	for i, ch := range a {
		if ch != nil {
			scratch.Bind(ch.node, pods[i].Res, ch.gpus)
			touched[ch.node] = true
			w += weight(pods[i].Res)
		}
	}
	for _, ch := range other {
		if ch != nil {
			touched[ch.node] = true
		}
	}
	var f float64
	for n := range touched {
		f += frag.NodeScore(scratch.Nodes[n], typical)
	}
	return w, f
}

// weight values every placed pod; GPU pods by their milli-GPU.
func weight(p model.PodRes) int64 { return p.TotalMilliGpu() + 1 }

// topK returns the k feasible nodes with the best FGD score (name breaks ties).
func topK(c *sim.Cluster, fgd sched.FGD, p model.PodRes, k int) []int {
	idx := c.Feasible(p)
	if len(idx) <= k {
		return idx
	}
	scores := fgd.Score(c, idx, p, nil)
	order := make([]int, len(idx))
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(a, b int) bool {
		sa, sb := scores[order[a]], scores[order[b]]
		if sa != sb {
			return sa > sb
		}
		return c.Nodes[idx[order[a]]].Name < c.Nodes[idx[order[b]]].Name
	})
	out := make([]int, k)
	for i := 0; i < k; i++ {
		out[i] = idx[order[i]]
	}
	return out
}

func toI32(v []int) []int32 {
	out := make([]int32, len(v))
	for i, x := range v {
		out[i] = int32(x)
	}
	return out
}

func fromI32(v []int32) []int {
	if len(v) == 0 {
		return nil
	}
	out := make([]int, len(v))
	for i, x := range v {
		out[i] = int(x)
	}
	return out
}
