// Package mip is gpudefrag's solver-backed scheduler. It warm-starts the solver
// with FGD's own placement, validates every answer against the cluster, and
// keeps the better of the two by the exact (Go-evaluated) objective, so it is
// never worse than FGD on a batch.
package mip

import (
	"context"
	"sort"
	"time"

	"google.golang.org/grpc"

	"github.com/santhosh-svg-kum/gpudefrag/frag"
	"github.com/santhosh-svg-kum/gpudefrag/model"
	pb "github.com/santhosh-svg-kum/gpudefrag/proto/gpudefrag/v1"
	"github.com/santhosh-svg-kum/gpudefrag/sched"
	"github.com/santhosh-svg-kum/gpudefrag/sim"
)

// Solver is the subset of the gRPC client the placer needs (fakeable).
type Solver interface {
	Place(ctx context.Context, in *pb.PlaceRequest, opts ...grpc.CallOption) (*pb.PlaceResponse, error)
}

type Stats struct {
	Batches   int
	Improved  int            // batches where the solver beat FGD
	Fallbacks map[string]int // rpc, status, invalid
	NoFit     int            // batches where no pod fit any node (nothing to solve)
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
	// IdleWeight is the penalty (milli-GPU of fragmentation) for putting GPU
	// pods on a node whose GPUs are all free; < 0 = lexicographic (never
	// open an idle node if a used one works). Keeps whole nodes free for
	// large jobs, which FGD's measure under-values when they are rare.
	IdleWeight float64
	// ObjectiveTypical, if set, is the typical-pod distribution the solver
	// optimizes (e.g. GPU-weighted); the FGD hint always uses Typical.
	ObjectiveTypical []frag.TargetPod
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

// Request is one job to place: a single pod, or a gang placed all-or-nothing
// (Local: within one topology domain).
type Request struct {
	Pods  []model.Pod
	Local bool
}

// Decide chooses placements for single pods without changing c (nil = not placed).
func (pl *Placer) Decide(c *sim.Cluster, pods []model.Pod) []*Choice {
	reqs := make([]Request, len(pods))
	for i, p := range pods {
		reqs[i] = Request{Pods: []model.Pod{p}}
	}
	out := make([]*Choice, len(pods))
	for i, d := range pl.DecideRequests(c, reqs) {
		if d != nil {
			out[i] = d[0]
		}
	}
	return out
}

// batch is a flattened request list: pod i belongs to gang[i] (0 = single).
type batch struct {
	pods  []model.Pod
	gang  []int
	local map[int]bool
}

// DecideRequests returns, per request, a choice for every pod or nil.
func (pl *Placer) DecideRequests(c *sim.Cluster, reqs []Request) [][]*Choice {
	pl.Stats.Batches++
	if pl.Stats.Fallbacks == nil {
		pl.Stats.Fallbacks = map[string]int{}
	}
	fgd := sched.NewFGD(pl.Typical)
	b := batch{local: map[int]bool{}}
	var first []int // index of each request's first pod
	for r, req := range reqs {
		first = append(first, len(b.pods))
		g := 0
		if len(req.Pods) > 1 {
			g = r + 1
			b.local[g] = req.Local
		}
		for _, p := range req.Pods {
			b.pods = append(b.pods, p)
			b.gang = append(b.gang, g)
		}
	}

	// FGD (gang-aware) hint on a scratch copy.
	scratch := sim.NewCluster(c.Nodes)
	hint := make([]*choice, len(b.pods))
	for r, req := range reqs {
		if len(req.Pods) == 1 {
			if n, g, ok := sched.Place(scratch, fgd, req.Pods[0], nil); ok {
				hint[first[r]] = &choice{n, g}
			}
			continue
		}
		if nodes, gpus, ok := sched.PlaceGang(scratch, fgd, req.Pods, req.Local, nil); ok {
			for k := range nodes {
				hint[first[r]+k] = &choice{nodes[k], gpus[k]}
			}
		}
	}

	best := hint
	if sol, reason := pl.solve(c, b, hint, fgd); reason == "empty" {
		pl.Stats.NoFit++
	} else if reason != "" {
		pl.Stats.Fallbacks[reason]++
	} else if better(c, b.pods, sol, hint, pl.objective()) {
		best = sol
		pl.Stats.Improved++
	}

	out := make([][]*Choice, len(reqs))
	for r, req := range reqs {
		if best[first[r]] == nil {
			continue
		}
		out[r] = make([]*Choice, len(req.Pods))
		for k := range req.Pods {
			ch := best[first[r]+k]
			out[r][k] = &Choice{Node: ch.node, GPUs: ch.gpus}
		}
	}
	return out
}

func (pl *Placer) solve(c *sim.Cluster, b batch, hint []*choice, fgd sched.FGD) ([]*choice, string) {
	pods := b.pods
	cands := make([]map[int]bool, len(pods))
	touched := map[int]bool{}
	req := &pb.PlaceRequest{
		TimeLimitS:    pl.TimeLimit.Seconds(),
		Deterministic: pl.Deterministic,
		Workers:       int32(pl.Workers),
	}
	for _, tp := range pl.objective() {
		req.Classes = append(req.Classes, &pb.TypicalClass{Cpu: tp.Res.MilliCPU, GpuMilli: tp.Res.GpuMilli, GpuNum: int32(tp.Res.GpuNum), Pct: tp.Pct})
	}
	for i, p := range pods {
		cands[i] = map[int]bool{}
		if b.gang[i] != 0 {
			// Gang pods may need whole domains: offer every feasible node.
			for _, n := range c.Feasible(p.Res) {
				cands[i][n] = true
			}
		} else {
			for _, n := range topK(c, fgd, p.Res, pl.K) {
				cands[i][n] = true
			}
		}
		if hint[i] != nil {
			cands[i][hint[i].node] = true
		}
		pp := &pb.Pod{Id: int32(i), Cpu: p.Res.MilliCPU, Mem: p.Res.MemMiB, GpuNum: int32(p.Res.GpuNum),
			GpuMilli: p.Res.GpuMilli, Weight: weight(p.Res), Gang: int32(b.gang[i]), DomainLocal: b.local[b.gang[i]]}
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
		pn := &pb.Node{Id: int32(id), CpuLeft: n.CPULeft, MemLeft: n.MemLeft, GpuLeft: append([]int64(nil), n.GpuLeft...), Domain: n.Domain}
		for _, tp := range pl.objective() {
			pn.ClassAccess = append(pn.ClassAccess, n.Accessible(tp.Res))
		}
		req.Nodes = append(req.Nodes, pn)
	}
	if len(req.Nodes) == 0 {
		return nil, "empty"
	}
	if !pl.Direct {
		req.Patterns = pl.patterns(c, pods, ids, cands, hint)
		req.IdleWeight = pl.IdleWeight
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
	if !ok || !gangsWhole(c, b, sol) {
		return nil, "invalid"
	}
	return sol, ""
}

// gangsWhole checks every gang is fully placed or not at all, and local
// gangs sit in one domain.
func gangsWhole(c *sim.Cluster, b batch, sol []*choice) bool {
	placed, size := map[int]int{}, map[int]int{}
	domains := map[int]map[string]bool{}
	for i, g := range b.gang {
		if g == 0 {
			continue
		}
		size[g]++
		if sol[i] != nil {
			placed[g]++
			if domains[g] == nil {
				domains[g] = map[string]bool{}
			}
			domains[g][c.Nodes[sol[i].node].Domain] = true
		}
	}
	for g, n := range size {
		if placed[g] != 0 && placed[g] != n {
			return false
		}
		if b.local[g] && len(domains[g]) > 1 {
			return false
		}
	}
	return true
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
		for _, p := range genPatterns(c, n, here, pods, pl.objective(), size, limit) {
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
		h.frag = frag.NodeScore(scratch.Nodes[0], pl.objective())
		nd := c.Nodes[n]
		h.opensIdle = nd.GpuNum() > 0 && nd.FullyFree() == nd.GpuNum() && usesGPU(h.gpus)
		out = append(out, toPB(h))
	}
	return out
}

func (pl *Placer) objective() []frag.TargetPod {
	if pl.ObjectiveTypical != nil {
		return pl.ObjectiveTypical
	}
	return pl.Typical
}

func toPB(p pattern) *pb.Pattern {
	pp := &pb.Pattern{Node: int32(p.node), Frag: p.frag, OpensIdle: p.opensIdle}
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
