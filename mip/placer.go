// Package mip is gpudefrag's solver-backed scheduler. It warm-starts the solver
// with FGD's own placement, validates every answer against the cluster, and
// keeps the better of the two by the exact (Go-evaluated) objective, so it is
// never worse than FGD on a batch.
package mip

import (
	"context"
	"runtime"
	"sort"
	"sync"
	"sync/atomic"
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
	Improved  int             // batches where the solver beat FGD
	Fallbacks map[string]int  // rpc, status, invalid
	NoFit     int             // batches where no pod fit any node (nothing to solve)
	SolveWall []time.Duration // RPC round trip (kept for the existing reports)
	// Per-solved-batch breakdown of SolveWall's surroundings: Go-side pattern
	// generation (pattern formulation only), the RPC round trip, and the wall
	// time the solver itself reports (model build + search, excludes transport).
	PatternWall []time.Duration
	RPCWall     []time.Duration
	SolverWall  []time.Duration
	SearchWall  []time.Duration // CP-SAT search only; SolverWall - SearchWall is model build
}

type Placer struct {
	Solver  Solver
	Typical []frag.TargetPod
	K       int // candidate nodes per pod
	// PatternWorkers bounds the goroutines generating per-node patterns;
	// 0 means min(GOMAXPROCS, 8), 1 is serial. Output does not depend on it.
	PatternWorkers int
	TimeLimit      time.Duration // solver budget per batch
	Deterministic  bool
	Workers        int
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
	// WarmStartBest warm-starts from the better (by the batch objective) of
	// the FGD and Binpack placements instead of FGD's alone. Off by default.
	WarmStartBest bool
	// GangFree breaks batch-objective ties (equal weight and fragmentation)
	// in favor of the placement that leaves free whole nodes concentrated in
	// fewer topology domains (see freeNodeScore). Off by default.
	GangFree bool
	// Decompose solves groups of pods with disjoint candidate nodes (and no
	// shared gang) as separate concurrent solver requests and merges the
	// answers. Off by default.
	Decompose bool
	// CrossBatchCache keeps the pattern cache across batches, so a node whose
	// state and candidate pod shapes repeat in a later batch reuses its
	// patterns. Off by default. It assumes the objective does not change
	// between batches. The cache is dropped between batches once it holds more
	// than CacheMaxEntries (default 20000) entries.
	CrossBatchCache bool
	CacheMaxEntries int
	cache           *patternCache
	Stats           Stats
	mu              sync.Mutex // guards Stats appends from concurrent group solves
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

	// FGD (gang-aware) hint, or the better of FGD and Binpack if WarmStartBest.
	hint := pl.hintFor(c, reqs, first, fgd)
	if pl.WarmStartBest {
		if bp := pl.hintFor(c, reqs, first, sched.Binpack{}); better(c, b.pods, bp, hint, pl.objective(), pl.GangFree) {
			hint = bp
		}
	}

	best := hint
	if sol, reason := pl.solve(c, b, hint, fgd); reason == "empty" {
		pl.Stats.NoFit++
	} else if reason != "" {
		pl.Stats.Fallbacks[reason]++
	} else if better(c, b.pods, sol, hint, pl.objective(), pl.GangFree) {
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

// hintFor places the requests in order on a scratch copy with pol.
func (pl *Placer) hintFor(c *sim.Cluster, reqs []Request, first []int, pol sched.Policy) []*choice {
	scratch := sim.NewCluster(c.Nodes)
	n := 0
	for _, req := range reqs {
		n += len(req.Pods)
	}
	hint := make([]*choice, n)
	for r, req := range reqs {
		if len(req.Pods) == 1 {
			if n, g, ok := sched.Place(scratch, pol, req.Pods[0], nil); ok {
				hint[first[r]] = &choice{n, g}
			}
			continue
		}
		if nodes, gpus, ok := sched.PlaceGang(scratch, pol, req.Pods, req.Local, nil); ok {
			for k := range nodes {
				hint[first[r]+k] = &choice{nodes[k], gpus[k]}
			}
		}
	}
	return hint
}

func (pl *Placer) solve(c *sim.Cluster, b batch, hint []*choice, fgd sched.FGD) ([]*choice, string) {
	if pl.Decompose {
		groups := independentGroups(pl.candidates(c, b, hint, fgd), b.gang)
		if len(groups) > 1 {
			return pl.solveGroups(c, b, hint, fgd, groups)
		}
	}
	return pl.solveOne(c, b, hint, fgd)
}

// candidates returns each pod's candidate nodes: the FGD top-K (every
// feasible node for gang pods) plus the hint's node.
func (pl *Placer) candidates(c *sim.Cluster, b batch, hint []*choice, fgd sched.FGD) []map[int]bool {
	cands := make([]map[int]bool, len(b.pods))
	for i, p := range b.pods {
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
	}
	return cands
}

// solveGroups solves each group as its own request, concurrently, and merges
// the answers back into batch order. Any failing group fails the whole batch
// (so the caller keeps the hint); groups with no candidate nodes stay unplaced.
func (pl *Placer) solveGroups(c *sim.Cluster, b batch, hint []*choice, fgd sched.FGD, groups [][]int) ([]*choice, string) {
	sols := make([][]*choice, len(groups))
	reasons := make([]string, len(groups))
	var wg sync.WaitGroup
	for g, idx := range groups {
		wg.Add(1)
		go func() {
			defer wg.Done()
			sub := batch{local: b.local}
			subHint := make([]*choice, len(idx))
			for k, i := range idx {
				sub.pods = append(sub.pods, b.pods[i])
				sub.gang = append(sub.gang, b.gang[i])
				subHint[k] = hint[i]
			}
			sols[g], reasons[g] = pl.solveOne(c, sub, subHint, fgd)
		}()
	}
	wg.Wait()
	out := make([]*choice, len(b.pods))
	solved := false
	for g, idx := range groups {
		switch reasons[g] {
		case "":
			solved = true
			for k, i := range idx {
				out[i] = sols[g][k]
			}
		case "empty":
		default:
			return nil, reasons[g]
		}
	}
	if !solved {
		return nil, "empty"
	}
	return out, ""
}

func (pl *Placer) solveOne(c *sim.Cluster, b batch, hint []*choice, fgd sched.FGD) ([]*choice, string) {
	pods := b.pods
	cands := pl.candidates(c, b, hint, fgd)
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
		tp := time.Now()
		req.Patterns = pl.patterns(c, pods, ids, cands, hint)
		req.IdleWeight = pl.IdleWeight
		pl.mu.Lock()
		pl.Stats.PatternWall = append(pl.Stats.PatternWall, time.Since(tp))
		pl.mu.Unlock()
	}

	ctx, cancel := context.WithTimeout(context.Background(), pl.TimeLimit*3/2+2*time.Second)
	defer cancel()
	t0 := time.Now()
	resp, err := pl.Solver.Place(ctx, req)
	rpc := time.Since(t0)
	pl.mu.Lock()
	pl.Stats.SolveWall = append(pl.Stats.SolveWall, rpc)
	pl.Stats.RPCWall = append(pl.Stats.RPCWall, rpc)
	if err == nil {
		pl.Stats.SolverWall = append(pl.Stats.SolverWall, time.Duration(resp.WallS*float64(time.Second)))
		pl.Stats.SearchWall = append(pl.Stats.SearchWall, time.Duration(resp.SearchWallS*float64(time.Second)))
	}
	pl.mu.Unlock()
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

// patterns generates every node's patterns on a bounded worker pool. Each
// node is independent and only reads the cluster, and results are joined in
// node order, so the output is identical to a serial run.
func (pl *Placer) patterns(c *sim.Cluster, pods []model.Pod, nodes []int, cands []map[int]bool, hint []*choice) []*pb.Pattern {
	w := pl.PatternWorkers
	if w <= 0 {
		w = min(runtime.GOMAXPROCS(0), 8)
	}
	return pl.patternsN(w, c, pods, nodes, cands, hint)
}

func (pl *Placer) patternsN(workers int, c *sim.Cluster, pods []model.Pod, nodes []int, cands []map[int]bool, hint []*choice) []*pb.Pattern {
	size, limit := pl.MaxPatternSize, pl.MaxPatterns
	if size <= 0 {
		size = 3
	}
	if limit <= 0 {
		limit = 2000
	}
	per := make([][]*pb.Pattern, len(nodes))
	cache := pl.batchCache()
	one := func(k int) {
		n := nodes[k]
		var here []int
		for i := range pods {
			if cands[i][n] && len(here) < 64 {
				here = append(here, i)
			}
		}
		var out []*pb.Pattern
		for _, p := range cache.gen(c, n, here, pods, pl.objective(), size, limit) {
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
		per[k] = append(out, toPB(h))
	}
	if workers > len(nodes) {
		workers = len(nodes)
	}
	if workers <= 1 {
		for k := range nodes {
			one(k)
		}
	} else {
		var next atomic.Int64
		var wg sync.WaitGroup
		for ; workers > 0; workers-- {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for k := int(next.Add(1)) - 1; k < len(nodes); k = int(next.Add(1)) - 1 {
					one(k)
				}
			}()
		}
		wg.Wait()
	}
	var out []*pb.Pattern
	for _, ps := range per {
		out = append(out, ps...)
	}
	return out
}

// batchCache returns the pattern cache for one batch: fresh, or the shared
// cross-batch one (reset once it outgrows CacheMaxEntries).
func (pl *Placer) batchCache() *patternCache {
	if !pl.CrossBatchCache {
		return newPatternCache()
	}
	limit := int64(pl.CacheMaxEntries)
	if limit <= 0 {
		limit = 20000
	}
	pl.mu.Lock()
	defer pl.mu.Unlock()
	if pl.cache == nil || pl.cache.size.Load() > limit {
		pl.cache = newPatternCache()
	}
	return pl.cache
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
// strictly less fragmentation on the nodes either one touches. With gangFree,
// a tie on both is broken by freeNodeScore.
func better(c *sim.Cluster, pods []model.Pod, a, b []*choice, typical []frag.TargetPod, gangFree bool) bool {
	wa, fa := evaluate(c, pods, a, b, typical)
	wb, fb := evaluate(c, pods, b, a, typical)
	if wa != wb || fa < fb-1e-6 || fb < fa-1e-6 {
		return wa > wb || (wa == wb && fa < fb-1e-6)
	}
	return gangFree && freeNodeScore(c, pods, a) > freeNodeScore(c, pods, b)
}

// freeNodeScore is the sum over topology domains of (free whole GPU nodes)^2
// after applying the placement; the square rewards keeping free nodes
// together, where a gang that must sit in one domain can use them.
func freeNodeScore(c *sim.Cluster, pods []model.Pod, a []*choice) int64 {
	scratch := sim.NewCluster(c.Nodes)
	for i, ch := range a {
		if ch != nil {
			scratch.Bind(ch.node, pods[i].Res, ch.gpus)
		}
	}
	free := map[string]int64{}
	for _, n := range scratch.Nodes {
		if n.GpuNum() == 0 {
			continue
		}
		whole := true
		for _, g := range n.GpuLeft {
			if g != model.Milli {
				whole = false
				break
			}
		}
		if whole {
			free[n.Domain]++
		}
	}
	var s int64
	for _, k := range free {
		s += k * k
	}
	return s
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
