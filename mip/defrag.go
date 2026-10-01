package mip

import (
	"context"
	"math"
	"sort"
	"time"

	"google.golang.org/grpc"

	"gpupack/model"
	pb "gpupack/proto/gpupack/v1"
	"gpupack/sim"
)

// DefragSolver is the subset of the gRPC client the defragmenter needs.
type DefragSolver interface {
	Defrag(ctx context.Context, in *pb.DefragRequest, opts ...grpc.CallOption) (*pb.DefragResponse, error)
}

// Running is a running pod the defragmenter may move. Cost is its migration
// cost in milli-GPU-seconds (lost work since checkpoint plus restart).
type Running struct {
	ID   int
	Node int
	GPUs []int
	Res  model.PodRes
	Cost float64
}

type Move struct {
	ID   int
	To   int
	GPUs []int
}

type Plan struct {
	Moves      []Move
	Target     int
	TargetGPUs []int
	Cost       float64
}

// Defragmenter finds the cheapest migrations that unblock one pending pod.
type Defragmenter struct {
	Solver        DefragSolver
	MaxMoves      int
	TimeLimit     time.Duration
	Targets       int // candidate target nodes (default 16)
	Receivers     int // extra receiving nodes by free GPU (default 64)
	Deterministic bool
	Workers       int
}

// Plan returns a validated plan, or nil and a reason: no_target, rpc, status, invalid.
func (d *Defragmenter) Plan(c *sim.Cluster, blocked model.PodRes, running []Running) (*Plan, string) {
	targets := d.targets(c, blocked, running)
	if len(targets) == 0 {
		return nil, "no_target"
	}
	inTarget := map[int]bool{}
	for _, t := range targets {
		inTarget[t] = true
	}
	var movable []Running
	for _, r := range running {
		if inTarget[r.Node] && r.Res.GpuNum > 0 {
			movable = append(movable, r)
		}
	}
	nodes := map[int]bool{}
	for _, t := range targets {
		nodes[t] = true
	}
	for _, n := range d.receivers(c) {
		nodes[n] = true
	}
	ids := make([]int, 0, len(nodes))
	for n := range nodes {
		ids = append(ids, n)
	}
	sort.Ints(ids)

	req := &pb.DefragRequest{MaxMoves: int32(d.MaxMoves), TimeLimitS: d.TimeLimit.Seconds(),
		Deterministic: d.Deterministic, Workers: int32(d.Workers)}
	for _, id := range ids {
		n := c.Nodes[id]
		req.Nodes = append(req.Nodes, &pb.Node{Id: int32(id), CpuLeft: n.CPULeft, MemLeft: n.MemLeft, GpuLeft: append([]int64(nil), n.GpuLeft...)})
	}
	b := &pb.Pod{Cpu: blocked.MilliCPU, Mem: blocked.MemMiB, GpuNum: int32(blocked.GpuNum), GpuMilli: blocked.GpuMilli}
	for _, t := range targets {
		b.Candidates = append(b.Candidates, int32(t))
	}
	req.Blocked = []*pb.Pod{b}
	byID := map[int]Running{}
	for _, r := range movable {
		byID[r.ID] = r
		m := &pb.Movable{Id: int32(r.ID), Node: int32(r.Node), Gpus: toI32(r.GPUs), Cpu: r.Res.MilliCPU, Mem: r.Res.MemMiB,
			GpuNum: int32(r.Res.GpuNum), GpuMilli: r.Res.GpuMilli, Cost: int64(math.Round(r.Cost))}
		for _, id := range ids {
			if id != r.Node && c.Nodes[id].Accessible(r.Res) {
				m.Dests = append(m.Dests, int32(id))
			}
		}
		req.Movable = append(req.Movable, m)
	}

	ctx, cancel := context.WithTimeout(context.Background(), d.TimeLimit*3/2+2*time.Second)
	defer cancel()
	resp, err := d.Solver.Defrag(ctx, req)
	if err != nil {
		return nil, "rpc"
	}
	if resp.Status != "OPTIMAL" && resp.Status != "FEASIBLE" {
		return nil, "status"
	}
	p, ok := validatePlan(c, blocked, byID, inTarget, resp, d.MaxMoves)
	if !ok {
		return nil, "invalid"
	}
	return p, ""
}

// targets are nodes that could host blocked if their movable GPU pods left,
// cheapest (least movable GPU to clear) first.
func (d *Defragmenter) targets(c *sim.Cluster, blocked model.PodRes, running []Running) []int {
	type cand struct {
		node int
		mass int64
	}
	byNode := map[int][]Running{}
	for _, r := range running {
		if r.Res.GpuNum > 0 {
			byNode[r.Node] = append(byNode[r.Node], r)
		}
	}
	var cs []cand
	for i, n := range c.Nodes {
		if n.GpuNum() < blocked.GpuNum || !n.Accessible(blocked) {
			continue
		}
		h := n.Copy()
		var mass int64
		for _, r := range byNode[i] {
			h.CPULeft += r.Res.MilliCPU
			h.MemLeft += r.Res.MemMiB
			for _, g := range r.GPUs {
				h.GpuLeft[g] += r.Res.GpuMilli
			}
			mass += r.Res.TotalMilliGpu()
		}
		if sim.Fits(h, blocked) {
			cs = append(cs, cand{i, mass})
		}
	}
	sort.Slice(cs, func(a, b int) bool {
		if cs[a].mass != cs[b].mass {
			return cs[a].mass < cs[b].mass
		}
		return c.Nodes[cs[a].node].Name < c.Nodes[cs[b].node].Name
	})
	k := d.Targets
	if k <= 0 {
		k = 16
	}
	out := make([]int, 0, k)
	for _, x := range cs[:min(k, len(cs))] {
		out = append(out, x.node)
	}
	return out
}

func (d *Defragmenter) receivers(c *sim.Cluster) []int {
	k := d.Receivers
	if k <= 0 {
		k = 64
	}
	idx := make([]int, 0, len(c.Nodes))
	for i, n := range c.Nodes {
		if n.TotalGpuLeft() > 0 {
			idx = append(idx, i)
		}
	}
	sort.Slice(idx, func(a, b int) bool {
		la, lb := c.Nodes[idx[a]].TotalGpuLeft(), c.Nodes[idx[b]].TotalGpuLeft()
		if la != lb {
			return la > lb
		}
		return c.Nodes[idx[a]].Name < c.Nodes[idx[b]].Name
	})
	return idx[:min(k, len(idx))]
}

// validatePlan replays the plan on a scratch cluster: every move from a real
// movable pod to devices with room, then the blocked pod on a target.
func validatePlan(c *sim.Cluster, blocked model.PodRes, byID map[int]Running, targets map[int]bool, resp *pb.DefragResponse, maxMoves int) (*Plan, bool) {
	if len(resp.Moves) > maxMoves || len(resp.Placements) != 1 {
		return nil, false
	}
	scratch := sim.NewCluster(c.Nodes)
	p := &Plan{}
	seen := map[int]bool{}
	for _, m := range resp.Moves {
		r, ok := byID[int(m.Pod)]
		if !ok || seen[r.ID] || int(m.ToNode) == r.Node || int(m.ToNode) < 0 || int(m.ToNode) >= len(c.Nodes) {
			return nil, false
		}
		seen[r.ID] = true
		scratch.Unbind(r.Node, r.Res, r.GPUs)
		p.Cost += r.Cost
	}
	for _, m := range resp.Moves {
		r := byID[int(m.Pod)]
		gpus := fromI32(m.Gpus)
		if !fitsOn(scratch, int(m.ToNode), r.Res, gpus) {
			return nil, false
		}
		scratch.Bind(int(m.ToNode), r.Res, gpus)
		p.Moves = append(p.Moves, Move{ID: r.ID, To: int(m.ToNode), GPUs: gpus})
	}
	a := resp.Placements[0]
	gpus := fromI32(a.Gpus)
	if !targets[int(a.Node)] || !fitsOn(scratch, int(a.Node), blocked, gpus) {
		return nil, false
	}
	p.Target, p.TargetGPUs = int(a.Node), gpus
	return p, true
}

func fitsOn(c *sim.Cluster, node int, p model.PodRes, gpus []int) bool {
	n := c.Nodes[node]
	if len(gpus) != p.GpuNum || p.MilliCPU > n.CPULeft || p.MemMiB > n.MemLeft || (p.GpuNum > 0 && !n.Accessible(p)) {
		return false
	}
	seen := map[int]bool{}
	for _, g := range gpus {
		if g < 0 || g >= n.GpuNum() || seen[g] || n.GpuLeft[g] < p.GpuMilli {
			return false
		}
		seen[g] = true
	}
	return true
}
