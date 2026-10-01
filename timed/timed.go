// Package timed is the time-driven simulation (suite 1b): pods arrive and
// finish over virtual time, a single busy-server scheduler decides batches
// with modeled latency, binds are revalidated when applied, and an optional
// defragmenter migrates running pods to unblock fragmentation-blocked ones.
package timed

import (
	"math"
	"time"

	"gpupack/mip"
	"gpupack/model"
	"gpupack/sched"
	"gpupack/sim"
	"gpupack/trace"
)

// Decider chooses placements for a batch without mutating the cluster.
type Decider interface {
	Decide(c *sim.Cluster, pods []model.Pod) []*mip.Choice
}

// PolicyDecider runs a one-pod-at-a-time baseline on a scratch copy.
type PolicyDecider struct{ Policy sched.Policy }

func (d PolicyDecider) Decide(c *sim.Cluster, pods []model.Pod) []*mip.Choice {
	scratch := sim.NewCluster(c.Nodes)
	out := make([]*mip.Choice, len(pods))
	for i, p := range pods {
		if n, g, ok := sched.Place(scratch, d.Policy, p, nil); ok {
			out[i] = &mip.Choice{Node: n, GPUs: g}
		}
	}
	return out
}

type Config struct {
	Nodes   []*model.NodeRes
	Jobs    []trace.Job // sorted by Arrive
	Decider Decider
	Batch   int // max pods per session
	// Latency maps a decision's wall time and batch size to virtual seconds.
	// Default: max(wall, 1ms per pod).
	Latency func(wall time.Duration, pods int) float64
	// DrainLimit bounds the run at last arrival + DrainLimit (default 7 days).
	DrainLimit float64
	// MeasureFrom ends the warm-up: latency is recorded only for pods arriving
	// at or after it, and allocation is averaged over [MeasureFrom, last arrival].
	MeasureFrom float64
	// ConflictBackoff delays the next session after a bind conflict (default 1s).
	ConflictBackoff float64
	Defrag          *DefragConfig
}

type Result struct {
	GpuPendingLatency []float64 // seconds, GPU pods, in bind order
	AllocTimeAvg      float64   // mean allocated GPU fraction over [0, last arrival]
	Unplaced          int
	UnplacedGpuMilli  int64
	BindConflicts     int
	Sessions          int
	SolveWall         []time.Duration
	Defrag            DefragStats
}

type job struct {
	trace.Job
	id        int
	enqueued  float64
	node      int
	gpus      []int
	runStart  float64 // when current run makes progress from
	workDone  float64 // checkpointed work carried into the current run
	ckptPhase float64
	gen       int
	running   bool
}

type state struct {
	cfg        Config
	eng        sim.Engine
	c          *sim.Cluster
	jobs       []*job
	pending    []*job
	busy       bool
	dirty      bool
	backoff    bool // last apply had conflicts: delay the next session
	res        Result
	horizon    float64
	lastT      float64
	allocInt   float64
	lastDefrag float64
}

func Run(cfg Config) Result {
	if cfg.Batch <= 0 {
		cfg.Batch = 16
	}
	if cfg.Latency == nil {
		cfg.Latency = func(w time.Duration, n int) float64 { return math.Max(w.Seconds(), 0.001*float64(n)) }
	}
	if cfg.DrainLimit <= 0 {
		cfg.DrainLimit = 7 * 86400
	}
	if cfg.ConflictBackoff <= 0 {
		cfg.ConflictBackoff = 1
	}
	s := &state{cfg: cfg, c: sim.NewCluster(cfg.Nodes), lastDefrag: math.Inf(-1), lastT: cfg.MeasureFrom}
	if n := len(cfg.Jobs); n > 0 {
		s.horizon = cfg.Jobs[n-1].Arrive
	}
	for i, j := range cfg.Jobs {
		jb := &job{Job: j, id: i}
		if cfg.Defrag != nil {
			jb.ckptPhase = cfg.Defrag.phase(i)
		}
		s.jobs = append(s.jobs, jb)
		s.eng.Push(j.Arrive, sim.RankArrival, func() { s.arrive(jb) })
	}
	s.eng.Run(s.horizon + cfg.DrainLimit)
	s.account()
	if w := s.horizon - cfg.MeasureFrom; w > 0 {
		s.res.AllocTimeAvg = s.allocInt / (w * float64(s.c.TotalGpuMilli()))
	}
	for _, j := range s.pending {
		s.res.Unplaced++
		s.res.UnplacedGpuMilli += j.Pod.Res.TotalMilliGpu()
	}
	return s.res
}

// account integrates allocated milli-GPU over [0, horizon].
func (s *state) account() {
	now := math.Min(s.eng.Now(), s.horizon)
	if now > s.lastT {
		s.allocInt += float64(s.c.UsedGpuMilli()) * (now - s.lastT)
		s.lastT = now
	}
}

func (s *state) arrive(j *job) {
	j.enqueued = s.eng.Now()
	s.pending = append(s.pending, j)
	s.dirty = true
	s.trigger()
}

func (s *state) trigger() {
	if s.busy || !s.dirty || len(s.pending) == 0 {
		return
	}
	s.busy = true
	at := s.eng.Now()
	if s.backoff {
		at += s.cfg.ConflictBackoff
		s.backoff = false
	}
	s.eng.Push(at, sim.RankSession, s.session)
}

func (s *state) session() {
	s.dirty = false
	// Like kube's unschedulable queue: only pods that currently fit somewhere
	// are tried; the rest wait for a cluster change (or defrag).
	var batch []*job
	fits := map[model.PodRes]bool{} // many pending pods share a shape
	for _, j := range s.pending {
		if len(batch) == s.cfg.Batch {
			break
		}
		ok, seen := fits[j.Pod.Res]
		if !seen {
			ok = s.c.AnyFits(j.Pod.Res)
			fits[j.Pod.Res] = ok
		}
		if ok {
			batch = append(batch, j)
		}
	}
	if len(batch) == 0 {
		s.busy = false
		if s.cfg.Defrag != nil {
			s.maybeDefrag()
			s.trigger()
		}
		return
	}
	s.res.Sessions++
	pods := make([]model.Pod, len(batch))
	for i, j := range batch {
		pods[i] = j.Pod
	}
	t0 := time.Now()
	dec := s.cfg.Decider.Decide(s.c, pods)
	wall := time.Since(t0)
	s.res.SolveWall = append(s.res.SolveWall, wall)
	lat := s.cfg.Latency(wall, len(pods))
	chosen := append([]*job(nil), batch...)
	s.eng.Push(s.eng.Now()+lat, sim.RankBindApply, func() { s.apply(chosen, dec) })
}

func (s *state) apply(batch []*job, dec []*mip.Choice) {
	s.account()
	placed := map[*job]bool{}
	for i, ch := range dec {
		if ch == nil {
			continue
		}
		j := batch[i]
		if !canBind(s.c, ch.Node, j.Pod.Res, ch.GPUs) {
			s.res.BindConflicts++
			s.dirty, s.backoff = true, true
			continue
		}
		s.start(j, ch.Node, ch.GPUs, s.eng.Now())
		placed[j] = true
	}
	rest := s.pending[:0]
	for _, j := range s.pending {
		if !placed[j] {
			rest = append(rest, j)
		}
	}
	s.pending = rest
	s.busy = false
	if s.cfg.Defrag != nil {
		s.maybeDefrag()
	}
	s.trigger()
}

// start binds j and schedules its completion; progress begins at runFrom.
func (s *state) start(j *job, node int, gpus []int, runFrom float64) {
	s.c.Bind(node, j.Pod.Res, gpus)
	if j.gen == 0 && j.Pod.Res.GpuNum > 0 && j.Arrive >= s.cfg.MeasureFrom {
		s.res.GpuPendingLatency = append(s.res.GpuPendingLatency, s.eng.Now()-j.enqueued)
	}
	j.node, j.gpus, j.running, j.runStart = node, gpus, true, runFrom
	j.gen++
	gen := j.gen
	s.eng.Push(runFrom+(j.Duration-j.workDone), sim.RankCompletion, func() { s.complete(j, gen) })
}

func (s *state) complete(j *job, gen int) {
	if gen != j.gen || !j.running {
		return // superseded by a migration
	}
	s.account()
	s.c.Unbind(j.node, j.Pod.Res, j.gpus)
	j.running = false
	s.dirty = true
	s.trigger()
}

// canBind checks that p fits on node with exactly these devices.
func canBind(c *sim.Cluster, node int, p model.PodRes, gpus []int) bool {
	if node < 0 || node >= len(c.Nodes) || len(gpus) != p.GpuNum {
		return false
	}
	n := c.Nodes[node]
	if p.MilliCPU > n.CPULeft || p.MemMiB > n.MemLeft || (p.GpuNum > 0 && !n.Accessible(p)) {
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
