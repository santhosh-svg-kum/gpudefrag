package timed

import (
	"testing"

	"gpupack/mip"
	"gpupack/model"
	"gpupack/sched"
	"gpupack/sim"
	"gpupack/trace"
)

// spread puts pod i of each batch on node (global count % nodes).
type spread struct{ n *int }

func (s spread) DecideRequests(c *sim.Cluster, reqs []mip.Request) [][]*mip.Choice {
	out := make([][]*mip.Choice, len(reqs))
	for i, r := range reqs {
		p := r.Pods[0]
		node := *s.n % len(c.Nodes)
		if sim.Fits(c.Nodes[node], p.Res) {
			out[i] = []*mip.Choice{{Node: node, GPUs: sched.BestFitGPUs(c.Nodes[node], p.Res)}}
			*s.n++
		}
	}
	return out
}

type scripted struct {
	plan  *mip.Plan
	calls int
	seen  []mip.Running
}

func (s *scripted) Plan(c *sim.Cluster, blocked model.PodRes, running []mip.Running) (*mip.Plan, string) {
	s.calls++
	s.seen = running
	return s.plan, ""
}

func textbook(cost float64) (Config, *scripted) {
	one := func(name string, at float64) trace.Job {
		return trace.Job{Pod: model.Pod{Name: name, Res: model.PodRes{MilliCPU: 1000, MemMiB: 1, GpuNum: 1, GpuMilli: 1000}}, Arrive: at, Duration: 7200}
	}
	big := trace.Job{Pod: model.Pod{Name: "big", Res: model.PodRes{MilliCPU: 1000, MemMiB: 1, GpuNum: 2, GpuMilli: 1000}}, Arrive: 1000, Duration: 100}
	pl := &scripted{plan: &mip.Plan{Moves: []mip.Move{{ID: 1, To: 0, GPUs: []int{1}}}, Target: 1, TargetGPUs: []int{0, 1}, Cost: cost}}
	cfg := Config{
		Nodes:   []*model.NodeRes{model.NewNode("a", 64000, 1<<20, 2, ""), model.NewNode("b", 64000, 1<<20, 2, "")},
		Jobs:    []trace.Job{one("q0", 0), one("q1", 0), big},
		Decider: spread{new(int)}, Batch: 1, Latency: fixed(0),
		Defrag: &DefragConfig{Planner: pl, Interval: 60, CheckpointInterval: 1800, Restart: 30, BenefitHorizon: 3600, Margin: 0.2, Seed: 1},
	}
	return cfg, pl
}

func TestDefragUnblocksTextbookCase(t *testing.T) {
	cfg, pl := textbook(1000)
	r := Run(cfg)
	if pl.calls != 1 || r.Defrag.Plans != 1 || r.Defrag.Migrations != 1 {
		t.Fatalf("calls=%d stats=%+v", pl.calls, r.Defrag)
	}
	if r.Unplaced != 0 || r.GpuPendingLatency[2] != 0 {
		t.Fatalf("big job should bind at the defrag instant: %+v", r.GpuPendingLatency)
	}
	if len(pl.seen) != 2 {
		t.Fatalf("planner must see both running GPU pods: %+v", pl.seen)
	}
	// q1 ran 1000s; with checkpoint phase < 1800 it loses (1000 - ckpt) + 30s restart.
	if r.Defrag.LostGpuSec < 30 || r.Defrag.LostGpuSec > 1030 {
		t.Fatalf("lost %v", r.Defrag.LostGpuSec)
	}
}

func TestDefragHysteresisRejectsExpensivePlans(t *testing.T) {
	// benefit = 2000 milli * 3600 s; cost above benefit/1.2 must be rejected.
	cfg, _ := textbook(2000 * 3600)
	r := Run(cfg)
	if r.Defrag.Plans != 0 || r.Defrag.Rejected["hysteresis"] == 0 {
		t.Fatalf("%+v", r.Defrag)
	}
}

func TestCheckpointMath(t *testing.T) {
	cases := []struct{ done, phase, interval, want float64 }{
		{100, 300, 1800, 0},     // before first checkpoint
		{500, 300, 1800, 300},   // after first
		{2200, 300, 1800, 2100}, // after second
	}
	for _, c := range cases {
		if got := lastCheckpoint(c.done, c.phase, c.interval); got != c.want {
			t.Errorf("%+v: got %v", c, got)
		}
	}
}
