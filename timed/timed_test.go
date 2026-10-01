package timed

import (
	"math"
	"math/rand"
	"reflect"
	"testing"
	"time"

	"gpupack/mip"
	"gpupack/model"
	"gpupack/sched"
	"gpupack/sim"
	"gpupack/trace"
)

func gpuJob(name string, arrive, dur float64, milli int64) trace.Job {
	return trace.Job{Pod: model.Pod{Name: name, Res: model.PodRes{MilliCPU: 1000, MemMiB: 1, GpuNum: 1, GpuMilli: milli}}, Arrive: arrive, Duration: dur}
}

func fixed(sec float64) func(time.Duration, int) float64 {
	return func(time.Duration, int) float64 { return sec }
}

func nodes(gpus ...int) []*model.NodeRes {
	var out []*model.NodeRes
	for i, g := range gpus {
		out = append(out, model.NewNode(string(rune('a'+i)), 64000, 1<<20, g, ""))
	}
	return out
}

func bestFit() Decider { return PolicyDecider{Policy: sched.BestFit{}} }

// Design doc §7.5: job1 at 0 decided until 3; job2 arrives at 2, waits, is
// decided from 3 to 6 -> pending 4s.
func TestBusyServer(t *testing.T) {
	r := Run(Config{Nodes: nodes(4), Jobs: []trace.Job{gpuJob("j1", 0, 100, 1000), gpuJob("j2", 2, 100, 1000)},
		Decider: bestFit(), Batch: 1, Latency: fixed(3)})
	if !reflect.DeepEqual(r.GpuPendingLatency, []float64{3, 4}) {
		t.Fatalf("latencies %v", r.GpuPendingLatency)
	}
	if r.Sessions != 2 {
		t.Fatalf("sessions %d", r.Sessions)
	}
}

func TestCompletionFreesCapacity(t *testing.T) {
	r := Run(Config{Nodes: nodes(1), Jobs: []trace.Job{gpuJob("j1", 0, 10, 1000), gpuJob("j2", 1, 10, 1000)},
		Decider: bestFit(), Batch: 4, Latency: fixed(0)})
	if !reflect.DeepEqual(r.GpuPendingLatency, []float64{0, 9}) || r.Unplaced != 0 {
		t.Fatalf("%+v", r)
	}
}

func TestUnplaceableCountedAtEnd(t *testing.T) {
	j := gpuJob("big", 0, 10, 1000)
	j.Pod.Res.GpuNum = 8
	r := Run(Config{Nodes: nodes(2), Jobs: []trace.Job{j}, Decider: bestFit(), Batch: 4, Latency: fixed(0)})
	if r.Unplaced != 1 || r.UnplacedGpuMilli != 8000 || len(r.GpuPendingLatency) != 0 {
		t.Fatalf("%+v", r)
	}
}

// bogus puts everyone on GPU 0 the first time, then behaves.
type bogus struct{ calls *int }

func (b bogus) Decide(c *sim.Cluster, pods []model.Pod) []*mip.Choice {
	*b.calls++
	if *b.calls > 1 {
		return bestFit().Decide(c, pods)
	}
	out := make([]*mip.Choice, len(pods))
	for i := range out {
		out[i] = &mip.Choice{Node: 0, GPUs: []int{0}}
	}
	return out
}

func TestBindRevalidationRequeues(t *testing.T) {
	r := Run(Config{Nodes: nodes(2), Jobs: []trace.Job{gpuJob("a", 0, 10, 1000), gpuJob("b", 0, 10, 1000)},
		Decider: bogus{new(int)}, Batch: 4, Latency: fixed(1)})
	if r.BindConflicts == 0 {
		t.Fatal("second bind on the same GPU must conflict")
	}
	if r.Unplaced != 0 {
		t.Fatalf("conflicting pod must be retried and eventually placed: %+v", r)
	}
}

func TestAllocationTimeAverage(t *testing.T) {
	// one GPU busy for 50 of the 100s window (window = [0, last arrival]).
	r := Run(Config{Nodes: nodes(1), Jobs: []trace.Job{gpuJob("a", 0, 50, 1000), {Pod: model.Pod{Name: "cpu", Res: model.PodRes{MilliCPU: 1}}, Arrive: 100, Duration: 1}},
		Decider: bestFit(), Batch: 4, Latency: fixed(0)})
	if math.Abs(r.AllocTimeAvg-0.5) > 1e-9 {
		t.Fatalf("alloc %v", r.AllocTimeAvg)
	}
}

func TestDeterministic(t *testing.T) {
	var jobs []trace.Job
	rng := rand.New(rand.NewSource(3))
	for i := 0; i < 200; i++ {
		jobs = append(jobs, gpuJob(string(rune(i)), float64(i), 5+rng.Float64()*50, []int64{250, 500, 1000}[rng.Intn(3)]))
	}
	cfg := Config{Nodes: nodes(2, 4, 8), Jobs: jobs, Decider: bestFit(), Batch: 8, Latency: fixed(0.5)}
	a, b := Run(cfg), Run(cfg)
	a.SolveWall, b.SolveWall = nil, nil // wall clock is the only nondeterministic field
	if !reflect.DeepEqual(a, b) {
		t.Fatal("not deterministic")
	}
}
