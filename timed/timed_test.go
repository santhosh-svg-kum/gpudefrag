package timed

import (
	"fmt"
	"math"
	"math/rand"
	"reflect"
	"testing"
	"time"

	"github.com/santhosh-svg-kum/gpudefrag/mip"
	"github.com/santhosh-svg-kum/gpudefrag/model"
	"github.com/santhosh-svg-kum/gpudefrag/sched"
	"github.com/santhosh-svg-kum/gpudefrag/sim"
	"github.com/santhosh-svg-kum/gpudefrag/trace"
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

func (b bogus) DecideRequests(c *sim.Cluster, reqs []mip.Request) [][]*mip.Choice {
	*b.calls++
	if *b.calls > 1 {
		return bestFit().DecideRequests(c, reqs)
	}
	out := make([][]*mip.Choice, len(reqs))
	for i := range out {
		out[i] = []*mip.Choice{{Node: 0, GPUs: []int{0}}}
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

// A decider that always conflicts must not livelock the simulator: retries
// back off in virtual time and the run ends at the drain limit.
type alwaysBad struct{}

func (alwaysBad) DecideRequests(c *sim.Cluster, reqs []mip.Request) [][]*mip.Choice {
	out := make([][]*mip.Choice, len(reqs))
	for i := range out {
		out[i] = []*mip.Choice{{Node: 0, GPUs: nil}} // wrong GPU count
	}
	return out
}

func TestPersistentConflictsBackOff(t *testing.T) {
	r := Run(Config{Nodes: nodes(1), Jobs: []trace.Job{gpuJob("a", 0, 10, 1000)}, Decider: alwaysBad{},
		Batch: 1, Latency: fixed(0), DrainLimit: 100})
	if r.Unplaced != 1 || r.BindConflicts == 0 || r.BindConflicts > 101 {
		t.Fatalf("%+v", r)
	}
}

func TestNoHeadOfLineBlocking(t *testing.T) {
	big := gpuJob("big", 0, 10, 1000)
	big.Pod.Res.GpuNum = 8
	r := Run(Config{Nodes: nodes(2), Jobs: []trace.Job{big, gpuJob("small", 1, 10, 1000)},
		Decider: bestFit(), Batch: 1, Latency: fixed(0)})
	if len(r.GpuPendingLatency) != 1 || r.GpuPendingLatency[0] != 0 {
		t.Fatalf("small pod must not wait behind an unplaceable one: %+v", r)
	}
}

func TestWarmupExcludedFromMetrics(t *testing.T) {
	// j1 (0..50) is warm-up; j2 waits 9s behind it but arrives at 41 > MeasureFrom=40.
	r := Run(Config{Nodes: nodes(1), Jobs: []trace.Job{gpuJob("j1", 0, 50, 1000), gpuJob("j2", 41, 49, 1000)},
		Decider: bestFit(), Batch: 1, Latency: fixed(0), MeasureFrom: 40})
	if !reflect.DeepEqual(r.GpuPendingLatency, []float64{9}) {
		t.Fatalf("latencies %v", r.GpuPendingLatency)
	}
	// window [40, 41]: one GPU fully used
	if math.Abs(r.AllocTimeAvg-1) > 1e-9 {
		t.Fatalf("alloc %v", r.AllocTimeAvg)
	}
}

func gangJob(name string, arrive, dur float64, pods, gpus int) trace.Job {
	j := trace.Job{Arrive: arrive, Duration: dur}
	for k := 0; k < pods; k++ {
		j.Gang = append(j.Gang, model.Pod{Name: fmt.Sprintf("%s-%d", name, k), Res: model.PodRes{MilliCPU: 1000, MemMiB: 1, GpuNum: gpus, GpuMilli: 1000}})
	}
	j.Pod = model.Pod{Name: name, Res: j.Gang[0].Res}
	return j
}

func domainNodes(doms ...string) []*model.NodeRes {
	var out []*model.NodeRes
	for i, d := range doms {
		n := model.NewNode(fmt.Sprintf("n%d", i), 96000, 1<<20, 8, "")
		n.Domain = d
		out = append(out, n)
	}
	return out
}

func TestGangBindsAllOrNothingAndCompletesTogether(t *testing.T) {
	// 1-GPU job holds n0 until 100; the 2x8 gang must wait for two free nodes.
	r := Run(Config{Nodes: domainNodes("d1", "d1"), Jobs: []trace.Job{gpuJob("small", 0, 100, 1000), gangJob("g", 10, 50, 2, 8)},
		Decider: bestFit(), Batch: 4, Latency: fixed(0), LocalGangMax: 4})
	if !reflect.DeepEqual(r.GangPendingLatency, []float64{90}) || !reflect.DeepEqual(r.JCT, []float64{100, 140}) {
		t.Fatalf("gang wait %v jct %v", r.GangPendingLatency, r.JCT)
	}
}

func TestLocalGangRespectsDomains(t *testing.T) {
	// One free node in each domain: a local 2-pod gang cannot start, a non-local one can.
	jobs := []trace.Job{gangJob("g", 0, 10, 2, 8)}
	nodes := domainNodes("d1", "d2")
	if r := Run(Config{Nodes: nodes, Jobs: jobs, Decider: bestFit(), Latency: fixed(0), LocalGangMax: 4, DrainLimit: 100}); r.Unplaced != 1 {
		t.Fatalf("local gang split across domains: %+v", r)
	}
	if r := Run(Config{Nodes: nodes, Jobs: jobs, Decider: bestFit(), Latency: fixed(0), LocalGangMax: 0}); r.Unplaced != 0 {
		t.Fatalf("unconstrained gang should run: %+v", r)
	}
}

func TestPendingAtEndBySize(t *testing.T) {
	big := gpuJob("big", 0, 10, 1000)
	big.Pod.Res.GpuNum = 8
	r := Run(Config{Nodes: nodes(2), Jobs: []trace.Job{big, gpuJob("s", 0, 10, 500)}, Decider: bestFit(), Latency: fixed(0), DrainLimit: 1})
	if !reflect.DeepEqual(r.PendingByGpus, map[int]int{8: 1}) {
		t.Fatalf("%v", r.PendingByGpus)
	}
}
