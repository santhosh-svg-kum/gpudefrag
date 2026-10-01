package workload

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"gpupack/model"
	"gpupack/trace"
)

func baseJobs() []trace.Job {
	return []trace.Job{
		{Pod: model.Pod{Name: "a", Res: model.PodRes{MilliCPU: 1000, GpuNum: 1, GpuMilli: 500}}, Duration: 100},
		{Pod: model.Pod{Name: "b", Res: model.PodRes{MilliCPU: 1000, GpuNum: 2, GpuMilli: 1000}}, Duration: 300},
		{Pod: model.Pod{Name: "c", Res: model.PodRes{MilliCPU: 1000}}, Duration: 50},
		{Pod: model.Pod{Name: "bad", Res: model.PodRes{MilliCPU: 1000}}, Duration: 0},
	}
}

func TestSynthesizeHitsOfferedLoad(t *testing.T) {
	const capMilli = 100_000
	jobs := Synthesize(rand.New(rand.NewSource(1)), baseJobs(), capMilli, 0.9, 50000, 7*86400)
	if len(jobs) != 50000 {
		t.Fatal(len(jobs))
	}
	var work float64
	for i, j := range jobs {
		if j.Duration <= 0 || j.Pod.Name == "bad" {
			t.Fatal("zero-duration rows must be dropped")
		}
		if i > 0 && j.Arrive < jobs[i-1].Arrive {
			t.Fatal("arrivals must be sorted")
		}
		work += float64(j.Pod.Res.TotalMilliGpu()) * j.Duration
	}
	rho := work / (capMilli * jobs[len(jobs)-1].Arrive)
	if math.Abs(rho-0.9) > 0.05 {
		t.Fatalf("offered load %.3f, want 0.9", rho)
	}
	if jobs[0].Pod.Name == jobs[1].Pod.Name {
		t.Fatal("synthesized pods need unique names")
	}
}

func TestSynthesizeCapsDurations(t *testing.T) {
	b := baseJobs()
	b[1].Duration = 1e9
	for _, j := range Synthesize(rand.New(rand.NewSource(2)), b, 100_000, 1, 1000, 3600) {
		if j.Duration > 3600 {
			t.Fatal("duration not capped")
		}
	}
}

func TestSynthesizeDeterministic(t *testing.T) {
	a := Synthesize(rand.New(rand.NewSource(5)), baseJobs(), 100_000, 1, 100, 1e6)
	b := Synthesize(rand.New(rand.NewSource(5)), baseJobs(), 100_000, 1, 100, 1e6)
	for i := range a {
		if a[i] != b[i] {
			t.Fatal("same seed must give same jobs")
		}
	}
}

func TestSubsetNodesStratified(t *testing.T) {
	var ns []*model.NodeRes
	for i := 0; i < 80; i++ {
		ns = append(ns, model.NewNode(fmt.Sprintf("v%02d", i), 1, 1, 8, "V100"))
	}
	for i := 0; i < 16; i++ {
		ns = append(ns, model.NewNode(fmt.Sprintf("t%02d", i), 1, 1, 2, "T4"))
	}
	sub := SubsetNodes(rand.New(rand.NewSource(1)), ns, 0.125)
	v, t4 := 0, 0
	for _, n := range sub {
		if n.GpuType == "V100" {
			v++
		} else {
			t4++
		}
	}
	if v != 10 || t4 != 2 {
		t.Fatalf("v=%d t4=%d", v, t4)
	}
}

func TestArrivalRate(t *testing.T) {
	r := ArrivalRate(baseJobs(), 100_000, 1, 1e9)
	// mean work = (500*100 + 2000*300 + 0*50)/3
	if want := 100_000 / ((500*100 + 2000*300 + 0) / 3.0); math.Abs(r-want) > 1e-12 {
		t.Fatalf("rate %v want %v", r, want)
	}
}
