package workload

import (
	"math"
	"math/rand"
	"reflect"
	"strings"
	"testing"

	"gpupack/model"
)

func pods(spec ...any) []model.Pod {
	// spec: name, cpu, gpuNum, gpuMilli repeated
	var out []model.Pod
	for i := 0; i < len(spec); i += 4 {
		out = append(out, model.Pod{Name: spec[i].(string), Res: model.PodRes{
			MilliCPU: int64(spec[i+1].(int)), MemMiB: 10, GpuNum: spec[i+2].(int), GpuMilli: int64(spec[i+3].(int))}})
	}
	return out
}

func repeat(n int, cpu int, num int, milli int, prefix string) []model.Pod {
	var out []model.Pod
	for i := 0; i < n; i++ {
		out = append(out, model.Pod{Name: prefix + string(rune('a'+i)), Res: model.PodRes{MilliCPU: int64(cpu), MemMiB: int64(i), GpuNum: num, GpuMilli: int64(milli)}})
	}
	return out
}

func TestTypicalPodsKeepsAllWhenCutoffNotReached(t *testing.T) {
	var ps []model.Pod
	ps = append(ps, repeat(6, 1000, 1, 500, "a")...)
	ps = append(ps, repeat(3, 2000, 1, 1000, "b")...)
	ps = append(ps, repeat(1, 4000, 0, 0, "c")...)
	tp := TypicalPods(ps)
	if len(tp) != 3 || tp[0].Pct != 0.6 || tp[1].Pct != 0.3 || tp[2].Pct != 0.1 {
		t.Fatalf("%+v", tp)
	}
	if tp[0].Res.MemMiB != 0 {
		t.Fatal("memory must not be part of the class key")
	}
}

func TestTypicalPodsCutsAt95AndRenormalises(t *testing.T) {
	var ps []model.Pod
	ps = append(ps, repeat(15, 1000, 1, 500, "a")...)
	ps = append(ps, repeat(4, 2000, 1, 1000, "b")...)
	ps = append(ps, repeat(1, 4000, 0, 0, "c")...)
	tp := TypicalPods(ps)
	if len(tp) != 2 {
		t.Fatalf("want 2 classes, got %+v", tp)
	}
	if math.Abs(tp[0].Pct-15.0/19) > 1e-9 || math.Abs(tp[0].Pct+tp[1].Pct-1) > 1e-9 {
		t.Fatalf("%+v", tp)
	}
}

func TestTypicalPodsTieOrderIsReverseLess(t *testing.T) {
	ps := pods("x", 1000, 0, 0, "y", 2000, 0, 0)
	tp := TypicalPods(ps)
	if tp[0].Res.MilliCPU != 2000 {
		t.Fatalf("ties must put the larger resource first: %+v", tp)
	}
}

func TestPrepareIsDeterministicPerSeed(t *testing.T) {
	ps := repeat(20, 1000, 1, 500, "p")
	a := Prepare(rand.New(rand.NewSource(42)), ps, 10000, 0)
	b := Prepare(rand.New(rand.NewSource(42)), ps, 10000, 0)
	c := Prepare(rand.New(rand.NewSource(43)), ps, 10000, 0)
	if !reflect.DeepEqual(a, b) {
		t.Fatal("same seed must give same order")
	}
	if reflect.DeepEqual(a, c) {
		t.Fatal("different seeds should differ")
	}
	if ps[0].Name != "pa" {
		t.Fatal("input must not be mutated")
	}
}

func TestPrepareTunesUp(t *testing.T) {
	ps := pods("b", 1, 1, 500, "a", 1, 1, 500)                 // 1000 milli
	out := Prepare(rand.New(rand.NewSource(1)), ps, 1000, 2.0) // target 2000
	var total int64
	for i, p := range out {
		total += p.Res.TotalMilliGpu()
		if i >= 2 && !strings.Contains(p.Name, "-tuned-") {
			t.Fatalf("tuned pods must be appended: %v", out)
		}
	}
	if total != 2000 || len(out) != 4 {
		t.Fatalf("total=%d n=%d", total, len(out))
	}
}

func TestPrepareTunesDown(t *testing.T) {
	ps := repeat(6, 1, 1, 500, "p") // 3000 milli
	out := Prepare(rand.New(rand.NewSource(1)), ps, 1000, 1.0)
	var total int64
	for _, p := range out {
		total += p.Res.TotalMilliGpu()
	}
	if total > 1000 || len(out) != 2 {
		t.Fatalf("total=%d n=%d", total, len(out))
	}
}

func TestTypicalPodsWeightedByGPU(t *testing.T) {
	var ps []model.Pod
	ps = append(ps, repeat(8, 1000, 1, 500, "s")...)  // share pods
	ps = append(ps, repeat(2, 1000, 8, 1000, "w")...) // 8-GPU pods
	tp := TypicalPodsWeighted(ps, 1)                  // whole-GPU pod counts 1 + 8*1 = 9
	if tp[0].Res.GpuNum != 8 || math.Abs(tp[0].Pct-18.0/26) > 1e-9 {
		t.Fatalf("%+v", tp)
	}
	if !reflect.DeepEqual(TypicalPodsWeighted(ps, 0), TypicalPods(ps)) {
		t.Fatal("weight 0 must equal FGD's default")
	}
}
