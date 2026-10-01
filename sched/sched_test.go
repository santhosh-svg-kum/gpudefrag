package sched

import (
	"math/rand"
	"reflect"
	"testing"

	"github.com/santhosh-svg-kum/gpudefrag/frag"
	"github.com/santhosh-svg-kum/gpudefrag/model"
	"github.com/santhosh-svg-kum/gpudefrag/sim"
)

func node(name string, cpuLeft int64, gpuLeft ...int64) *model.NodeRes {
	n := model.NewNode(name, 128000, 1<<20, len(gpuLeft), "")
	n.CPULeft = cpuLeft
	copy(n.GpuLeft, gpuLeft)
	return n
}

func place(t *testing.T, pol Policy, nodes []*model.NodeRes, p model.PodRes) (string, []int) {
	t.Helper()
	c := sim.NewCluster(nodes)
	i, gpus, ok := Place(c, pol, model.Pod{Name: "p", Res: p}, rand.New(rand.NewSource(1)))
	if !ok {
		t.Fatal("not placed")
	}
	return c.Nodes[i].Name, gpus
}

func TestBestFit(t *testing.T) {
	// raw: A=66, B=97 -> normalised A=0, B=100
	got, gpus := place(t, BestFit{}, []*model.NodeRes{node("A", 64000, 1000, 1000), node("B", 8000, 500)},
		model.PodRes{MilliCPU: 4000, GpuNum: 1, GpuMilli: 400})
	if got != "B" || !reflect.DeepEqual(gpus, []int{0}) {
		t.Fatalf("%s %v", got, gpus)
	}
}

func TestGpuPacking(t *testing.T) {
	// A fully free: 100/3-2 = 31; B uses a shared GPU: 100 - 60/10 = 94
	got, gpus := place(t, GpuPacking{}, []*model.NodeRes{node("A", 64000, 1000, 1000), node("B", 64000, 600, 1000)},
		model.PodRes{MilliCPU: 1000, GpuNum: 1, GpuMilli: 400})
	if got != "B" || !reflect.DeepEqual(gpus, []int{0}) {
		t.Fatalf("%s %v", got, gpus)
	}
}

func TestGpuClustering(t *testing.T) {
	a := node("A", 64000, 1000, 500)
	a.Affinity["share-gpu"] = 1       // 25*6500/8000 + 75 = 95
	b := node("B", 64000, 1000, 1000) // 25*6000/8000 + 25 = 43
	got, _ := place(t, GpuClustering{}, []*model.NodeRes{b, a}, model.PodRes{MilliCPU: 1000, GpuNum: 1, GpuMilli: 400})
	if got != "A" {
		t.Fatal(got)
	}
}

func TestDotProd(t *testing.T) {
	// A: 1-(0.5*0.0625+0.25*0.0625)/2 = 0.977 -> 97 ; B: 1-(0.125*0.0625+0.075*0.0625)/2 -> 99
	got, _ := place(t, DotProd{}, []*model.NodeRes{node("A", 64000, 1000, 1000), node("B", 16000, 600)},
		model.PodRes{MilliCPU: 8000, GpuNum: 1, GpuMilli: 500})
	if got != "B" {
		t.Fatal(got)
	}
}

func TestFGD(t *testing.T) {
	typ := []frag.TargetPod{{Res: model.PodRes{MilliCPU: 1000, GpuNum: 1, GpuMilli: 1000}, Pct: 1}}
	// B gpu1: frag 500 -> 0, sigmoid(0.5)=62; A: 0 -> 500, sigmoid(-0.5)=37
	got, gpus := place(t, NewFGD(typ), []*model.NodeRes{node("A", 64000, 1000, 1000), node("B", 64000, 1000, 500)},
		model.PodRes{MilliCPU: 1000, GpuNum: 1, GpuMilli: 500})
	if got != "B" || !reflect.DeepEqual(gpus, []int{1}) {
		t.Fatalf("%s %v", got, gpus)
	}
}

func TestTiesGoToSmallestName(t *testing.T) {
	got, _ := place(t, GpuClustering{}, []*model.NodeRes{node("b", 64000, 1000), node("a", 64000, 1000)},
		model.PodRes{MilliCPU: 1, GpuNum: 1, GpuMilli: 1000})
	if got != "a" {
		t.Fatal(got)
	}
}

func TestRandomUsesRNGAndSkipsSingleNode(t *testing.T) {
	nodes := []*model.NodeRes{node("a", 64000, 1000), node("b", 64000, 1000), node("c", 64000, 1000)}
	p := model.Pod{Name: "p", Res: model.PodRes{MilliCPU: 1, GpuNum: 1, GpuMilli: 1000}}
	c := sim.NewCluster(nodes)
	rng := rand.New(rand.NewSource(7))
	i, _, _ := Place(c, Random{}, p, rng)
	if want := rand.New(rand.NewSource(7)).Intn(3); i != want {
		t.Fatalf("got %d want %d", i, want)
	}

	one := sim.NewCluster(nodes[:1])
	r1, r2 := rand.New(rand.NewSource(9)), rand.New(rand.NewSource(9))
	Place(one, Random{}, p, r1)
	if r1.Int() != r2.Int() {
		t.Fatal("a single feasible node must not consume randomness")
	}
}

func TestRandomFitGPU(t *testing.T) {
	n := node("a", 64000, 1000, 100, 600, 700)
	seen := map[int]bool{}
	rng := rand.New(rand.NewSource(3))
	for k := 0; k < 200; k++ {
		g := Random{}.SelectGPUs(n, model.PodRes{GpuNum: 1, GpuMilli: 500}, rng)
		seen[g[0]] = true
	}
	if seen[1] || len(seen) != 3 {
		t.Fatalf("random fit must pick among fitting GPUs only: %v", seen)
	}
}

func TestExclusiveGPUsTakeFirstFullyFree(t *testing.T) {
	n := node("a", 64000, 500, 1000, 1000, 1000)
	if got := ExclusiveGPUs(n, model.PodRes{GpuNum: 2, GpuMilli: 1000}); !reflect.DeepEqual(got, []int{1, 2}) {
		t.Fatal(got)
	}
	if got := BestFitGPUs(n, model.PodRes{GpuNum: 1, GpuMilli: 300}); !reflect.DeepEqual(got, []int{0}) {
		t.Fatal(got)
	}
	if ExclusiveGPUs(n, model.PodRes{MilliCPU: 1}) != nil {
		t.Fatal("cpu pods get no GPUs")
	}
}

func TestUnplaceable(t *testing.T) {
	c := sim.NewCluster([]*model.NodeRes{node("a", 64000, 1000)})
	_, _, ok := Place(c, BestFit{}, model.Pod{Res: model.PodRes{MilliCPU: 1, GpuNum: 1, GpuMilli: 1000, GpuType: "H100"}}, rand.New(rand.NewSource(1)))
	if ok {
		t.Fatal("pod with unavailable GPU type must not place")
	}
}

func TestNew(t *testing.T) {
	for _, name := range Names {
		if p, err := New(name, nil); err != nil || p.Name() != name {
			t.Fatalf("%s: %v", name, err)
		}
	}
	if _, err := New("nope", nil); err == nil {
		t.Fatal("unknown policy")
	}
}
