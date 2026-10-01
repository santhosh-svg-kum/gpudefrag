package calib

import (
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"gpupack/model"
)

func TestDiscretize(t *testing.T) {
	// 10 GPUs: arrive% = arrived/10/10.
	s := []Sample{{0, 0}, {50, 50}, {150, 100}, {250, 200}, {2000, 900}}
	c := Discretize(s, 10)
	// k=0: samples (0,0) and (50 -> 0.5 rounds half-even to 0) -> mean(0, 0.5)
	// k=1: no exact sample; window [0,2] -> mean(0, 0.5, 1, 2) = 0.875 -> 0.88
	if c[0] != 0.25 || c[1] != 0.88 {
		t.Fatalf("k0=%v k1=%v", c[0], c[1])
	}
	// k=2: 150 -> 1.5 rounds half-even to 2; 250 -> 2.5 rounds to 2. mean(1.0,2.0)=1.5
	if c[2] != 1.5 {
		t.Fatalf("k2=%v", c[2])
	}
	// k=3: no exact sample, window [2,4] has the two k=2 samples
	if c[3] != 1.5 {
		t.Fatalf("k3=%v", c[3])
	}
	if !math.IsNaN(c[10]) || c[20] != 9 {
		t.Fatalf("k10=%v k20=%v", c[10], c[20])
	}
}

func TestLoadReference(t *testing.T) {
	head := "workload,sc_policy,tune,seed,total_gpus"
	for k := 0; k <= 130; k++ {
		head += "," + itoa(k)
	}
	row := func(w, p, seed string, v float64) string {
		s := w + "," + p + ",1.3," + seed + ",6212"
		for k := 0; k <= 130; k++ {
			s += "," + ftoa(v)
		}
		return s
	}
	body := strings.Join([]string{head,
		row("openb_pod_list_default", "06-FGD", "42", 95),
		row("openb_pod_list_default", "01-Random", "42", 87),
		row("openb_pod_list_cpu050", "06-FGD", "42", 1),
	}, "\n")
	path := filepath.Join(t.TempDir(), "ref.csv")
	os.WriteFile(path, []byte(body), 0o644)
	ref, err := LoadReference(path, "openb_pod_list_default", "1.3")
	if err != nil {
		t.Fatal(err)
	}
	if ref["FGD"][42][100] != 95 || ref["Random"][42][130] != 87 || len(ref) != 2 {
		t.Fatalf("%v", ref)
	}
}

func flat(v float64) Curve {
	var c Curve
	for i := range c {
		c[i] = v
	}
	return c
}

func TestGate(t *testing.T) {
	seeds := []int64{42, 43}
	ref := map[string]map[int64]Curve{"FGD": {42: flat(95), 43: flat(95)}}
	ok := map[string]map[int64]Curve{"FGD": {42: flat(95.5), 43: flat(94.9)}}
	if g := Gate(ok, ref, []string{"FGD"}, seeds, GatePoints, 1.0); !g.Pass {
		t.Fatalf("should pass: %+v", g)
	}
	bad := map[string]map[int64]Curve{"FGD": {42: flat(93), 43: flat(93)}}
	if g := Gate(bad, ref, []string{"FGD"}, seeds, GatePoints, 1.0); g.Pass {
		t.Fatal("2pp off must fail")
	}
	missing := map[string]map[int64]Curve{"FGD": {42: flat(95)}}
	if g := Gate(missing, ref, []string{"FGD"}, seeds, GatePoints, 1.0); g.Pass {
		t.Fatal("missing seed must fail")
	}
	nan := flat(95)
	nan[130] = math.NaN()
	withNaN := map[string]map[int64]Curve{"FGD": {42: nan, 43: flat(95)}}
	if g := Gate(withNaN, ref, []string{"FGD"}, seeds, GatePoints, 1.0); g.Pass {
		t.Fatal("NaN point must fail")
	}
	if g := Gate(ok, ref, []string{"FGD", "BestFit"}, seeds, GatePoints, 1.0); g.Pass {
		t.Fatal("policy missing from reference must fail")
	}
}

func TestRunIsDeterministicAndIsolated(t *testing.T) {
	nodes := []*model.NodeRes{model.NewNode("a", 64000, 1<<20, 2, ""), model.NewNode("b", 64000, 1<<20, 2, "")}
	var pods []model.Pod
	for i := 0; i < 10; i++ {
		pods = append(pods, model.Pod{Name: "p" + itoa(i), Res: model.PodRes{MilliCPU: 1000, MemMiB: 1, GpuNum: 1, GpuMilli: int64(100 + 80*i)}})
	}
	cfg := RunConfig{Nodes: nodes, Pods: pods, Policy: "FGD", Seed: 42, Ratio: 1.3}
	a, err := Run(cfg)
	if err != nil {
		t.Fatal(err)
	}
	b, _ := Run(cfg)
	if len(a) != len(b) {
		t.Fatal("len")
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("run not deterministic at %d", i)
		}
	}
	if nodes[0].TotalGpuLeft() != 2000 {
		t.Fatal("Run mutated the input nodes")
	}
	last := a[len(a)-1]
	if last.Arrived < 4600 || last.Arrived > 5200 || last.Used > 4000 {
		t.Fatalf("arrivals must stop at or below 130%% and use is capped: %+v", last)
	}
}

func itoa(i int) string     { return strconv.Itoa(i) }
func ftoa(f float64) string { return strconv.FormatFloat(f, 'f', -1, 64) }
