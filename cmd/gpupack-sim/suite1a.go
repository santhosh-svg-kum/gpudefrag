package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"gpupack/calib"
	"gpupack/mip"
	"gpupack/model"
	"gpupack/trace"
	"gpupack/workload"
)

// variant is a baseline name or "gpupack:b=<batch>:t=<time limit>[:k=<cands>]".
type variant struct {
	Name  string
	Batch int
	Limit time.Duration
	K     int
}

func parseVariant(s string) (variant, error) {
	parts := strings.Split(s, ":")
	v := variant{Name: s, Batch: 1}
	if parts[0] != "gpupack" {
		return v, nil
	}
	v.Batch, v.Limit, v.K = 16, 500*time.Millisecond, 16
	for _, p := range parts[1:] {
		k, val, ok := strings.Cut(p, "=")
		if !ok {
			return v, fmt.Errorf("bad variant option %q", p)
		}
		var err error
		switch k {
		case "b":
			v.Batch, err = strconv.Atoi(val)
		case "t":
			v.Limit, err = time.ParseDuration(val)
		case "k":
			v.K, err = strconv.Atoi(val)
		default:
			err = fmt.Errorf("unknown option %q", k)
		}
		if err != nil {
			return v, err
		}
	}
	return v, nil
}

type runSummary struct {
	Seed         int64     `json:"seed"`
	Curve        []float64 `json:"-"`
	FinalAlloc   float64   `json:"final_alloc_pct"`
	FailedGpuPct float64   `json:"failed_gpu_pct"` // unplaced GPU demand / capacity
	Improved     int       `json:"improved_batches"`
	Batches      int       `json:"batches"`
	Fallbacks    map[string]int
	SolveP50     float64 `json:"solve_p50_ms"`
	SolveP99     float64 `json:"solve_p99_ms"`
	WallS        float64 `json:"wall_s"`
}

func runSuite1a(args []string) error {
	fs := flag.NewFlagSet("suite1a", flag.ExitOnError)
	data := fs.String("data", "data", "data directory")
	out := fs.String("out", "results/suite1a", "output directory")
	variants := fs.String("variants", "FGD,BestFit,gpupack:b=1:t=500ms,gpupack:b=16:t=500ms", "comma-separated variants")
	seedLo := fs.Int64("seed-from", 42, "first seed")
	seedHi := fs.Int64("seed-to", 51, "last seed")
	parallel := fs.Int("parallel", 5, "concurrent runs")
	solverDir := fs.String("solver-dir", "solver", "python solver project")
	addr := fs.String("solver", "", "solver address (default: start one locally)")
	fs.Parse(args)

	nodes, err := trace.LoadOpenbNodes(filepath.Join(*data, "openb", "openb_node_list_gpu_node.csv"))
	if err != nil {
		return err
	}
	pods, err := trace.LoadOpenbPods(filepath.Join(*data, "openb", "openb_pod_list_default.csv"))
	if err != nil {
		return err
	}
	typical := workload.TypicalPods(pods)
	var totalGpus int64
	for _, n := range nodes {
		totalGpus += int64(n.GpuNum())
	}
	var vs []variant
	needSolver := false
	for _, s := range strings.Split(*variants, ",") {
		v, err := parseVariant(s)
		if err != nil {
			return err
		}
		vs = append(vs, v)
		needSolver = needSolver || strings.HasPrefix(s, "gpupack")
	}
	var client *mip.Client
	if needSolver {
		if *addr == "" {
			a, stop, err := mip.StartLocal(context.Background(), *solverDir)
			if err != nil {
				return err
			}
			defer stop()
			*addr = a
		}
		if client, err = mip.Dial(*addr); err != nil {
			return err
		}
		defer client.Close()
	}
	cpWorkers := max(1, runtime.NumCPU() / *parallel)

	type job struct {
		v    variant
		seed int64
	}
	results := map[string][]runSummary{}
	var mu sync.Mutex
	var errs []error
	jobs := make(chan job)
	var wg sync.WaitGroup
	for w := 0; w < *parallel; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := range jobs {
				t0 := time.Now()
				cfg := calib.RunConfig{Nodes: nodes, Pods: pods, Policy: j.v.Name, Seed: j.seed, Ratio: 1.3}
				var pl *mip.Placer
				if strings.HasPrefix(j.v.Name, "gpupack") {
					pl = &mip.Placer{Solver: client, Typical: typical, K: j.v.K, TimeLimit: j.v.Limit, Deterministic: true, Workers: cpWorkers}
					cfg.Placer, cfg.Batch = pl, j.v.Batch
				}
				r, err := calib.RunDetailed(cfg)
				mu.Lock()
				if err != nil {
					errs = append(errs, err)
					mu.Unlock()
					continue
				}
				c := calib.Discretize(r.Samples, totalGpus)
				last := r.Samples[len(r.Samples)-1]
				s := runSummary{Seed: j.seed, Curve: c[:], WallS: time.Since(t0).Seconds(),
					FinalAlloc:   100 * float64(last.Used) / float64(totalGpus*model.Milli),
					FailedGpuPct: 100 * float64(r.FailedMilli) / float64(totalGpus*model.Milli)}
				if pl != nil {
					s.Improved, s.Batches, s.Fallbacks = pl.Stats.Improved, pl.Stats.Batches, pl.Stats.Fallbacks
					s.SolveP50, s.SolveP99 = pctl(pl.Stats.SolveWall, 0.5), pctl(pl.Stats.SolveWall, 0.99)
				}
				results[j.v.Name] = append(results[j.v.Name], s)
				fmt.Fprintf(os.Stderr, "  %-28s seed %d: final %.2f%% (%.0fs)\n", j.v.Name, j.seed, s.FinalAlloc, s.WallS)
				mu.Unlock()
			}
		}()
	}
	for _, v := range vs {
		for s := *seedLo; s <= *seedHi; s++ {
			jobs <- job{v, s}
		}
	}
	close(jobs)
	wg.Wait()
	if len(errs) > 0 {
		return errors.Join(errs...)
	}
	if err := os.MkdirAll(*out, 0o755); err != nil {
		return err
	}
	for _, rs := range results {
		sort.Slice(rs, func(a, b int) bool { return rs[a].Seed < rs[b].Seed })
	}
	b, _ := json.MarshalIndent(results, "", "  ")
	if err := os.WriteFile(filepath.Join(*out, "runs.json"), b, 0o644); err != nil {
		return err
	}
	report := suite1aReport(vs, results, *seedLo, *seedHi)
	fmt.Print(report)
	return os.WriteFile(filepath.Join(*out, "report.md"), []byte(report), 0o644)
}

func suite1aReport(vs []variant, results map[string][]runSummary, lo, hi int64) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# Suite 1a — openb default, FGD protocol (arrival-only, 130%% load)\n\n")
	fmt.Fprintf(&b, "Seeds %d–%d, mean ± 95%% CI. Allocation = allocated GPU / cluster GPU at the given arrived load.\n\n", lo, hi)
	fmt.Fprintf(&b, "| variant | alloc@100%% | alloc@120%% | alloc@130%% | final alloc %% | unplaced GPU demand %% of cap | improved batches | solve p50 / p99 ms | fallbacks |\n")
	fmt.Fprintf(&b, "|---|---|---|---|---|---|---|---|---|\n")
	for _, v := range vs {
		rs := results[v.Name]
		col := func(f func(r runSummary) float64) string {
			x := make([]float64, len(rs))
			for i, r := range rs {
				x[i] = f(r)
			}
			m, ci := meanCI(x)
			return fmt.Sprintf("%.2f ± %.2f", m, ci)
		}
		at := func(k int) func(r runSummary) float64 { return func(r runSummary) float64 { return r.Curve[k] } }
		imp, bat := 0, 0
		var p50, p99 float64
		fb := map[string]int{}
		for _, r := range rs {
			imp += r.Improved
			bat += r.Batches
			p50, p99 = max(p50, r.SolveP50), max(p99, r.SolveP99)
			for k, n := range r.Fallbacks {
				fb[k] += n
			}
		}
		impS, solveS, fbS := "–", "–", "–"
		if bat > 0 {
			impS = fmt.Sprintf("%d / %d", imp, bat)
			solveS = fmt.Sprintf("%.0f / %.0f", p50, p99)
			fbS = fmt.Sprint(fb)
		}
		fmt.Fprintf(&b, "| %s | %s | %s | %s | %s | %s | %s | %s | %s |\n", v.Name, col(at(100)), col(at(120)), col(at(130)),
			col(func(r runSummary) float64 { return r.FinalAlloc }), col(func(r runSummary) float64 { return r.FailedGpuPct }), impS, solveS, fbS)
	}
	return b.String()
}

func meanCI(x []float64) (float64, float64) {
	n := float64(len(x))
	if n == 0 {
		return math.NaN(), math.NaN()
	}
	var m float64
	for _, v := range x {
		m += v
	}
	m /= n
	if n < 2 {
		return m, 0
	}
	var ss float64
	for _, v := range x {
		ss += (v - m) * (v - m)
	}
	t := map[int]float64{2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45, 8: 2.36, 9: 2.31, 10: 2.26}[int(n)]
	if t == 0 {
		t = 1.96
	}
	return m, t * math.Sqrt(ss/(n-1)) / math.Sqrt(n)
}

func pctl(d []time.Duration, q float64) float64 {
	if len(d) == 0 {
		return 0
	}
	s := append([]time.Duration(nil), d...)
	sort.Slice(s, func(a, b int) bool { return s[a] < s[b] })
	return float64(s[min(len(s)-1, int(q*float64(len(s))))].Microseconds()) / 1000
}
