package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"gpupack/mip"
	"gpupack/model"
	"gpupack/sched"
	"gpupack/timed"
	"gpupack/trace"
	"gpupack/workload"
)

type run1b struct {
	Variant    string  `json:"variant"`
	Load       float64 `json:"load"`
	Seed       int64   `json:"seed"`
	P50, P95   float64 // GPU pending latency, seconds
	P99        float64
	Alloc      float64 `json:"alloc_pct"`
	Unplaced   int     `json:"unplaced"`
	Conflicts  int     `json:"bind_conflicts"`
	Plans      int     `json:"defrag_plans"`
	Migrations int     `json:"migrations"`
	LostGpuH   float64 `json:"lost_gpu_hours"`
	Rejected   map[string]int
	SolveP99   float64 `json:"solve_p99_ms"`
	WallS      float64 `json:"wall_s"`
}

func runSuite1b(args []string) error {
	fs := flag.NewFlagSet("suite1b", flag.ExitOnError)
	data := fs.String("data", "data", "data directory")
	out := fs.String("out", "results/suite1b", "output directory")
	variants := fs.String("variants", "FGD,BestFit,gpupack,FGD+defrag,gpupack+defrag", "comma-separated variants")
	loads := fs.String("loads", "0.9,1.0,1.1", "offered GPU loads")
	frac := fs.Float64("cluster-frac", 0.125, "stratified fraction of openb nodes to simulate")
	maxDur := fs.Duration("max-dur", 2*time.Hour, "cap on job durations")
	warm := fs.Duration("warmup", 2*time.Hour, "warm-up excluded from metrics")
	measure := fs.Duration("measure", 6*time.Hour, "measurement window")
	seedLo := fs.Int64("seed-from", 42, "first seed")
	seedHi := fs.Int64("seed-to", 46, "last seed")
	batch := fs.Int("batch", 16, "pods per scheduling session")
	limit := fs.Duration("time", 500*time.Millisecond, "gpupack solver budget per batch")
	parallel := fs.Int("parallel", 5, "concurrent runs")
	solverDir := fs.String("solver-dir", "solver", "python solver project")
	addr := fs.String("solver", "", "solver address (default: start one locally)")
	fs.Parse(args)

	allNodes, err := trace.LoadOpenbNodes(filepath.Join(*data, "openb", "openb_node_list_gpu_node.csv"))
	if err != nil {
		return err
	}
	nodes := workload.SubsetNodes(rand.New(rand.NewSource(7)), allNodes, *frac)
	base, err := trace.LoadOpenbJobs(filepath.Join(*data, "openb", "openb_pod_list_default.csv"))
	if err != nil {
		return err
	}
	var basePods []model.Pod
	for _, j := range base {
		basePods = append(basePods, j.Pod)
	}
	typical := workload.TypicalPods(basePods)
	var capMilli int64
	for _, nd := range nodes {
		capMilli += int64(nd.GpuNum()) * model.Milli
	}

	var client *mip.Client
	if strings.Contains(*variants, "gpupack") || strings.Contains(*variants, "defrag") {
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
		variant string
		load    float64
		seed    int64
	}
	var mu sync.Mutex
	var runs []run1b
	jobs := make(chan job)
	var wg sync.WaitGroup
	for w := 0; w < *parallel; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := range jobs {
				t0 := time.Now()
				window := (*warm + *measure).Seconds()
				n := int(workload.ArrivalRate(base, capMilli, j.load, maxDur.Seconds()) * window)
				arrivals := workload.Synthesize(rand.New(rand.NewSource(j.seed)), base, capMilli, j.load, n, maxDur.Seconds())
				cfg := timed.Config{Nodes: nodes, Jobs: arrivals, Batch: *batch, MeasureFrom: warm.Seconds(), DrainLimit: 1}
				name, defrag := strings.CutSuffix(j.variant, "+defrag")
				switch name {
				case "gpupack":
					cfg.Decider = &mip.Placer{Solver: client, Typical: typical, K: 16, TimeLimit: *limit, Deterministic: true, Workers: cpWorkers}
				default:
					pol, err := sched.New(name, typical)
					if err != nil {
						panic(err)
					}
					cfg.Decider = timed.PolicyDecider{Policy: pol}
				}
				if defrag {
					cfg.Defrag = &timed.DefragConfig{
						Planner:  &mip.Defragmenter{Solver: client, MaxMoves: 4, TimeLimit: time.Second, Deterministic: true, Workers: cpWorkers},
						Interval: 60, CheckpointInterval: 1800, Restart: 60, BenefitHorizon: 3600, Margin: 0.2, Seed: j.seed,
					}
				}
				r := timed.Run(cfg)
				lat := append([]float64(nil), r.GpuPendingLatency...)
				sort.Float64s(lat)
				q := func(p float64) float64 {
					if len(lat) == 0 {
						return 0
					}
					return lat[min(len(lat)-1, int(p*float64(len(lat))))]
				}
				res := run1b{Variant: j.variant, Load: j.load, Seed: j.seed, P50: q(0.5), P95: q(0.95), P99: q(0.99),
					Alloc: 100 * r.AllocTimeAvg, Unplaced: r.Unplaced, Conflicts: r.BindConflicts,
					Plans: r.Defrag.Plans, Migrations: r.Defrag.Migrations, LostGpuH: r.Defrag.LostGpuSec / 3600,
					Rejected: r.Defrag.Rejected, SolveP99: pctl(r.SolveWall, 0.99), WallS: time.Since(t0).Seconds()}
				mu.Lock()
				runs = append(runs, res)
				fmt.Fprintf(os.Stderr, "  %-16s load %.1f seed %d: p95 %.0fs alloc %.2f%% unplaced %d moves %d (%.0fs)\n",
					j.variant, j.load, j.seed, res.P95, res.Alloc, res.Unplaced, res.Migrations, res.WallS)
				mu.Unlock()
			}
		}()
	}
	vs := strings.Split(*variants, ",")
	var ls []float64
	for _, l := range strings.Split(*loads, ",") {
		v, err := strconv.ParseFloat(l, 64)
		if err != nil {
			return err
		}
		ls = append(ls, v)
	}
	for _, l := range ls {
		for _, v := range vs {
			for s := *seedLo; s <= *seedHi; s++ {
				jobs <- job{v, l, s}
			}
		}
	}
	close(jobs)
	wg.Wait()

	if err := os.MkdirAll(*out, 0o755); err != nil {
		return err
	}
	b, _ := json.MarshalIndent(runs, "", "  ")
	if err := os.WriteFile(filepath.Join(*out, "runs.json"), b, 0o644); err != nil {
		return err
	}
	desc := fmt.Sprintf("%d of %d openb GPU nodes (stratified by GPU type and count); Poisson arrivals at the given offered GPU load; (pod, duration) pairs resampled from openb with durations capped at %s; %s warm-up excluded, metrics over the next %s", len(nodes), len(allNodes), *maxDur, *warm, *measure)
	report := suite1bReport(vs, ls, runs, *seedLo, *seedHi, desc)
	fmt.Print(report)
	return os.WriteFile(filepath.Join(*out, "report.md"), []byte(report), 0o644)
}

func suite1bReport(vs []string, ls []float64, runs []run1b, lo, hi int64, desc string) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# Suite 1b — timed replay, SYNTHETIC workload bootstrapped from openb\n\n")
	fmt.Fprintf(&b, "%s. Seeds %d–%d, mean ± 95%% CI. Latency = pending time (s) of GPU pods that arrived in the window and were bound. Alloc = time-averaged allocated GPU. \"Pending at end\" = backlog when the window closes.\n\n", desc, lo, hi)
	for _, l := range ls {
		fmt.Fprintf(&b, "## Offered load %.1f\n\n", l)
		fmt.Fprintf(&b, "| variant | p50 s | p95 s | p99 s | alloc %% | pending at end | migrations | lost GPU-h | solve p99 ms |\n|---|---|---|---|---|---|---|---|---|\n")
		for _, v := range vs {
			var rs []run1b
			for _, r := range runs {
				if r.Variant == v && r.Load == l {
					rs = append(rs, r)
				}
			}
			col := func(f func(r run1b) float64, prec int) string {
				x := make([]float64, len(rs))
				for i, r := range rs {
					x[i] = f(r)
				}
				m, ci := meanCI(x)
				return fmt.Sprintf("%.*f ± %.*f", prec, m, prec, ci)
			}
			fmt.Fprintf(&b, "| %s | %s | %s | %s | %s | %s | %s | %s | %s |\n", v,
				col(func(r run1b) float64 { return r.P50 }, 1), col(func(r run1b) float64 { return r.P95 }, 0),
				col(func(r run1b) float64 { return r.P99 }, 0), col(func(r run1b) float64 { return r.Alloc }, 2),
				col(func(r run1b) float64 { return float64(r.Unplaced) }, 1), col(func(r run1b) float64 { return float64(r.Migrations) }, 1),
				col(func(r run1b) float64 { return r.LostGpuH }, 1), col(func(r run1b) float64 { return r.SolveP99 }, 0))
		}
		b.WriteString("\n")
	}
	return b.String()
}
