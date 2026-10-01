package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
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

const day = 86400.0

type run2 struct {
	Variant    string  `json:"variant"`
	Compress   float64 `json:"compress"`
	Window     int     `json:"window"`
	P50, P95   float64
	P99        float64
	GangP95    float64 `json:"gang_p95"`
	JCT        float64 `json:"jct_mean_h"`
	Alloc      float64 `json:"alloc_pct"`
	Pending    int     `json:"pending_at_end"`
	Migrations int     `json:"migrations"`
	LostGpuH   float64 `json:"lost_gpu_hours"`
	SolveP99   float64 `json:"solve_p99_ms"`
	WallS      float64 `json:"wall_s"`
}

// busiestWeeks returns start offsets (seconds) of the n busiest non-overlapping
// 7-day windows by submitted GPU-hours, each preceded by `warm` of history.
func busiestWeeks(jobs []trace.Job, n int, warm float64) []float64 {
	if len(jobs) == 0 {
		return nil
	}
	days := int(jobs[len(jobs)-1].Arrive/day) + 1
	load := make([]float64, days)
	for _, j := range jobs {
		load[int(j.Arrive/day)] += float64(len(j.Gang)*j.Pod.Res.GpuNum) * j.Duration
	}
	type win struct {
		start int
		sum   float64
	}
	var ws []win
	first := int(warm/day + 0.999)
	for d := first; d+7 <= days; d++ {
		var s float64
		for k := d; k < d+7; k++ {
			s += load[k]
		}
		ws = append(ws, win{d, s})
	}
	sort.Slice(ws, func(a, b int) bool { return ws[a].sum > ws[b].sum })
	var out []float64
	for _, w := range ws {
		ok := true
		for _, o := range out {
			if abs(float64(w.start)*day-o) < 7*day {
				ok = false
			}
		}
		if ok {
			out = append(out, float64(w.start)*day)
		}
		if len(out) == n {
			break
		}
	}
	sort.Float64s(out)
	return out
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

func runSuite2(args []string) error {
	fs := flag.NewFlagSet("suite2", flag.ExitOnError)
	data := fs.String("data", "data", "data directory")
	out := fs.String("out", "results/suite2", "output directory")
	cluster := fs.String("cluster", "Venus", "Helios cluster")
	variants := fs.String("variants", "Binpack,FGD,gpupack,Binpack+defrag,gpupack+defrag", "comma-separated variants")
	compress := fs.String("compress", "1.0,1.25", "time-compression factors (load growth)")
	windows := fs.Int("windows", 4, "busiest non-overlapping weeks to replay")
	domainSize := fs.Int("domain", 4, "nodes per topology domain; gangs up to this many pods stay in one domain")
	batch := fs.Int("batch", 16, "jobs per scheduling session")
	limit := fs.Duration("time", 500*time.Millisecond, "gpupack solver budget per batch")
	parallel := fs.Int("parallel", 4, "concurrent runs")
	solverDir := fs.String("solver-dir", "solver", "python solver project")
	addr := fs.String("solver", "", "solver address (default: start one locally)")
	fs.Parse(args)

	all, st, err := trace.LoadHelios(filepath.Join(*data, "helios", "data", *cluster, "cluster_log.csv"))
	if err != nil {
		return err
	}
	gpus, err := heliosGPUs(filepath.Join(*data, "helios", "data", *cluster, "cluster_gpu_number.csv"))
	if err != nil {
		return err
	}
	var nodes []*model.NodeRes
	for i := 0; i < gpus/8; i++ {
		n := model.NewNode(fmt.Sprintf("%s-%03d", *cluster, i), model.MaxSpecCPU, 1<<30, 8, "")
		n.Domain = fmt.Sprintf("rack-%02d", i / *domainSize)
		nodes = append(nodes, n)
	}
	var pods []model.Pod
	for _, j := range all {
		pods = append(pods, j.Gang...)
	}
	typical := workload.TypicalPods(pods)
	warm := 3 * day
	starts := busiestWeeks(all, *windows, warm)

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
		c       float64
		w       int
	}
	var mu sync.Mutex
	var runs []run2
	jobs := make(chan job)
	var wg sync.WaitGroup
	for k := 0; k < *parallel; k++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := range jobs {
				t0 := time.Now()
				from, to := starts[j.w]-warm, starts[j.w]+7*day
				var arr []trace.Job
				for _, x := range all {
					if x.Arrive >= from && x.Arrive < to {
						x.Arrive = (x.Arrive - from) / j.c
						arr = append(arr, x)
					}
				}
				cfg := timed.Config{Nodes: nodes, Jobs: arr, Batch: *batch, LocalGangMax: *domainSize,
					MeasureFrom: warm / j.c, DrainLimit: 1}
				name, defrag := strings.CutSuffix(j.variant, "+defrag")
				if name == "gpupack" {
					cfg.Decider = &mip.Placer{Solver: client, Typical: typical, K: 16, TimeLimit: *limit, Deterministic: true, Workers: cpWorkers}
				} else {
					pol, err := sched.New(name, typical)
					if err != nil {
						panic(err)
					}
					cfg.Decider = timed.PolicyDecider{Policy: pol}
				}
				if defrag {
					cfg.Defrag = &timed.DefragConfig{
						Planner:  &mip.Defragmenter{Solver: client, MaxMoves: 4, TimeLimit: time.Second, Deterministic: true, Workers: cpWorkers},
						Interval: 60, CheckpointInterval: 1800, Restart: 60, BenefitHorizon: 3600, Margin: 0.2, Seed: int64(j.w),
					}
				}
				r := timed.Run(cfg)
				q := quantiles(r.GpuPendingLatency)
				res := run2{Variant: j.variant, Compress: j.c, Window: j.w, P50: q(0.5), P95: q(0.95), P99: q(0.99),
					GangP95: quantiles(r.GangPendingLatency)(0.95), JCT: mean(r.JCT) / 3600, Alloc: 100 * r.AllocTimeAvg,
					Pending: r.Unplaced, Migrations: r.Defrag.Migrations, LostGpuH: r.Defrag.LostGpuSec / 3600,
					SolveP99: pctl(r.SolveWall, 0.99), WallS: time.Since(t0).Seconds()}
				mu.Lock()
				runs = append(runs, res)
				fmt.Fprintf(os.Stderr, "  %-16s x%.2f week %d: p95 %.0fs gang p95 %.0fs alloc %.2f%% pending %d moves %d (%.0fs)\n",
					j.variant, j.c, j.w, res.P95, res.GangP95, res.Alloc, res.Pending, res.Migrations, res.WallS)
				mu.Unlock()
			}
		}()
	}
	vs := strings.Split(*variants, ",")
	var cs []float64
	for _, x := range strings.Split(*compress, ",") {
		v, err := strconv.ParseFloat(x, 64)
		if err != nil {
			return err
		}
		cs = append(cs, v)
	}
	for _, c := range cs {
		for _, v := range vs {
			for w := range starts {
				jobs <- job{v, c, w}
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
	var weeks []string
	for _, s := range starts {
		weeks = append(weeks, st.Start.Add(time.Duration(s)*time.Second).Format("2006-01-02"))
	}
	desc := fmt.Sprintf("Helios %s backtest: %d nodes x 8 GPUs, synthetic topology domains of %d nodes (gangs of <= %d pods must stay in one domain). Real submit times and durations; windows = the %d busiest weeks by submitted GPU-hours (starting %s), each after 3 days of warm-up replay. Dropped rows: %d CPU-only, %d uneven gangs, %d zero-duration",
		*cluster, len(nodes), *domainSize, *domainSize, len(starts), strings.Join(weeks, ", "), st.CPUOnly, st.Uneven, st.ZeroDuration)
	report := suite2Report(vs, cs, runs, desc)
	fmt.Print(report)
	return os.WriteFile(filepath.Join(*out, "report.md"), []byte(report), 0o644)
}

func suite2Report(vs []string, cs []float64, runs []run2, desc string) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# Suite 2 — gang training backtest\n\n%s. Mean ± 95%% CI across windows. Wait = submit → bind (s) for GPU jobs arriving in the window.\n\n", desc)
	for _, c := range cs {
		fmt.Fprintf(&b, "## Time compression x%.2f\n\n", c)
		fmt.Fprintf(&b, "| variant | wait p50 s | wait p95 s | wait p99 s | gang wait p95 s | JCT mean h | alloc %% | pending at end | migrations | lost GPU-h | solve p99 ms |\n|---|---|---|---|---|---|---|---|---|---|---|\n")
		for _, v := range vs {
			var rs []run2
			for _, r := range runs {
				if r.Variant == v && r.Compress == c {
					rs = append(rs, r)
				}
			}
			col := func(f func(r run2) float64, prec int) string {
				x := make([]float64, len(rs))
				for i, r := range rs {
					x[i] = f(r)
				}
				m, ci := meanCI(x)
				return fmt.Sprintf("%.*f ± %.*f", prec, m, prec, ci)
			}
			fmt.Fprintf(&b, "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n", v,
				col(func(r run2) float64 { return r.P50 }, 0), col(func(r run2) float64 { return r.P95 }, 0),
				col(func(r run2) float64 { return r.P99 }, 0), col(func(r run2) float64 { return r.GangP95 }, 0),
				col(func(r run2) float64 { return r.JCT }, 2), col(func(r run2) float64 { return r.Alloc }, 2),
				col(func(r run2) float64 { return float64(r.Pending) }, 1), col(func(r run2) float64 { return float64(r.Migrations) }, 1),
				col(func(r run2) float64 { return r.LostGpuH }, 1), col(func(r run2) float64 { return r.SolveP99 }, 0))
		}
		b.WriteString("\n")
	}
	return b.String()
}

func heliosGPUs(path string) (int, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return 0, err
	}
	lines := strings.Split(strings.TrimSpace(string(b)), "\n")
	cols := strings.Split(lines[len(lines)-1], ",")
	return strconv.Atoi(strings.TrimSpace(cols[len(cols)-1]))
}

func quantiles(x []float64) func(float64) float64 {
	s := append([]float64(nil), x...)
	sort.Float64s(s)
	return func(p float64) float64 {
		if len(s) == 0 {
			return 0
		}
		return s[min(len(s)-1, int(p*float64(len(s))))]
	}
}

func mean(x []float64) float64 {
	if len(x) == 0 {
		return 0
	}
	var t float64
	for _, v := range x {
		t += v
	}
	return t / float64(len(x))
}
