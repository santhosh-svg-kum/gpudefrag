package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"gpupack/calib"
	"gpupack/sched"
	"gpupack/trace"
)

func runCalib(args []string) error {
	fs := flag.NewFlagSet("calib", flag.ExitOnError)
	data := fs.String("data", "data", "directory populated by scripts/fetch-data.sh")
	out := fs.String("out", "results/calib", "output directory")
	policies := fs.String("policies", strings.Join(sched.Names, ","), "comma-separated policies")
	seedLo := fs.Int64("seed-from", 42, "first seed")
	seedHi := fs.Int64("seed-to", 51, "last seed")
	tol := fs.Float64("tol", 1.0, "gate tolerance in percentage points")
	workers := fs.Int("workers", runtime.NumCPU(), "parallel runs")
	fs.Parse(args)

	nodes, err := trace.LoadOpenbNodes(filepath.Join(*data, "openb", "openb_node_list_gpu_node.csv"))
	if err != nil {
		return err
	}
	pods, err := trace.LoadOpenbPods(filepath.Join(*data, "openb", "openb_pod_list_default.csv"))
	if err != nil {
		return err
	}
	ref, err := calib.LoadReference(filepath.Join(*data, "ref", "analysis_allo_discrete.csv"), "openb_pod_list_default", "1.3")
	if err != nil {
		return err
	}
	var totalGpus int64
	for _, n := range nodes {
		totalGpus += int64(n.GpuNum())
	}

	pols := strings.Split(*policies, ",")
	var seeds []int64
	for s := *seedLo; s <= *seedHi; s++ {
		seeds = append(seeds, s)
	}

	type job struct {
		pol  string
		seed int64
	}
	jobs := make(chan job)
	var mu sync.Mutex
	var errs []error
	ours := map[string]map[int64]calib.Curve{}
	var wg sync.WaitGroup
	start := time.Now()
	for w := 0; w < *workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := range jobs {
				t0 := time.Now()
				s, err := calib.Run(calib.RunConfig{Nodes: nodes, Pods: pods, Policy: j.pol, Seed: j.seed, Ratio: 1.3})
				mu.Lock()
				if err != nil {
					errs = append(errs, err)
				} else {
					if ours[j.pol] == nil {
						ours[j.pol] = map[int64]calib.Curve{}
					}
					ours[j.pol][j.seed] = calib.Discretize(s, totalGpus)
					fmt.Fprintf(os.Stderr, "  %-13s seed %d done in %s\n", j.pol, j.seed, time.Since(t0).Round(time.Millisecond))
				}
				mu.Unlock()
			}
		}()
	}
	for _, p := range pols {
		for _, s := range seeds {
			jobs <- job{p, s}
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
	if err := writeCurves(filepath.Join(*out, "curves.csv"), ours, pols, seeds); err != nil {
		return err
	}
	g := calib.Gate(ours, ref, pols, seeds, calib.GatePoints, *tol)
	b, _ := json.MarshalIndent(g, "", "  ")
	if err := os.WriteFile(filepath.Join(*out, "gate.json"), b, 0o644); err != nil {
		return err
	}

	fmt.Printf("calibration vs FGD reference (openb default, tune 1.3, seeds %d-%d, %s)\n", *seedLo, *seedHi, time.Since(start).Round(time.Second))
	fmt.Printf("%-14s %-44s %-44s %6s %5s %s\n", "policy", "ours @ "+fmtInts(calib.GatePoints), "ref", "maxΔ", "exact", "")
	for _, p := range g.Policies {
		status := "PASS"
		if !p.Pass {
			status = "FAIL " + p.Reason
		}
		fmt.Printf("%-14s %-44s %-44s %6.2f %2d/%-2d %s\n", p.Policy, fmtFloats(p.Ours), fmtFloats(p.Ref), p.MaxAbsDiff, p.ExactSeeds, len(seeds), status)
	}
	if !g.Pass {
		return fmt.Errorf("calibration gate FAILED (tolerance %.1f pp)", *tol)
	}
	fmt.Println("calibration gate PASSED")
	return nil
}

func writeCurves(path string, ours map[string]map[int64]calib.Curve, pols []string, seeds []int64) error {
	var b strings.Builder
	b.WriteString("policy,seed")
	for k := 0; k <= 130; k++ {
		fmt.Fprintf(&b, ",%d", k)
	}
	b.WriteString("\n")
	for _, p := range pols {
		for _, s := range seeds {
			fmt.Fprintf(&b, "%s,%d", p, s)
			c := ours[p][s]
			for _, v := range c {
				b.WriteString("," + strconv.FormatFloat(v, 'f', 2, 64))
			}
			b.WriteString("\n")
		}
	}
	return os.WriteFile(path, []byte(b.String()), 0o644)
}

func fmtFloats(v []float64) string {
	s := make([]string, len(v))
	for i, x := range v {
		s[i] = strconv.FormatFloat(x, 'f', 2, 64)
	}
	return strings.Join(s, " ")
}

func fmtInts(v []int) string {
	s := make([]string, len(v))
	for i, x := range v {
		s[i] = strconv.Itoa(x)
	}
	return strings.Join(s, ",")
}
