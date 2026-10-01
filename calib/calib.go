// Package calib reproduces FGD's suite-1a experiment (arrival-only replay at
// 130% load) and checks our baselines against FGD's published curves.
package calib

import (
	"encoding/csv"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"

	"gpupack/model"
	"gpupack/sched"
	"gpupack/sim"
	"gpupack/workload"
)

// Sample is cumulative arrived GPU demand vs. allocated GPU, both in milli-GPU.
type Sample struct{ Arrived, Used int64 }

// Curve is allocation % indexed by arrived % (0..130); NaN where no sample.
type Curve [131]float64

type RunConfig struct {
	Nodes  []*model.NodeRes
	Pods   []model.Pod
	Policy string
	Seed   int64
	Ratio  float64 // workload inflation, 1.3 in FGD's experiments
}

// Run replays one seeded arrival sequence and samples allocation after every
// pod, including pods that fail to place (FGD's "[Alloc]" log lines).
func Run(cfg RunConfig) ([]Sample, error) {
	pol, err := sched.New(cfg.Policy, workload.TypicalPods(cfg.Pods))
	if err != nil {
		return nil, err
	}
	c := sim.NewCluster(cfg.Nodes)
	rng := rand.New(rand.NewSource(cfg.Seed))
	pods := workload.Prepare(rng, cfg.Pods, c.TotalGpuMilli(), cfg.Ratio)

	out := make([]Sample, 0, len(pods)+1)
	out = append(out, Sample{})
	var arrived int64
	for _, p := range pods {
		arrived += p.Res.TotalMilliGpu()
		sched.Place(c, pol, p, rng)
		out = append(out, Sample{Arrived: arrived, Used: c.UsedGpuMilli()})
	}
	return out, nil
}

func round2(x float64) float64 { return math.RoundToEven(x*100) / 100 }

// Discretize reproduces FGD's merge_alloc_discrete.py.
func Discretize(samples []Sample, totalGpus int64) Curve {
	type pt struct{ arr, alloc float64 }
	pts := make([]pt, len(samples))
	for i, s := range samples {
		pts[i] = pt{
			arr:   math.RoundToEven(float64(s.Arrived) / float64(totalGpus) / 10),
			alloc: round2(float64(s.Used) / float64(totalGpus) / 10),
		}
	}
	mean := func(lo, hi float64) (float64, bool) {
		var sum float64
		n := 0
		for _, p := range pts {
			if p.arr >= lo && p.arr <= hi {
				sum += p.alloc
				n++
			}
		}
		return sum / float64(n), n > 0
	}
	var c Curve
	for k := range c {
		v, ok := mean(float64(k), float64(k))
		if !ok {
			v, ok = mean(float64(k-1), float64(k+1))
		}
		if ok {
			c[k] = round2(v)
		} else {
			c[k] = math.NaN()
		}
	}
	return c
}

// LoadReference reads FGD's analysis_allo_discrete.csv for one workload and
// tune ratio, keyed by policy name (without the "NN-" prefix) and seed.
func LoadReference(path, workloadName, tune string) (map[string]map[int64]Curve, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	recs, err := csv.NewReader(f).ReadAll()
	if err != nil {
		return nil, err
	}
	col := map[string]int{}
	for i, h := range recs[0] {
		col[h] = i
	}
	out := map[string]map[int64]Curve{}
	for _, r := range recs[1:] {
		if r[col["workload"]] != workloadName || r[col["tune"]] != tune {
			continue
		}
		pol := r[col["sc_policy"]]
		if i := strings.Index(pol, "-"); i >= 0 {
			pol = pol[i+1:]
		}
		seed, err := strconv.ParseInt(r[col["seed"]], 10, 64)
		if err != nil {
			return nil, err
		}
		var c Curve
		for k := range c {
			c[k] = math.NaN()
			if i, ok := col[strconv.Itoa(k)]; ok && r[i] != "" {
				if c[k], err = strconv.ParseFloat(r[i], 64); err != nil {
					return nil, err
				}
			}
		}
		if out[pol] == nil {
			out[pol] = map[int64]Curve{}
		}
		out[pol][seed] = c
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("%s: no rows for workload %s tune %s", path, workloadName, tune)
	}
	return out, nil
}

// GatePoints are the arrived-load percentages the calibration gate checks.
var GatePoints = []int{50, 80, 100, 110, 120, 130}

type PolicyGate struct {
	Policy     string    `json:"policy"`
	Ours       []float64 `json:"ours"`
	Ref        []float64 `json:"ref"`
	MaxAbsDiff float64   `json:"max_abs_diff"`
	ExactSeeds int       `json:"exact_seeds"` // seeds whose curve matches ref within 0.05 at every point
	Pass       bool      `json:"pass"`
	Reason     string    `json:"reason,omitempty"`
}

type GateResult struct {
	Points   []int        `json:"points"`
	Tol      float64      `json:"tol_pp"`
	Seeds    []int64      `json:"seeds"`
	Policies []PolicyGate `json:"policies"`
	Pass     bool         `json:"pass"`
}

// Gate compares seed-mean curves. It fails on any missing policy, seed, or
// NaN point rather than passing vacuously.
func Gate(ours, ref map[string]map[int64]Curve, policies []string, seeds []int64, points []int, tol float64) GateResult {
	g := GateResult{Points: points, Tol: tol, Seeds: seeds, Pass: len(policies) > 0 && len(seeds) > 0}
	for _, pol := range policies {
		pg := PolicyGate{Policy: pol, Pass: true}
		fail := func(why string) { pg.Pass, pg.Reason = false, why }
		for _, k := range points {
			var so, sr float64
			for _, s := range seeds {
				o, ok1 := ours[pol][s]
				r, ok2 := ref[pol][s]
				if !ok1 || !ok2 {
					fail(fmt.Sprintf("seed %d missing (ours=%v ref=%v)", s, ok1, ok2))
					continue
				}
				so += o[k]
				sr += r[k]
			}
			mo, mr := so/float64(len(seeds)), sr/float64(len(seeds))
			d := math.Abs(mo - mr)
			if math.IsNaN(d) {
				fail(fmt.Sprintf("NaN at %d%%", k))
			} else if d > tol {
				fail(fmt.Sprintf("|%.2f-%.2f|>%.1f at %d%%", mo, mr, tol, k))
			}
			pg.Ours = append(pg.Ours, round2(mo))
			pg.Ref = append(pg.Ref, round2(mr))
			if !math.IsNaN(d) {
				pg.MaxAbsDiff = max(pg.MaxAbsDiff, round2(d))
			}
		}
		for _, s := range seeds {
			o, ok1 := ours[pol][s]
			r, ok2 := ref[pol][s]
			if !ok1 || !ok2 {
				continue
			}
			exact := true
			for _, k := range points {
				if !(math.Abs(o[k]-r[k]) <= 0.05) {
					exact = false
				}
			}
			if exact {
				pg.ExactSeeds++
			}
		}
		g.Pass = g.Pass && pg.Pass
		g.Policies = append(g.Policies, pg)
	}
	return g
}
