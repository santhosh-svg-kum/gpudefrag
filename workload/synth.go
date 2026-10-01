package workload

import (
	"fmt"
	"math"
	"math/rand"
	"sort"

	"gpupack/model"
	"gpupack/trace"
)

// Synthesize generates n jobs for a timed run (design doc §10.1, parametric
// generation): (pod, duration) pairs are bootstrapped from base, durations
// are capped at maxDur, and Poisson arrivals are scaled so the offered GPU
// load is rho x capacity. Jobs with non-positive duration are dropped.
func Synthesize(rng *rand.Rand, base []trace.Job, capacityMilli int64, rho float64, n int, maxDur float64) []trace.Job {
	pool := durationPool(base, maxDur)
	rate := ArrivalRate(base, capacityMilli, rho, maxDur)

	out := make([]trace.Job, n)
	var t float64
	for i := range out {
		t += rng.ExpFloat64() / rate
		j := pool[rng.Intn(len(pool))]
		j.Arrive = t
		j.Pod.Name = fmt.Sprintf("%s-%d", j.Pod.Name, i)
		out[i] = j
	}
	sort.SliceStable(out, func(a, b int) bool { return out[a].Arrive < out[b].Arrive })
	return out
}

func durationPool(base []trace.Job, maxDur float64) []trace.Job {
	var pool []trace.Job
	for _, j := range base {
		if j.Duration > 0 {
			j.Duration = min(j.Duration, maxDur)
			pool = append(pool, j)
		}
	}
	return pool
}

// ArrivalRate is the Poisson rate (per second) giving offered GPU load rho.
func ArrivalRate(base []trace.Job, capacityMilli int64, rho, maxDur float64) float64 {
	pool := durationPool(base, maxDur)
	var work float64
	for _, j := range pool {
		work += float64(j.Pod.Res.TotalMilliGpu()) * j.Duration
	}
	return rho * float64(capacityMilli) / (work / float64(len(pool)))
}

// SubsetNodes keeps round(frac x count) nodes of every (GPU type, GPU count)
// group, so a smaller cluster keeps the fleet's hardware mix.
func SubsetNodes(rng *rand.Rand, nodes []*model.NodeRes, frac float64) []*model.NodeRes {
	groups := map[string][]*model.NodeRes{}
	var keys []string
	for _, n := range nodes {
		k := fmt.Sprintf("%s/%d", n.GpuType, n.GpuNum())
		if _, ok := groups[k]; !ok {
			keys = append(keys, k)
		}
		groups[k] = append(groups[k], n)
	}
	sort.Strings(keys)
	var out []*model.NodeRes
	for _, k := range keys {
		g := append([]*model.NodeRes(nil), groups[k]...)
		sort.Slice(g, func(a, b int) bool { return g[a].Name < g[b].Name })
		rng.Shuffle(len(g), func(a, b int) { g[a], g[b] = g[b], g[a] })
		out = append(out, g[:int(math.Round(frac*float64(len(g))))]...)
	}
	sort.Slice(out, func(a, b int) bool { return out[a].Name < out[b].Name })
	return out
}
