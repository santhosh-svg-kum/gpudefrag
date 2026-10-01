package workload

import (
	"fmt"
	"math/rand"
	"sort"

	"gpupack/trace"
)

// Synthesize generates n jobs for a timed run (design doc §10.1, parametric
// generation): (pod, duration) pairs are bootstrapped from base, durations
// are capped at maxDur, and Poisson arrivals are scaled so the offered GPU
// load is rho x capacity. Jobs with non-positive duration are dropped.
func Synthesize(rng *rand.Rand, base []trace.Job, capacityMilli int64, rho float64, n int, maxDur float64) []trace.Job {
	var pool []trace.Job
	var work float64
	for _, j := range base {
		if j.Duration <= 0 {
			continue
		}
		j.Duration = min(j.Duration, maxDur)
		pool = append(pool, j)
		work += float64(j.Pod.Res.TotalMilliGpu()) * j.Duration
	}
	meanWork := work / float64(len(pool))
	rate := rho * float64(capacityMilli) / meanWork // arrivals per second

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
