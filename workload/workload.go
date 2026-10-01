// Package workload turns a trace into the submission sequence used by FGD's
// experiments, and derives the typical-pod distribution. Ported from
// kubernetes-scheduler-simulator pkg/simulator (SortClusterPods,
// TunePodsByNodeTotalResource) and pkg/utils/frag.go (GetTypicalPods),
// Apache-2.0.
package workload

import (
	"fmt"
	"math/rand"
	"sort"

	"github.com/santhosh-svg-kum/gpudefrag/frag"
	"github.com/santhosh-svg-kum/gpudefrag/model"
)

// Prepare returns the pod submission order for one seeded run: pods sorted by
// name, shuffled, then tuned so total GPU demand reaches ratio x capacity
// (ratio <= 0 disables tuning). rng must be freshly seeded; it is left
// positioned exactly where FGD's global rand would be, so later random
// policy decisions draw from the same stream.
func Prepare(rng *rand.Rand, pods []model.Pod, capacityMilli int64, ratio float64) []model.Pod {
	_ = rng.Int() // FGD draws once to log the seed.

	sorted := append([]model.Pod(nil), pods...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].Name < sorted[j].Name })
	out := append([]model.Pod(nil), sorted...)
	rng.Shuffle(len(out), func(i, j int) { out[i], out[j] = out[j], out[i] })

	if ratio <= 0 {
		return out
	}
	var total int64
	for _, p := range out {
		total += p.Res.TotalMilliGpu()
	}
	target := ratio * float64(capacityMilli)
	switch {
	case float64(total) > target:
		for float64(total) > target && len(out) > 0 {
			i := rng.Intn(len(out))
			total -= out[i].Res.TotalMilliGpu()
			out = append(out[:i], out[i+1:]...)
		}
	case float64(total) < target:
		for i := 0; ; i++ {
			p := sorted[rng.Intn(len(sorted))]
			// FGD compares using the per-GPU milli, not the total; kept for fidelity.
			if float64(total+p.Res.GpuMilli) > target {
				break
			}
			p.Name = fmt.Sprintf("%s-tuned-%d", p.Name, i)
			total += p.Res.TotalMilliGpu()
			out = append(out, p)
		}
	}
	return out
}

const popularityThreshold = 95 // percent of pods covered by the typical classes

// TypicalPods is FGD's GetTypicalPods with its experiment defaults
// (CPU pods included, 95% popularity threshold, step 1, no GPU weighting).
func TypicalPods(pods []model.Pod) []frag.TargetPod { return TypicalPodsWeighted(pods, 0) }

// TypicalPodsWeighted is GetTypicalPods with FGD's gpuResWeight option: a
// whole-GPU pod counts 1 + gpus x weight, so classes that hold many GPUs
// matter in proportion to the capacity they need, not just their count.
func TypicalPodsWeighted(pods []model.Pod, gpuWeight float64) []frag.TargetPod {
	counts := map[model.PodRes]float64{}
	var total float64
	for _, p := range pods {
		k := p.Res
		k.MemMiB = 0
		w := 1.0
		if gpuWeight > 0 && k.GpuMilli == model.Milli {
			w = 1 + float64(k.GpuNum)*gpuWeight
		}
		counts[k] += w
		total += w
	}
	list := make([]frag.TargetPod, 0, len(counts))
	for k, c := range counts {
		list = append(list, frag.TargetPod{Res: k, Pct: c})
	}
	sort.Slice(list, func(i, j int) bool {
		if list[i].Pct != list[j].Pct {
			return list[i].Pct > list[j].Pct
		}
		return list[j].Res.Less(list[i].Res)
	})

	expected := popularityThreshold * total / 100
	var cum float64
	i, num := 0, 0
	for cum < expected {
		num++
		for i < num && i < len(list) {
			cum += list[i].Pct
			list[i].Pct /= total
			i++
		}
	}
	if i >= len(list) {
		return list
	}
	out := list[:i]
	for j := range out {
		out[j].Pct /= cum / total
	}
	return out
}
