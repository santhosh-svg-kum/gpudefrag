// Package frag implements FGD's GPU fragmentation measure (Weng et al.,
// USENIX ATC '23), ported from kubernetes-scheduler-simulator
// pkg/utils/frag.go (Apache-2.0).
//
// A node's fragmentation is the expected amount of its free GPU that a task
// drawn from the "typical pods" distribution could not use.
package frag

import "gpupack/model"

// Kind classifies a (node, typical pod) pair, as in FGD's FragRatioDataMap.
type Kind int

const (
	Q1LackBoth Kind = iota
	Q2LackGPU
	Q3Satisfied
	Q4LackCPU
	XLSatisfied
	XRLackCPU
	NoAccess
	numKinds
)

// TargetPod is one class of the typical-pod distribution.
type TargetPod struct {
	Res model.PodRes
	Pct float64 // share of the distribution, 0..1
}

// Amount is fragmentation in milli-GPU, split by Kind.
type Amount [numKinds]float64

func (a Amount) SumExceptQ3() (s float64) {
	for k, v := range a {
		if Kind(k) != Q3Satisfied {
			s += v
		}
	}
	return s
}

// NodePodKind is FGD's GetNodePodFrag.
func NodePodKind(n *model.NodeRes, p model.PodRes) Kind {
	cpuOK := n.CPULeft >= p.MilliCPU
	if p.GpuMilli == 0 {
		if cpuOK {
			return XLSatisfied
		}
		return XRLackCPU
	}
	if !n.Accessible(p) {
		return NoAccess
	}
	switch gpuOK := CanHostOnGPU(n, p); {
	case gpuOK && cpuOK:
		return Q3Satisfied
	case gpuOK:
		return Q4LackCPU
	case cpuOK:
		return Q2LackGPU
	default:
		return Q1LackBoth
	}
}

// CanHostOnGPU reports whether GpuNum devices each have at least GpuMilli left.
func CanHostOnGPU(n *model.NodeRes, p model.PodRes) bool {
	req := p.GpuNum
	for _, left := range n.GpuLeft {
		if left >= p.GpuMilli {
			req--
			if req <= 0 {
				return true
			}
		}
	}
	return false
}

// GpuFragMilli is the free milli-GPU on devices too small for p's per-GPU request.
func GpuFragMilli(n *model.NodeRes, p model.PodRes) (m int64) {
	for _, left := range n.GpuLeft {
		if left < p.GpuMilli {
			m += left
		}
	}
	return m
}

// NodeAmount is FGD's NodeGpuShareFragAmount.
func NodeAmount(n *model.NodeRes, typical []TargetPod) Amount {
	var a Amount
	total := float64(n.TotalGpuLeft())
	for _, tp := range typical {
		k := NodePodKind(n, tp.Res)
		if k == Q3Satisfied {
			fm := float64(GpuFragMilli(n, tp.Res))
			a[Q2LackGPU] += tp.Pct * fm
			a[Q3Satisfied] += tp.Pct * (total - fm)
		} else {
			a[k] += tp.Pct * total
		}
	}
	return a
}

// NodeScore is FGD's NodeGpuShareFragAmountScore: fragmented milli-GPU on n.
func NodeScore(n *model.NodeRes, typical []TargetPod) float64 {
	return NodeAmount(n, typical).SumExceptQ3()
}
