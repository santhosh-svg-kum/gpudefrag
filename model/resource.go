// Package model holds the resource types shared by the simulator, the
// baselines and the solver client. Units are integers: milli-CPU, MiB and
// milli-GPU (1 GPU = Milli).
package model

import (
	"fmt"
	"sort"
	"strings"
)

const (
	Milli      = 1000   // milli-GPU per GPU
	MaxSpecCPU = 128000 // FGD normalisation constant (milli-CPU)
	MaxSpecGPU = 8000   // FGD normalisation constant (milli-GPU)
)

// PodRes is a pod's request. GpuMilli is per GPU: a share pod has GpuNum 1
// and GpuMilli < Milli; a whole-GPU pod has GpuMilli == Milli.
type PodRes struct {
	MilliCPU int64
	MemMiB   int64
	GpuNum   int
	GpuMilli int64
	GpuType  string // "" = any; "A|B" = one of
}

type Pod struct {
	Name string
	Res  PodRes
}

func (p PodRes) TotalMilliGpu() int64 { return p.GpuMilli * int64(p.GpuNum) }

func (p PodRes) IsShare() bool { return p.GpuNum == 1 && p.GpuMilli < Milli }

// Affinity is FGD's GPU-affinity tag, used by the GpuClustering policy.
func (p PodRes) Affinity() string {
	switch {
	case p.GpuNum == 0:
		return "no-gpu"
	case p.IsShare():
		return "share-gpu"
	default:
		return fmt.Sprintf("%d-gpu", p.GpuNum)
	}
}

// Less is FGD's PodResource ordering. Memory is not part of it.
func (p PodRes) Less(o PodRes) bool {
	switch {
	case p.MilliCPU != o.MilliCPU:
		return p.MilliCPU < o.MilliCPU
	case p.GpuMilli != o.GpuMilli:
		return p.GpuMilli < o.GpuMilli
	case p.GpuNum != o.GpuNum:
		return p.GpuNum < o.GpuNum
	default:
		return p.GpuType < o.GpuType
	}
}

// NodeRes is a node's capacity and what is left of it.
type NodeRes struct {
	Name     string
	CPUCap   int64
	CPULeft  int64
	MemCap   int64
	MemLeft  int64
	GpuType  string
	GpuLeft  []int64        // milli-GPU left per device
	Affinity map[string]int // count of GPU pods per affinity tag (zero counts are deleted)
}

func NewNode(name string, cpu, mem int64, gpus int, gpuType string) *NodeRes {
	left := make([]int64, gpus)
	for i := range left {
		left[i] = Milli
	}
	return &NodeRes{Name: name, CPUCap: cpu, CPULeft: cpu, MemCap: mem, MemLeft: mem,
		GpuType: gpuType, GpuLeft: left, Affinity: map[string]int{}}
}

func (n *NodeRes) GpuNum() int { return len(n.GpuLeft) }

func (n *NodeRes) TotalGpuLeft() (t int64) {
	for _, v := range n.GpuLeft {
		t += v
	}
	return t
}

func (n *NodeRes) FullyFree() (c int) {
	for _, v := range n.GpuLeft {
		if v == Milli {
			c++
		}
	}
	return c
}

// SortedGpuIdx returns device indices ordered by milli left (stable).
func (n *NodeRes) SortedGpuIdx(asc bool) []int {
	idx := make([]int, len(n.GpuLeft))
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(i, j int) bool {
		a, b := n.GpuLeft[idx[i]], n.GpuLeft[idx[j]]
		if asc {
			return a < b
		}
		return a > b
	})
	return idx
}

func (n *NodeRes) Copy() *NodeRes {
	c := *n
	c.GpuLeft = append([]int64(nil), n.GpuLeft...)
	c.Affinity = make(map[string]int, len(n.Affinity))
	for k, v := range n.Affinity {
		c.Affinity[k] = v
	}
	return &c
}

// Sub returns a copy with p subtracted, packing GPU requests into the devices
// with the least milli left first (FGD's NodeResource.Sub). Memory is not
// touched, matching FGD's fragmentation model.
func (n *NodeRes) Sub(p PodRes) (*NodeRes, error) {
	out := n.Copy()
	if out.CPULeft < p.MilliCPU || out.GpuNum() < p.GpuNum {
		return out, fmt.Errorf("node %s cannot fit %+v", n.Name, p)
	}
	out.CPULeft -= p.MilliCPU
	req := p.GpuNum
	if req == 0 {
		return out, nil
	}
	for _, i := range out.SortedGpuIdx(true) {
		if p.GpuMilli <= out.GpuLeft[i] {
			req--
			out.GpuLeft[i] -= p.GpuMilli
			if req <= 0 {
				return out, nil
			}
		}
	}
	return out, fmt.Errorf("node %s cannot fit %+v (%d GPUs short)", n.Name, p, req)
}

// Accessible reports whether the node's GPU type satisfies the pod's
// GPU-type constraint.
func (n *NodeRes) Accessible(p PodRes) bool {
	if p.GpuType == "" {
		return true
	}
	if n.GpuType == "" {
		return false
	}
	cnt := 0
	for _, t := range strings.Split(p.GpuType, "|") {
		if t == "" {
			continue
		}
		cnt++
		if t == n.GpuType {
			return true
		}
	}
	return cnt == 0
}
