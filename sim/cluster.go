// Package sim holds the simulated cluster state and (from M3) the
// discrete-event engine.
package sim

import (
	"fmt"

	"github.com/santhosh-svg-kum/gpudefrag/frag"
	"github.com/santhosh-svg-kum/gpudefrag/model"
)

// Cluster is the mutable state of one simulation run.
type Cluster struct {
	Nodes     []*model.NodeRes
	usedMilli int64
	totalGpus int64
}

// NewCluster deep-copies nodes so concurrent runs never share state.
func NewCluster(nodes []*model.NodeRes) *Cluster {
	c := &Cluster{Nodes: make([]*model.NodeRes, len(nodes))}
	for i, n := range nodes {
		c.Nodes[i] = n.Copy()
		c.totalGpus += int64(n.GpuNum())
		c.usedMilli += int64(n.GpuNum())*model.Milli - n.TotalGpuLeft()
	}
	return c
}

func (c *Cluster) TotalGpus() int64     { return c.totalGpus }
func (c *Cluster) TotalGpuMilli() int64 { return c.totalGpus * model.Milli }
func (c *Cluster) UsedGpuMilli() int64  { return c.usedMilli }

// Fits is the scheduler filter: CPU and memory fit (kube NodeResourcesFit)
// and, for GPU pods, FGD's Open-Gpu-Share filter.
func Fits(n *model.NodeRes, p model.PodRes) bool {
	if p.MilliCPU > n.CPULeft || p.MemMiB > n.MemLeft {
		return false
	}
	if p.GpuMilli == 0 {
		return true
	}
	if n.GpuNum() == 0 || !n.Accessible(p) {
		return false
	}
	return frag.CanHostOnGPU(n, p)
}

// Feasible returns the indices of nodes that pass Fits, in node order.
func (c *Cluster) Feasible(p model.PodRes) []int {
	var out []int
	for i, n := range c.Nodes {
		if Fits(n, p) {
			out = append(out, i)
		}
	}
	return out
}

// AnyFits reports whether at least one node passes Fits.
func (c *Cluster) AnyFits(p model.PodRes) bool {
	for _, n := range c.Nodes {
		if Fits(n, p) {
			return true
		}
	}
	return false
}

// Bind places p on node i using the given GPU device indices. Over-commit is
// an invariant violation and panics.
func (c *Cluster) Bind(i int, p model.PodRes, gpus []int) {
	n := c.Nodes[i]
	if len(gpus) != p.GpuNum {
		panic(fmt.Sprintf("invariant: pod wants %d GPUs, got devices %v", p.GpuNum, gpus))
	}
	n.CPULeft -= p.MilliCPU
	n.MemLeft -= p.MemMiB
	for _, g := range gpus {
		n.GpuLeft[g] -= p.GpuMilli
		if n.GpuLeft[g] < 0 {
			panic(fmt.Sprintf("invariant: GPU %d on %s over-committed", g, n.Name))
		}
	}
	if n.CPULeft < 0 || n.MemLeft < 0 {
		panic(fmt.Sprintf("invariant: node %s over-committed", n.Name))
	}
	if p.GpuNum > 0 {
		n.Affinity[p.Affinity()]++
	}
	c.usedMilli += p.TotalMilliGpu()
}

// Unbind reverses Bind (used by departures and migrations from M3).
func (c *Cluster) Unbind(i int, p model.PodRes, gpus []int) {
	n := c.Nodes[i]
	n.CPULeft += p.MilliCPU
	n.MemLeft += p.MemMiB
	for _, g := range gpus {
		n.GpuLeft[g] += p.GpuMilli
		if n.GpuLeft[g] > model.Milli {
			panic(fmt.Sprintf("invariant: GPU %d on %s freed beyond capacity", g, n.Name))
		}
	}
	if p.GpuNum > 0 {
		tag := p.Affinity()
		if n.Affinity[tag]--; n.Affinity[tag] <= 0 {
			delete(n.Affinity, tag)
		}
	}
	c.usedMilli -= p.TotalMilliGpu()
}
