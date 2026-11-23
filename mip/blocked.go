package mip

import (
	"github.com/santhosh-svg-kum/gpudefrag/model"
	"github.com/santhosh-svg-kum/gpudefrag/sim"
)

// blockedCandidates returns, per blocked pod, the nodes that could host it if
// their free GPU capacity were consolidated: CPU, memory and GPU type fit and
// the total free milli-GPU covers the request. Unlike sim.Fits this ignores
// per-device fragmentation, which is what defrag or provisioning would fix.
func blockedCandidates(c *sim.Cluster, pods []model.PodRes) []map[int]bool {
	cands := make([]map[int]bool, len(pods))
	for i, p := range pods {
		cands[i] = map[int]bool{}
		for j, n := range c.Nodes {
			if p.MilliCPU > n.CPULeft || p.MemMiB > n.MemLeft {
				continue
			}
			if p.GpuMilli > 0 && (n.GpuNum() == 0 || !n.Accessible(p) || n.TotalGpuLeft() < p.TotalMilliGpu()) {
				continue
			}
			cands[i][j] = true
		}
	}
	return cands
}

// GroupBlocked groups blocked pods (indices into pods) that share a candidate
// node or a non-zero gang id, so each group can be handled as one defrag or
// provisioning decision. It reuses independentGroups; ordering is
// deterministic.
func GroupBlocked(c *sim.Cluster, pods []model.PodRes, gang []int) [][]int {
	return independentGroups(blockedCandidates(c, pods), gang)
}
