package mip

import "sort"

// independentGroups partitions pod indices into groups whose candidate-node
// sets are pairwise disjoint and that split no gang. Pods sharing a candidate
// node, or a gang id (non-zero), land in one group. Each group is sorted
// ascending and groups are ordered by their smallest member, so the result is
// deterministic.
func independentGroups(cands []map[int]bool, gang []int) [][]int {
	n := len(cands)
	parent := make([]int, n)
	for i := range parent {
		parent[i] = i
	}
	var find func(int) int
	find = func(x int) int {
		for parent[x] != x {
			parent[x] = parent[parent[x]]
			x = parent[x]
		}
		return x
	}
	union := func(a, b int) {
		if ra, rb := find(a), find(b); ra != rb {
			if ra > rb {
				ra, rb = rb, ra
			}
			parent[rb] = ra
		}
	}
	byNode := map[int]int{}
	byGang := map[int]int{}
	for i := 0; i < n; i++ {
		for node := range cands[i] {
			if j, ok := byNode[node]; ok {
				union(i, j)
			} else {
				byNode[node] = i
			}
		}
		if i < len(gang) && gang[i] != 0 {
			if j, ok := byGang[gang[i]]; ok {
				union(i, j)
			} else {
				byGang[gang[i]] = i
			}
		}
	}
	idx := map[int]int{}
	var groups [][]int
	for i := 0; i < n; i++ {
		r := find(i)
		g, ok := idx[r]
		if !ok {
			g = len(groups)
			idx[r] = g
			groups = append(groups, nil)
		}
		groups[g] = append(groups[g], i)
	}
	for _, g := range groups {
		sort.Ints(g)
	}
	return groups
}
