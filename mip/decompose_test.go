package mip

import (
	"reflect"
	"testing"
)

func set(ns ...int) map[int]bool {
	m := map[int]bool{}
	for _, n := range ns {
		m[n] = true
	}
	return m
}

func TestIndependentGroups(t *testing.T) {
	cases := []struct {
		name  string
		cands []map[int]bool
		gang  []int
		want  [][]int
	}{
		{"empty", nil, nil, nil},
		{"disjoint", []map[int]bool{set(1, 2), set(3), set(4, 5)}, []int{0, 0, 0}, [][]int{{0}, {1}, {2}}},
		{"shared node", []map[int]bool{set(1, 2), set(2, 3), set(9)}, []int{0, 0, 0}, [][]int{{0, 1}, {2}}},
		{"chained", []map[int]bool{set(1, 2), set(5), set(2, 3), set(3, 5)}, []int{0, 0, 0, 0}, [][]int{{0, 1, 2, 3}}},
		{"gang links disjoint pods", []map[int]bool{set(1), set(7), set(9), set(1, 4)}, []int{3, 0, 3, 0}, [][]int{{0, 2, 3}, {1}}},
		{"empty cand set stays alone", []map[int]bool{set(), set(), set(1)}, []int{0, 0, 0}, [][]int{{0}, {1}, {2}}},
	}
	for _, tc := range cases {
		got := independentGroups(tc.cands, tc.gang)
		if !reflect.DeepEqual(got, tc.want) {
			t.Errorf("%s: got %v want %v", tc.name, got, tc.want)
		}
	}
}
