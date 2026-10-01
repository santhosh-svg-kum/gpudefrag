package sim

import (
	"container/heap"
	"fmt"
)

// Event ranks break ties at equal virtual time: freed capacity and new
// arrivals are visible before controllers run in the same instant.
const (
	RankCompletion = iota
	RankArrival
	RankBindApply
	RankSession
	RankDefrag
)

type event struct {
	t    float64
	rank int
	seq  uint64
	fn   func()
}

type eventHeap []event

func (h eventHeap) Len() int { return len(h) }
func (h eventHeap) Less(i, j int) bool {
	a, b := h[i], h[j]
	if a.t != b.t {
		return a.t < b.t
	}
	if a.rank != b.rank {
		return a.rank < b.rank
	}
	return a.seq < b.seq
}
func (h eventHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }
func (h *eventHeap) Push(x any)   { *h = append(*h, x.(event)) }
func (h *eventHeap) Pop() any {
	old := *h
	e := old[len(old)-1]
	*h = old[:len(old)-1]
	return e
}

// Engine is a single-threaded discrete-event loop with a virtual clock (seconds).
// The clock only moves by popping events; handlers may only schedule at or
// after the current time.
type Engine struct {
	h   eventHeap
	now float64
	seq uint64
}

func (e *Engine) Now() float64 { return e.now }
func (e *Engine) Len() int     { return len(e.h) }

func (e *Engine) Push(t float64, rank int, fn func()) {
	if t < e.now {
		panic(fmt.Sprintf("invariant: event scheduled in the past (%.6f < %.6f)", t, e.now))
	}
	e.seq++
	heap.Push(&e.h, event{t, rank, e.seq, fn})
}

// Run processes events until the queue is empty or the next event is after until.
func (e *Engine) Run(until float64) {
	for len(e.h) > 0 && e.h[0].t <= until {
		ev := heap.Pop(&e.h).(event)
		e.now = ev.t
		ev.fn()
	}
}
