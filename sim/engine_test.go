package sim

import (
	"reflect"
	"testing"
)

func TestEngineOrdersByTimeRankThenInsertion(t *testing.T) {
	var e Engine
	var got []string
	rec := func(s string) func() { return func() { got = append(got, s) } }
	e.Push(2, RankArrival, rec("arr@2"))
	e.Push(1, RankSession, rec("sess@1"))
	e.Push(1, RankCompletion, rec("done@1"))
	e.Push(1, RankSession, rec("sess@1b"))
	e.Run(10)
	want := []string{"done@1", "sess@1", "sess@1b", "arr@2"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v", got)
	}
	if e.Now() != 2 {
		t.Fatalf("now %v", e.Now())
	}
}

func TestEngineHandlersCanScheduleFollowUps(t *testing.T) {
	var e Engine
	n := 0
	var tick func()
	tick = func() {
		n++
		if n < 3 {
			e.Push(e.Now()+5, RankSession, tick)
		}
	}
	e.Push(0, RankSession, tick)
	e.Run(100)
	if n != 3 || e.Now() != 10 {
		t.Fatalf("n=%d now=%v", n, e.Now())
	}
}

func TestEngineStopsAtUntil(t *testing.T) {
	var e Engine
	ran := false
	e.Push(50, RankArrival, func() { ran = true })
	e.Run(10)
	if ran || e.Len() != 1 {
		t.Fatal("event after horizon must not run")
	}
}

func TestEnginePanicsOnPast(t *testing.T) {
	var e Engine
	e.Push(5, RankArrival, func() { e.Push(4, RankArrival, func() {}) })
	defer func() {
		if recover() == nil {
			t.Fatal("scheduling into the past must panic")
		}
	}()
	e.Run(10)
}
