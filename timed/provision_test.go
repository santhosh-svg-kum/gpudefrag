package timed

import (
	"math"
	"testing"

	"github.com/santhosh-svg-kum/gpudefrag/model"
)

func tmpl(name string) *model.NodeRes { return model.NewNode(name, 64000, 256000, 8, "A100") }

func TestFixedLatencyAppearsAfterLatency(t *testing.T) {
	f := &FixedLatency{Latency: 120, Template: tmpl}
	if !f.Request(10) {
		t.Fatal("request refused")
	}
	if got := f.Ready(129.9); len(got) != 0 {
		t.Fatalf("ready early: %v", got)
	}
	if f.Pending() != 1 || f.Active() != 0 {
		t.Fatalf("pending=%d active=%d", f.Pending(), f.Active())
	}
	got := f.Ready(130)
	if len(got) != 1 || got[0].Name != "prov-1" || len(got[0].GpuLeft) != 8 {
		t.Fatalf("ready = %+v", got)
	}
	if again := f.Ready(500); len(again) != 0 {
		t.Fatalf("node delivered twice: %v", again)
	}
	if f.Pending() != 0 || f.Active() != 1 {
		t.Fatalf("pending=%d active=%d", f.Pending(), f.Active())
	}
}

func TestFixedLatencyCapCountsPendingAndActive(t *testing.T) {
	f := &FixedLatency{Latency: 5, Cap: 2, Template: tmpl}
	if !f.Request(0) || !f.Request(0) {
		t.Fatal("first two requests must succeed")
	}
	if f.Request(0) {
		t.Fatal("third request must hit cap (pending)")
	}
	f.Ready(5)
	if f.Request(6) {
		t.Fatal("cap must count active nodes")
	}
	if !f.Release("prov-1", 7) {
		t.Fatal("release failed")
	}
	if !f.Request(7) {
		t.Fatal("release must free capacity")
	}
}

func TestFixedLatencyReleaseUnknownOrTwice(t *testing.T) {
	f := &FixedLatency{Latency: 1, Template: tmpl}
	f.Request(0)
	if f.Release("prov-1", 0.5) {
		t.Fatal("pending node is not releasable")
	}
	f.Ready(1)
	if f.Release("nope", 2) {
		t.Fatal("unknown name released")
	}
	if !f.Release("prov-1", 2) || f.Release("prov-1", 3) {
		t.Fatal("release must succeed exactly once")
	}
	if f.Active() != 0 {
		t.Fatalf("active=%d", f.Active())
	}
}

func TestFixedLatencyNodeSeconds(t *testing.T) {
	f := &FixedLatency{Latency: 10, Template: tmpl}
	f.Request(0)  // prov-1: billed from 0
	f.Request(20) // prov-2: billed from 20
	f.Ready(10)
	f.Release("prov-1", 50) // 50 s closed
	// at now=60: prov-1 closed 50, prov-2 pending/active since 20 -> 40
	if got := f.NodeSeconds(60); math.Abs(got-90) > 1e-9 {
		t.Fatalf("NodeSeconds = %v, want 90", got)
	}
}
