package timed

import (
	"fmt"
	"sort"

	"github.com/santhosh-svg-kum/gpudefrag/model"
)

// Provisioner is the node-lifecycle seam for the elastic mode sketched in
// docs/design/notes/2026-11-17-elastic-mode.md. Times are virtual seconds
// and must not go backwards. No suite uses it yet.
type Provisioner interface {
	// Request asks for one node. It reports false when the cap is reached.
	Request(now float64) (ok bool)
	// Ready returns, once each, the nodes whose latency has elapsed by now.
	Ready(now float64) []*model.NodeRes
	// Release returns a ready node and stops its billing. It reports false
	// for a name that is not an active provisioned node.
	Release(name string, now float64) bool
	// Pending is the number of requested nodes not yet ready.
	Pending() int
	// Active is the number of ready, unreleased nodes.
	Active() int
	// NodeSeconds is the billed time so far, counted from the request.
	NodeSeconds(now float64) float64
}

// FixedLatency is a fake Provisioner: every node appears Latency seconds
// after its request and comes from Template, which must return a fresh node
// per call. Cap bounds pending + active nodes (0 = unlimited).
type FixedLatency struct {
	Latency  float64
	Cap      int
	Template func(name string) *model.NodeRes

	seq     int
	pending []pendingNode
	active  map[string]float64 // name -> request time
	billed  float64            // closed (released) node-seconds
}

type pendingNode struct {
	name    string
	reqAt   float64
	readyAt float64
}

func (f *FixedLatency) Request(now float64) bool {
	if f.Cap > 0 && len(f.pending)+len(f.active) >= f.Cap {
		return false
	}
	f.seq++
	f.pending = append(f.pending, pendingNode{
		name: fmt.Sprintf("prov-%d", f.seq), reqAt: now, readyAt: now + f.Latency,
	})
	return true
}

func (f *FixedLatency) Ready(now float64) []*model.NodeRes {
	var out []*model.NodeRes
	keep := f.pending[:0]
	for _, p := range f.pending {
		if p.readyAt > now {
			keep = append(keep, p)
			continue
		}
		if f.active == nil {
			f.active = map[string]float64{}
		}
		f.active[p.name] = p.reqAt
		out = append(out, f.Template(p.name))
	}
	f.pending = keep
	return out
}

func (f *FixedLatency) Release(name string, now float64) bool {
	reqAt, ok := f.active[name]
	if !ok {
		return false
	}
	delete(f.active, name)
	f.billed += now - reqAt
	return true
}

func (f *FixedLatency) Pending() int { return len(f.pending) }
func (f *FixedLatency) Active() int  { return len(f.active) }

func (f *FixedLatency) NodeSeconds(now float64) float64 {
	total := f.billed
	names := make([]string, 0, len(f.active))
	for n := range f.active {
		names = append(names, n)
	}
	sort.Strings(names) // fixed summation order: deterministic floats
	for _, n := range names {
		total += now - f.active[n]
	}
	for _, p := range f.pending {
		total += now - p.reqAt
	}
	return total
}

var _ Provisioner = (*FixedLatency)(nil)
