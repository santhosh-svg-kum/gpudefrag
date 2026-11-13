package main

import (
	"fmt"
	"sort"
	"strconv"
)

// sweepPoint is one cell of a restart sweep: mean allocation of a defrag
// variant at the given restart time (seconds).
type sweepPoint struct {
	Restart float64
	Mean    float64
}

// breakEven returns the smallest restart time at which defrag stops beating
// the no-defrag baseline (mean <= base), interpolating linearly between the
// last winning point and the first non-winning one. ok is false when defrag
// wins at every swept restart time (or there are no points).
func breakEven(pts []sweepPoint, base float64) (float64, bool) {
	ps := append([]sweepPoint(nil), pts...)
	sort.Slice(ps, func(i, j int) bool { return ps[i].Restart < ps[j].Restart })
	for i, p := range ps {
		if p.Mean > base {
			continue
		}
		if i == 0 {
			return p.Restart, true
		}
		q := ps[i-1]
		return q.Restart + (p.Restart-q.Restart)*(q.Mean-base)/(q.Mean-p.Mean), true
	}
	return 0, false
}

func formatBreakEven(r float64, ok bool) string {
	if !ok {
		return "n/a"
	}
	return fmt.Sprintf("%s s", strconv.FormatFloat(r, 'f', -1, 64))
}
