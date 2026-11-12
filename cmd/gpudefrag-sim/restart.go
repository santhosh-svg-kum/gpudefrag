package main

import (
	"fmt"
	"strconv"
	"strings"
)

const restartTag = "@r"

// restartVariants expands each "+defrag" variant into one per restart time in
// sweep ("FGD+defrag@r300"); variants without defrag, and an empty sweep, are
// returned unchanged.
func restartVariants(vs []string, sweep string) ([]string, error) {
	if sweep == "" {
		return vs, nil
	}
	var rs []string
	for _, s := range strings.Split(sweep, ",") {
		r, err := strconv.ParseFloat(strings.TrimSpace(s), 64)
		if err != nil || r < 0 {
			return nil, fmt.Errorf("bad -restart-sweep value %q", s)
		}
		rs = append(rs, strconv.FormatFloat(r, 'f', -1, 64))
	}
	var out []string
	for _, v := range vs {
		if !strings.HasSuffix(v, "+defrag") {
			out = append(out, v)
			continue
		}
		for _, r := range rs {
			out = append(out, v+restartTag+r)
		}
	}
	return out, nil
}

// splitRestart separates a variant from its restart tag, returning def when
// the variant carries none.
func splitRestart(v string, def float64) (string, float64) {
	i := strings.LastIndex(v, restartTag)
	if i < 0 {
		return v, def
	}
	r, err := strconv.ParseFloat(v[i+len(restartTag):], 64)
	if err != nil {
		return v, def
	}
	return v[:i], r
}

const graceTag = "@g"

func parseSweep(sweep, flagName string) ([]string, error) {
	if sweep == "" {
		return nil, nil
	}
	var out []string
	for _, s := range strings.Split(sweep, ",") {
		x, err := strconv.ParseFloat(strings.TrimSpace(s), 64)
		if err != nil || x < 0 {
			return nil, fmt.Errorf("bad %s value %q", flagName, s)
		}
		out = append(out, strconv.FormatFloat(x, 'f', -1, 64))
	}
	return out, nil
}

// gridVariants expands each "+defrag" variant over the restart x grace grid
// ("FGD+defrag@r300@g30"). An empty sweep contributes no tag for its axis;
// variants without defrag are returned unchanged.
func gridVariants(vs []string, restartSweep, graceSweep string) ([]string, error) {
	rs, err := parseSweep(restartSweep, "-restart-sweep")
	if err != nil {
		return nil, err
	}
	gs, err := parseSweep(graceSweep, "-grace-sweep")
	if err != nil {
		return nil, err
	}
	if rs == nil {
		rs = []string{""}
	}
	if gs == nil {
		gs = []string{""}
	}
	var out []string
	for _, v := range vs {
		if !strings.HasSuffix(v, "+defrag") {
			out = append(out, v)
			continue
		}
		for _, r := range rs {
			for _, g := range gs {
				t := v
				if r != "" {
					t += restartTag + r
				}
				if g != "" {
					t += graceTag + g
				}
				out = append(out, t)
			}
		}
	}
	return out, nil
}

// splitVariant strips restart and grace tags from a variant, returning the
// defaults for tags it does not carry.
func splitVariant(v string, defRestart, defGrace float64) (base string, restart, grace float64) {
	restart, grace = defRestart, defGrace
	base = v
	if i := strings.LastIndex(base, graceTag); i >= 0 {
		if g, err := strconv.ParseFloat(base[i+len(graceTag):], 64); err == nil {
			grace, base = g, base[:i]
		}
	}
	base, restart = splitRestart(base, defRestart)
	return
}
