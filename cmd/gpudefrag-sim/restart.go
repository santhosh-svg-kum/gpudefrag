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
