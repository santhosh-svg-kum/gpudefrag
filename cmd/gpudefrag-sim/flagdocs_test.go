package main

import (
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"testing"
)

// TestFlagsDocumented fails when a flag registered by a subcommand is not
// mentioned (as "-name") in README.md or docs/RESULTS.md.
func TestFlagsDocumented(t *testing.T) {
	var docs strings.Builder
	var readme string
	for _, p := range []string{"../../README.md", "../../docs/RESULTS.md"} {
		b, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		docs.Write(b)
		if strings.HasSuffix(p, "README.md") {
			readme = string(b)
		}
	}
	files, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	re := regexp.MustCompile(`\bfs\.(?:String|Int|Int64|Bool|Float64|Duration|Uint)\(\s*"([^"]+)"`)
	seen := map[string]bool{}
	for _, f := range files {
		if strings.HasSuffix(f, "_test.go") {
			continue
		}
		b, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		for _, m := range re.FindAllStringSubmatch(string(b), -1) {
			seen[m[1]] = true
		}
	}
	if len(seen) < 10 {
		t.Fatalf("flag scan found only %d flags; regexp is stale", len(seen))
	}
	for name := range seen {
		pat := regexp.MustCompile("(^|[^A-Za-z0-9-])-{1,2}" + regexp.QuoteMeta(name) + `([^A-Za-z0-9-]|$)`)
		if !pat.MatchString(docs.String()) {
			t.Errorf("flag -%s is not documented in README.md or docs/RESULTS.md", name)
		}
	}
	for _, name := range staleFlags(readme, seen) {
		t.Errorf("README.md documents flag -%s but no subcommand registers it", name)
	}
}

// staleFlags returns the backticked "-flag" mentions in doc that are not in
// registered. Only backticked tokens that start with a lowercase letter are
// considered, so prose hyphens and negative numbers are ignored.
func staleFlags(doc string, registered map[string]bool) []string {
	re := regexp.MustCompile("`-{1,2}([a-z][a-z0-9-]*)")
	seen := map[string]bool{}
	var out []string
	for _, m := range re.FindAllStringSubmatch(doc, -1) {
		if !registered[m[1]] && !seen[m[1]] {
			seen[m[1]] = true
			out = append(out, m[1])
		}
	}
	sort.Strings(out)
	return out
}

func TestStaleFlags(t *testing.T) {
	reg := map[string]bool{"grace": true, "restart": true}
	got := staleFlags("use `-grace 30` and `-old-flag`, not `-restart`; `-old-flag` again", reg)
	if len(got) != 1 || got[0] != "old-flag" {
		t.Fatalf("got %v, want [old-flag]", got)
	}
	if got := staleFlags("`-grace=1` -unticked", reg); len(got) != 0 {
		t.Fatalf("got %v, want none", got)
	}
}
