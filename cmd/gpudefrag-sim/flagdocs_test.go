package main

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// TestFlagsDocumented fails when a flag registered by a subcommand is not
// mentioned (as "-name") in README.md or docs/RESULTS.md.
func TestFlagsDocumented(t *testing.T) {
	var docs strings.Builder
	for _, p := range []string{"../../README.md", "../../docs/RESULTS.md"} {
		b, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		docs.Write(b)
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
}
