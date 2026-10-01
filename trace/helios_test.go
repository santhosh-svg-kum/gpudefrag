package trace

import (
	"testing"
	"time"
)

func TestLoadHelios(t *testing.T) {
	jobs, st, err := LoadHelios("testdata/helios.csv")
	if err != nil {
		t.Fatal(err)
	}
	if st.CPUOnly != 1 || st.Uneven != 1 || st.ZeroDuration != 1 || len(jobs) != 4 {
		t.Fatalf("stats %+v jobs %d", st, len(jobs))
	}
	one := jobs[0]
	if len(one.Gang) != 1 || one.Gang[0].Res.GpuNum != 1 || one.Gang[0].Res.MilliCPU != 6000 || one.Duration != 3600 || one.Arrive != 0 {
		t.Fatalf("single %+v", one)
	}
	g := jobs[1]
	if len(g.Gang) != 2 || g.Gang[0].Res.GpuNum != 8 || g.Gang[1].Res.MilliCPU != 48000 || g.Arrive != 60 {
		t.Fatalf("gang %+v", g)
	}
	if g.Gang[0].Name == g.Gang[1].Name || g.Pod.Name != "2" {
		t.Fatal("gang pods need distinct names; Pod carries the job id")
	}
	if jobs[3].Gang[0].Res.MilliCPU != 100 || jobs[2].Gang[0].Res.GpuNum != 6 {
		t.Fatal("zero cpu gets the kube default")
	}
	if st.Start != time.Date(2020, 3, 1, 0, 0, 0, 0, time.UTC) {
		t.Fatal(st.Start)
	}
}
