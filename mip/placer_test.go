package mip

import (
	"context"
	"errors"
	"os/exec"
	"testing"
	"time"

	"google.golang.org/grpc"

	"gpupack/frag"
	"gpupack/model"
	pb "gpupack/proto/gpupack/v1"
	"gpupack/sim"
)

type fakeSolver struct {
	resp *pb.PlaceResponse
	err  error
	got  *pb.PlaceRequest
}

func (f *fakeSolver) Place(_ context.Context, r *pb.PlaceRequest, _ ...grpc.CallOption) (*pb.PlaceResponse, error) {
	f.got = r
	return f.resp, f.err
}

// FGD is misled by a typical-pod distribution of small share pods: it puts
// the 500m pod on the idle node (tie -> name "A"), stranding the 1-GPU pod.
func trap() (*sim.Cluster, []model.Pod, []frag.TargetPod) {
	a := model.NewNode("A", 64000, 1<<20, 1, "")
	b := model.NewNode("B", 64000, 1<<20, 1, "")
	b.GpuLeft[0] = 500
	pods := []model.Pod{
		{Name: "p0", Res: model.PodRes{MilliCPU: 1000, MemMiB: 1, GpuNum: 1, GpuMilli: 500}},
		{Name: "p1", Res: model.PodRes{MilliCPU: 1000, MemMiB: 1, GpuNum: 1, GpuMilli: 1000}},
	}
	typ := []frag.TargetPod{{Res: model.PodRes{MilliCPU: 1000, GpuNum: 1, GpuMilli: 300}, Pct: 1}}
	return sim.NewCluster([]*model.NodeRes{a, b}), pods, typ
}

func placer(s Solver, typ []frag.TargetPod) *Placer {
	return &Placer{Solver: s, Typical: typ, K: 8, TimeLimit: time.Second, Deterministic: true}
}

func TestAcceptsBetterSolution(t *testing.T) {
	c, pods, typ := trap()
	f := &fakeSolver{resp: &pb.PlaceResponse{Status: "OPTIMAL", Assignments: []*pb.Assignment{
		{Pod: 0, Node: 1, Gpus: []int32{0}}, {Pod: 1, Node: 0, Gpus: []int32{0}}}}}
	pl := placer(f, typ)
	placed := pl.PlaceBatch(c, pods)
	if !placed[0] || !placed[1] || c.UsedGpuMilli() != 2000 {
		t.Fatalf("placed=%v used=%d stats=%+v", placed, c.UsedGpuMilli(), pl.Stats)
	}
	if pl.Stats.Improved != 1 {
		t.Fatalf("stats %+v", pl.Stats)
	}
	if len(f.got.Hint) != 1 {
		t.Fatalf("FGD hint should place one pod, got %v", f.got.Hint)
	}
}

func TestFallsBackOnBadAnswers(t *testing.T) {
	cases := map[string]*fakeSolver{
		"rpc":     {err: errors.New("down")},
		"invalid": {resp: &pb.PlaceResponse{Status: "OPTIMAL", Assignments: []*pb.Assignment{{Pod: 0, Node: 0, Gpus: []int32{0}}, {Pod: 1, Node: 0, Gpus: []int32{0}}}}},
		"status":  {resp: &pb.PlaceResponse{Status: "INFEASIBLE"}},
		"gpu_cnt": {resp: &pb.PlaceResponse{Status: "OPTIMAL", Assignments: []*pb.Assignment{{Pod: 1, Node: 0}}}},
		"dup_pod": {resp: &pb.PlaceResponse{Status: "OPTIMAL", Assignments: []*pb.Assignment{{Pod: 0, Node: 1, Gpus: []int32{0}}, {Pod: 0, Node: 0, Gpus: []int32{0}}}}},
	}
	for reason, f := range cases {
		c, pods, typ := trap()
		pl := placer(f, typ)
		placed := pl.PlaceBatch(c, pods)
		if !placed[0] || placed[1] || c.UsedGpuMilli() != 1000 {
			t.Errorf("%s: must fall back to the FGD hint: placed=%v used=%d", reason, placed, c.UsedGpuMilli())
		}
		want := reason
		if reason == "gpu_cnt" || reason == "dup_pod" {
			want = "invalid"
		}
		if pl.Stats.Fallbacks[want] != 1 {
			t.Errorf("%s: fallbacks %v", reason, pl.Stats.Fallbacks)
		}
	}
}

func TestKeepsHintWhenNotBetter(t *testing.T) {
	c, pods, typ := trap()
	f := &fakeSolver{resp: &pb.PlaceResponse{Status: "FEASIBLE", Assignments: []*pb.Assignment{{Pod: 0, Node: 0, Gpus: []int32{0}}}}}
	pl := placer(f, typ)
	pl.PlaceBatch(c, pods)
	if pl.Stats.Improved != 0 || c.UsedGpuMilli() != 1000 {
		t.Fatalf("%+v", pl.Stats)
	}
}

func TestRealSolver(t *testing.T) {
	if _, err := exec.LookPath("uv"); err != nil {
		t.Skip("uv not installed")
	}
	addr, stop, err := StartLocal(context.Background(), "../solver")
	if err != nil {
		t.Fatal(err)
	}
	defer stop()
	cl, err := Dial(addr)
	if err != nil {
		t.Fatal(err)
	}
	defer cl.Close()
	c, pods, typ := trap()
	pl := placer(cl, typ)
	placed := pl.PlaceBatch(c, pods)
	if !placed[0] || !placed[1] {
		t.Fatalf("real solver should place both: %v %+v", placed, pl.Stats)
	}
}
