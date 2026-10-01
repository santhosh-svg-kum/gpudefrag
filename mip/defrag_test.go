package mip

import (
	"context"
	"os/exec"
	"testing"
	"time"

	"google.golang.org/grpc"

	"gpupack/model"
	pb "gpupack/proto/gpupack/v1"
	"gpupack/sim"
)

type fakeDefrag struct {
	resp *pb.DefragResponse
	got  *pb.DefragRequest
}

func (f *fakeDefrag) Defrag(_ context.Context, r *pb.DefragRequest, _ ...grpc.CallOption) (*pb.DefragResponse, error) {
	f.got = r
	return f.resp, nil
}

// Textbook case: two 2-GPU nodes each half used by a 1-GPU pod; a 2-GPU pod
// is blocked although 2 GPUs are free in total.
func defragFixture() (*sim.Cluster, []Running, model.PodRes) {
	a := model.NewNode("a", 64000, 1<<20, 2, "")
	b := model.NewNode("b", 64000, 1<<20, 2, "")
	c := sim.NewCluster([]*model.NodeRes{a, b})
	one := model.PodRes{MilliCPU: 1000, MemMiB: 1, GpuNum: 1, GpuMilli: 1000}
	c.Bind(0, one, []int{0})
	c.Bind(1, one, []int{0})
	run := []Running{{ID: 10, Node: 0, GPUs: []int{0}, Res: one, Cost: 100}, {ID: 11, Node: 1, GPUs: []int{0}, Res: one, Cost: 50}}
	blocked := model.PodRes{MilliCPU: 1000, MemMiB: 1, GpuNum: 2, GpuMilli: 1000}
	return c, run, blocked
}

func TestDefragRequestScope(t *testing.T) {
	c, run, blocked := defragFixture()
	f := &fakeDefrag{resp: &pb.DefragResponse{Status: "INFEASIBLE"}}
	d := &Defragmenter{Solver: f, MaxMoves: 4, TimeLimit: time.Second}
	if _, why := d.Plan(c, blocked, run); why != "status" {
		t.Fatalf("why=%s", why)
	}
	if len(f.got.Blocked[0].Candidates) != 2 || len(f.got.Movable) != 2 {
		t.Fatalf("both nodes are targets and both pods movable: %+v", f.got)
	}
}

func TestDefragValidatesPlans(t *testing.T) {
	c, run, blocked := defragFixture()
	good := &pb.DefragResponse{Status: "OPTIMAL", Moves: []*pb.Move{{Pod: 11, ToNode: 0, Gpus: []int32{1}}},
		Placements: []*pb.Assignment{{Node: 1, Gpus: []int32{0, 1}}}, Cost: 50}
	d := &Defragmenter{Solver: &fakeDefrag{resp: good}, MaxMoves: 4, TimeLimit: time.Second}
	p, why := d.Plan(c, blocked, run)
	if why != "" || p.Target != 1 || len(p.Moves) != 1 || p.Cost != 50 {
		t.Fatalf("plan=%+v why=%s", p, why)
	}

	bad := &pb.DefragResponse{Status: "OPTIMAL", Moves: []*pb.Move{{Pod: 11, ToNode: 0, Gpus: []int32{0}}}, // GPU 0 on a is busy
		Placements: []*pb.Assignment{{Node: 1, Gpus: []int32{0, 1}}}}
	d.Solver = &fakeDefrag{resp: bad}
	if _, why := d.Plan(c, blocked, run); why != "invalid" {
		t.Fatalf("why=%s", why)
	}
	tooMany := &pb.DefragResponse{Status: "OPTIMAL", Moves: []*pb.Move{{Pod: 10, ToNode: 1, Gpus: []int32{1}}, {Pod: 11, ToNode: 0, Gpus: []int32{1}}},
		Placements: []*pb.Assignment{{Node: 1, Gpus: []int32{0, 1}}}}
	d.Solver, d.MaxMoves = &fakeDefrag{resp: tooMany}, 1
	if _, why := d.Plan(c, blocked, run); why != "invalid" {
		t.Fatalf("max moves: why=%s", why)
	}
}

func TestDefragSkipsWhenNothingCanHelp(t *testing.T) {
	c, run, blocked := defragFixture()
	blocked.GpuNum = 4 // no 4-GPU node exists
	d := &Defragmenter{Solver: &fakeDefrag{}, MaxMoves: 4}
	if _, why := d.Plan(c, blocked, run); why != "no_target" {
		t.Fatalf("why=%s", why)
	}
}

func TestDefragRealSolver(t *testing.T) {
	if _, err := exec.LookPath("uv"); err != nil {
		t.Skip("uv not installed")
	}
	addr, stop, err := StartLocal(context.Background(), "../solver")
	if err != nil {
		t.Fatal(err)
	}
	defer stop()
	cl, _ := Dial(addr)
	defer cl.Close()
	c, run, blocked := defragFixture()
	d := &Defragmenter{Solver: cl, MaxMoves: 4, TimeLimit: time.Second}
	p, why := d.Plan(c, blocked, run)
	if why != "" || len(p.Moves) != 1 || p.Moves[0].ID != 11 || p.Cost != 50 {
		t.Fatalf("expected moving the cheaper pod: %+v %s", p, why)
	}
}
