package mip

import (
	"context"
	"math/rand"
	"os"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/protobuf/proto"

	pb "gpupack/proto/gpupack/v1"
	"gpupack/sched"
	"gpupack/sim"
	"gpupack/trace"
	"gpupack/workload"
)

type dumpSolver struct{ path string }

func (d dumpSolver) Place(_ context.Context, r *pb.PlaceRequest, _ ...grpc.CallOption) (*pb.PlaceResponse, error) {
	b, _ := proto.Marshal(r)
	os.WriteFile(d.path, b, 0o644)
	return &pb.PlaceResponse{Status: "INFEASIBLE"}, nil
}

// GPUPACK_DUMP=/path go test ./mip -run TestDumpRealRequest writes a realistic
// mid-run PlaceRequest (after 4000 FGD placements) for solver profiling.
func TestDumpRealRequest(t *testing.T) {
	path := os.Getenv("GPUPACK_DUMP")
	if path == "" {
		t.Skip("set GPUPACK_DUMP")
	}
	nodes, _ := trace.LoadOpenbNodes("../data/openb/openb_node_list_gpu_node.csv")
	pods, _ := trace.LoadOpenbPods("../data/openb/openb_pod_list_default.csv")
	typ := workload.TypicalPods(pods)
	c := sim.NewCluster(nodes)
	rng := rand.New(rand.NewSource(42))
	seq := workload.Prepare(rng, pods, c.TotalGpuMilli(), 1.3)
	fgd := sched.NewFGD(typ)
	for _, p := range seq[:4000] {
		sched.Place(c, fgd, p, nil)
	}
	pl := &Placer{Solver: dumpSolver{path}, Typical: typ, K: 16, TimeLimit: 500 * time.Millisecond}
	pl.PlaceBatch(c, seq[4000:4016])
}
