package mip

import (
	"context"
	"os/exec"
	"strings"
	"testing"
	"time"

	"google.golang.org/grpc"

	pb "github.com/santhosh-svg-kum/gpudefrag/proto/gpudefrag/v1"
)

type countingClient struct{ n int }

func (c *countingClient) Place(context.Context, *pb.PlaceRequest, ...grpc.CallOption) (*pb.PlaceResponse, error) {
	c.n++
	return &pb.PlaceResponse{}, nil
}
func (c *countingClient) Defrag(context.Context, *pb.DefragRequest, ...grpc.CallOption) (*pb.DefragResponse, error) {
	c.n++
	return &pb.DefragResponse{}, nil
}
func (c *countingClient) Health(context.Context, *pb.HealthRequest, ...grpc.CallOption) (*pb.HealthResponse, error) {
	c.n++
	return &pb.HealthResponse{}, nil
}

func TestRoundRobinSpreadsRequestsEvenly(t *testing.T) {
	cs := []*countingClient{{}, {}, {}}
	rr := newRoundRobin([]pb.SolverClient{cs[0], cs[1], cs[2]})
	for i := 0; i < 9; i++ {
		if _, err := rr.Place(context.Background(), &pb.PlaceRequest{}); err != nil {
			t.Fatal(err)
		}
	}
	rr.Defrag(context.Background(), &pb.DefragRequest{})
	for i, c := range cs {
		want := 3
		if i == 0 {
			want = 4
		}
		if c.n != want {
			t.Fatalf("client %d got %d requests, want %d", i, c.n, want)
		}
	}
}

func TestDialMultipleAddrs(t *testing.T) {
	cl, err := Dial("127.0.0.1:1,127.0.0.1:2")
	if err != nil {
		t.Fatal(err)
	}
	if len(cl.conns) != 2 {
		t.Fatalf("conns = %d, want 2", len(cl.conns))
	}
	if err := cl.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestStartLocalNStartsAndStopsAll(t *testing.T) {
	if _, err := exec.LookPath("uv"); err != nil {
		t.Skip("uv not installed")
	}
	addr, stop, err := StartLocalN(context.Background(), "../solver", 2)
	if err != nil {
		t.Fatal(err)
	}
	addrs := strings.Split(addr, ",")
	if len(addrs) != 2 || addrs[0] == addrs[1] {
		t.Fatalf("want 2 distinct addresses, got %q", addr)
	}
	cl, _ := Dial(addr)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	for i := 0; i < 4; i++ {
		if _, err := cl.Health(ctx, &pb.HealthRequest{}); err != nil {
			t.Fatalf("health %d: %v", i, err)
		}
	}
	cancel()
	cl.Close()
	stop()
	time.Sleep(500 * time.Millisecond)
	for _, a := range addrs {
		c, _ := Dial(a)
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		if _, err := c.Health(ctx, &pb.HealthRequest{}); err == nil {
			t.Fatalf("solver %s still answering after stop()", a)
		}
		cancel()
		c.Close()
	}
}

func TestStartLocalNRejectsNonPositive(t *testing.T) {
	if _, _, err := StartLocalN(context.Background(), "../solver", 0); err == nil {
		t.Fatal("n=0 must error")
	}
}
