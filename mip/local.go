package mip

import (
	"bufio"
	"context"
	"fmt"
	"os/exec"
	"strings"
	"sync/atomic"
	"syscall"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	pb "github.com/santhosh-svg-kum/gpudefrag/proto/gpudefrag/v1"
)

// Client is a gRPC connection to one or more solver services; requests are
// spread round-robin across them.
type Client struct {
	pb.SolverClient
	conns []*grpc.ClientConn
}

// roundRobin sends each call to the next client in turn.
type roundRobin struct {
	clients []pb.SolverClient
	next    atomic.Uint64
}

func newRoundRobin(cs []pb.SolverClient) *roundRobin { return &roundRobin{clients: cs} }

func (r *roundRobin) pick() pb.SolverClient {
	return r.clients[(r.next.Add(1)-1)%uint64(len(r.clients))]
}

func (r *roundRobin) Place(ctx context.Context, in *pb.PlaceRequest, opts ...grpc.CallOption) (*pb.PlaceResponse, error) {
	return r.pick().Place(ctx, in, opts...)
}

func (r *roundRobin) Defrag(ctx context.Context, in *pb.DefragRequest, opts ...grpc.CallOption) (*pb.DefragResponse, error) {
	return r.pick().Defrag(ctx, in, opts...)
}

func (r *roundRobin) Health(ctx context.Context, in *pb.HealthRequest, opts ...grpc.CallOption) (*pb.HealthResponse, error) {
	return r.pick().Health(ctx, in, opts...)
}

// Dial connects to a solver at addr, or to several at a comma-separated list
// of addresses, round-robining requests across them.
func Dial(addr string) (*Client, error) {
	c := &Client{}
	var cs []pb.SolverClient
	for _, a := range strings.Split(addr, ",") {
		conn, err := grpc.NewClient(a,
			grpc.WithTransportCredentials(insecure.NewCredentials()),
			grpc.WithDefaultCallOptions(grpc.MaxCallRecvMsgSize(64<<20), grpc.MaxCallSendMsgSize(64<<20)))
		if err != nil {
			c.Close()
			return nil, err
		}
		c.conns = append(c.conns, conn)
		cs = append(cs, pb.NewSolverClient(conn))
	}
	if len(cs) == 1 {
		c.SolverClient = cs[0]
	} else {
		c.SolverClient = newRoundRobin(cs)
	}
	return c, nil
}

func (c *Client) Close() error {
	var first error
	for _, conn := range c.conns {
		if err := conn.Close(); err != nil && first == nil {
			first = err
		}
	}
	return first
}

// StartLocalN starts n solver processes and returns their addresses joined by
// commas (accepted by Dial) and a function that stops all of them.
func StartLocalN(ctx context.Context, solverDir string, n int) (string, func(), error) {
	if n < 1 {
		return "", nil, fmt.Errorf("solver process count must be >= 1, got %d", n)
	}
	var addrs []string
	var stops []func()
	stopAll := func() {
		for _, s := range stops {
			s()
		}
	}
	for i := 0; i < n; i++ {
		a, stop, err := StartLocal(ctx, solverDir)
		if err != nil {
			stopAll()
			return "", nil, err
		}
		addrs = append(addrs, a)
		stops = append(stops, stop)
	}
	return strings.Join(addrs, ","), stopAll, nil
}

// StartLocal launches the Python solver from solverDir with `uv run` on a
// free port and returns its address and a stop function.
func StartLocal(ctx context.Context, solverDir string) (string, func(), error) {
	ctx, cancel := context.WithCancel(ctx)
	cmd := exec.CommandContext(ctx, "uv", "run", "--project", solverDir, "python", "-m", "gpudefrag_solver.server", "--port", "0", "--threads", "16")
	// uv starts python as a child; run both in their own process group so
	// stop() can kill the whole group instead of orphaning the server.
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cmd.Cancel = func() error { return syscall.Kill(-cmd.Process.Pid, syscall.SIGTERM) }
	out, err := cmd.StdoutPipe()
	if err != nil {
		cancel()
		return "", nil, err
	}
	if err := cmd.Start(); err != nil {
		cancel()
		return "", nil, err
	}
	stop := func() { cancel(); _ = cmd.Wait() }
	lines := make(chan string, 1)
	go func() {
		sc := bufio.NewScanner(out)
		for sc.Scan() {
			if strings.HasPrefix(sc.Text(), "LISTENING ") {
				lines <- sc.Text()
				return
			}
		}
		close(lines)
	}()
	select {
	case l, ok := <-lines:
		if !ok {
			stop()
			return "", nil, fmt.Errorf("solver exited before listening")
		}
		return "127.0.0.1:" + strings.TrimPrefix(l, "LISTENING "), stop, nil
	case <-time.After(60 * time.Second):
		stop()
		return "", nil, fmt.Errorf("solver did not start within 60s")
	}
}
