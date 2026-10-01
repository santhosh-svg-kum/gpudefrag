package mip

import (
	"bufio"
	"context"
	"fmt"
	"os/exec"
	"strings"
	"syscall"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	pb "github.com/santhosh-svg-kum/gpudefrag/proto/gpudefrag/v1"
)

// Client is a gRPC connection to a solver service.
type Client struct {
	pb.SolverClient
	conn *grpc.ClientConn
}

func Dial(addr string) (*Client, error) {
	conn, err := grpc.NewClient(addr,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(grpc.MaxCallRecvMsgSize(64<<20), grpc.MaxCallSendMsgSize(64<<20)))
	if err != nil {
		return nil, err
	}
	return &Client{SolverClient: pb.NewSolverClient(conn), conn: conn}, nil
}

func (c *Client) Close() error { return c.conn.Close() }

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
