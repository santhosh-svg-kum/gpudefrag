"""gRPC server for the gpupack solver.

    uv run python -m gpupack_solver.server --port 50051   (0 = pick a free port)

Prints "LISTENING <port>" on stdout once ready.
"""
import argparse
import sys
from concurrent import futures

import grpc

from .pb.gpupack.v1 import solver_pb2 as pb
from .pb.gpupack.v1 import solver_pb2_grpc as pbg
from .place import solve_place

VERSION = "0.1.0"


class Solver(pbg.SolverServicer):
    def Place(self, request, context):
        return solve_place(request)

    def Defrag(self, request, context):
        from .defrag import solve_defrag  # added in M3
        return solve_defrag(request)

    def Health(self, request, context):
        return pb.HealthResponse(version=VERSION)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=50051)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args(argv)
    opts = [("grpc.max_receive_message_length", 64 << 20), ("grpc.max_send_message_length", 64 << 20)]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=args.threads), options=opts)
    pbg.add_SolverServicer_to_server(Solver(), server)
    port = server.add_insecure_port(f"127.0.0.1:{args.port}")
    server.start()
    print(f"LISTENING {port}", flush=True)
    server.wait_for_termination()


if __name__ == "__main__":
    sys.exit(main())
