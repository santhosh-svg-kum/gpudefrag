#!/usr/bin/env bash
# Regenerates Go and Python stubs from proto/ (protoc ships with grpcio-tools).
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$(go env GOPATH)/bin:$PATH"
uv run --project solver python -m grpc_tools.protoc -I proto \
  --go_out=. --go_opt=module=gpupack \
  --go-grpc_out=. --go-grpc_opt=module=gpupack \
  proto/gpupack/v1/solver.proto
uv run --project solver python -m grpc_tools.protoc -I proto \
  --python_out=solver/src/gpupack_solver/pb --grpc_python_out=solver/src/gpupack_solver/pb \
  proto/gpupack/v1/solver.proto
# grpc_tools emits absolute imports; make them package-relative.
sed -i.bak 's/^from gpupack.v1 import solver_pb2/from . import solver_pb2/' solver/src/gpupack_solver/pb/gpupack/v1/solver_pb2_grpc.py && rm solver/src/gpupack_solver/pb/gpupack/v1/*.bak
