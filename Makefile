.PHONY: test data calib
test:
	go test ./...
data:
	./scripts/fetch-data.sh data
calib: data
	go run ./cmd/gpupack-sim calib -data data -out results/calib
