.PHONY: test data calib suite1a suite1b suite2 sweep-defrag
test:
	(cd solver && uv run pytest -q)
	go test ./...
data:
	./scripts/fetch-data.sh data
calib: data
	go run ./cmd/gpudefrag-sim calib -data data -out results/calib
suite1a: data
	go run ./cmd/gpudefrag-sim suite1a -data data -out results/suite1a
suite1b: data
	go run ./cmd/gpudefrag-sim suite1b -data data -out results/suite1b
suite2: data
	go run ./cmd/gpudefrag-sim suite2 -data data -out results/suite2
sweep-defrag: data
	go run ./cmd/gpudefrag-sim suite1b -data data -out results/sweep-defrag -variants FGD,FGD+defrag,gpudefrag+defrag -restart-sweep 60,300,900 -grace-sweep 0,30
