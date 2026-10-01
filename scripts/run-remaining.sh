#!/usr/bin/env bash
# Runs the remaining v1 benchmark suites one at a time (each writes its own report).
set -u
cd "$(dirname "$0")/.."
for c in 1.0 1.25 1.5; do
  ./bin/gpudefrag-sim suite2 -compress $c -out results/suite2-x$c > results/suite2-x$c.log 2>&1
done
for l in 0.9 1.0 1.1; do
  ./bin/gpudefrag-sim suite1b -loads $l -seed-from 47 -seed-to 51 -out results/suite1b-heldout-$l > results/suite1b-heldout-$l.log 2>&1
done
echo ALL_DONE > results/remaining.done
