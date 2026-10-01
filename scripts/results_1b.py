#!/usr/bin/env python3
"""Builds the suite 1b section of docs/RESULTS.md from generated reports:
original seeds (42-46, used to choose gpupack's defaults) and held-out seeds
(47-51, never used for tuning), each with paired comparisons vs FGD."""
import json, subprocess, sys

def paired(path):
    return subprocess.run(["python3", "scripts/paired.py", path, "FGD", "alloc_pct", "P95", "unplaced"],
                          capture_output=True, text=True).stdout

def table(report, load):
    r = open(report).read()
    start = r.index(f"## Offered load {load}")
    seg = r[start:]
    return seg[seg.index("| variant"):].split("\n\n")[0].strip()

out = ["## 2. Suite 1b: timed, SYNTHETIC workload from openb (with defrag)\n"]
r = open("results/suite1b-heldout-0.9/report.md").read().split("\n")[2]
out.append(r + "\n")
out.append("Two seed sets. Seeds 42–46 ran first and were used to diagnose large-job starvation and to choose gpupack's defaults (idle-node penalty, GPU-weighted fragmentation). Seeds 47–51 are **held out** and ran once with those defaults. Draw conclusions from the held-out set. Note that the original-seed gpupack rows predate the defaults change.\n")
for load in ("0.9", "1.0", "1.1"):
    out.append(f"### Offered load {load}\n")
    out.append("**Held-out seeds 47–51:**\n\n" + table(f"results/suite1b-heldout-{load}/report.md", load) + "\n")
    out.append("Paired vs FGD (held-out):\n\n```\n" + paired(f"results/suite1b-heldout-{load}/runs.json") + "```\n")
    # original seeds: runs.json holds all loads; filter
    runs = [r for r in json.load(open("results/suite1b/runs.json")) if abs(r["load"] - float(load)) < 1e-9]
    tmp = f"/tmp/claude-1b-{load}.json"
    json.dump(runs, open(tmp, "w"))
    out.append("Paired vs FGD (original seeds 42–46, pre-defaults gpupack):\n\n```\n" + paired(tmp) + "```\n")
sys.stdout.write("\n".join(out))
