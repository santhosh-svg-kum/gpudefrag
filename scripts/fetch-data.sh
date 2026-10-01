#!/usr/bin/env bash
# Downloads the Alibaba openb trace and FGD's published reference curves from
# hkust-adsl/kubernetes-scheduler-simulator (Apache-2.0) at a pinned commit and
# verifies their checksums. Data is never committed to this repo.
set -euo pipefail
COMMIT=8f3d6417353c5083c6d56617f255fadd8dc306bc
BASE="https://raw.githubusercontent.com/hkust-adsl/kubernetes-scheduler-simulator/$COMMIT"
DIR="${1:-data}"
mkdir -p "$DIR/openb" "$DIR/ref"
fetch() { [ -s "$DIR/$2" ] || curl -fsSL "$BASE/$1" -o "$DIR/$2"; }
fetch data/csv/openb_node_list_gpu_node.csv openb/openb_node_list_gpu_node.csv
fetch data/csv/openb_pod_list_default.csv openb/openb_pod_list_default.csv
fetch experiments/analysis/expected_results/analysis_allo_discrete.csv ref/analysis_allo_discrete.csv
# HeliosData (SenseTime, SC '21), CC-BY-4.0: GPU job traces with gang sizes.
mkdir -p "$DIR/helios"
[ -s "$DIR/helios/data.zip" ] || curl -fsSL https://raw.githubusercontent.com/S-Lab-System-Group/HeliosData/master/data.zip -o "$DIR/helios/data.zip"
cd "$DIR" && shasum -a 256 -c <<'SUMS'
2beca64b4d3dfa342036a34b56a495c6cef9225db836c81f541282cb1df320b5  openb/openb_node_list_gpu_node.csv
1ee7ed79c27a3b0861cda8ddba86a004c6aba904caafa329a76ae93ca63834a8  openb/openb_pod_list_default.csv
e00aff5923a52d53e47c16ec9d19190de4e788c15334865a77bd46e1bd006ea1  ref/analysis_allo_discrete.csv
3d22a5f6c0ae669e2fcbfe4200fa9c48664507bc397c677bad8f085222c032ac  helios/data.zip
SUMS
[ -s helios/data/Venus/cluster_log.csv ] || (cd helios && unzip -q -o data.zip)
