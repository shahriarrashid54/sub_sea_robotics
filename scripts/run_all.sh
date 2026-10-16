#!/usr/bin/env bash
# Reproduce the whole project from scratch on a CPU machine (~2.5 h on 4 cores).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3}

$PY -m seabed_sentinel info
$PY -m seabed_sentinel simulate --root data/sas_synth --n-train 2400 --n-val 300 --n-test 600 --workers 4
$PY -m seabed_sentinel train --config configs/detector_base.yaml
$PY -m seabed_sentinel evaluate --checkpoint runs/detector_base/best.pt --out-dir runs/detector_base/eval
$PY -m seabed_sentinel export --checkpoint runs/detector_base/best.pt --out runs/detector_base/model.onnx
$PY -m seabed_sentinel stream --onnx runs/detector_base/model.onnx --out-dir runs/stream
$PY -m seabed_sentinel pretrain --data-root data/sas_synth --out-dir runs/mae --epochs 5
$PY -m seabed_sentinel train --config configs/detector_scratch_10pct.yaml
$PY -m seabed_sentinel evaluate --checkpoint runs/detector_scratch_10pct/best.pt --out-dir runs/detector_scratch_10pct/eval
$PY -m seabed_sentinel train --config configs/detector_mae_10pct.yaml
$PY -m seabed_sentinel evaluate --checkpoint runs/detector_mae_10pct/best.pt --out-dir runs/detector_mae_10pct/eval
$PY scripts/demo_sas_processing.py --out assets
$PY scripts/make_figures.py
$PY -m seabed_sentinel plan --metrics runs/detector_base/eval/metrics.json --out runs/plan/plan.json
