PY ?= python3
DATA ?= data/sas_synth

.PHONY: install data train pretrain eval export stream demo-sas figures test lint all

install:
	$(PY) -m pip install -e ".[dev]"

data:            ## render the synthetic SAS dataset (train/val/test)
	$(PY) -m seabed_sentinel simulate --root $(DATA) --n-train 2400 --n-val 300 --n-test 600 --workers 4

train:           ## train the baseline detector
	$(PY) -m seabed_sentinel train --config configs/detector_base.yaml

pretrain:        ## MAE self-supervised pre-training of the encoder
	$(PY) -m seabed_sentinel pretrain --data-root $(DATA) --out-dir runs/mae --epochs 6

eval:            ## evaluate on the test split (mAP, PD/FA, breakdowns)
	$(PY) -m seabed_sentinel evaluate --checkpoint runs/detector_base/best.pt --data-root $(DATA) --out-dir runs/detector_base/eval

export:          ## export ONNX + parity check
	$(PY) -m seabed_sentinel export --checkpoint runs/detector_base/best.pt --out runs/detector_base/model.onnx

stream:          ## real-time streaming demo on a long simulated swath
	$(PY) -m seabed_sentinel stream --onnx runs/detector_base/model.onnx --out-dir runs/stream

demo-sas:        ## full-physics SAS echo simulation, back-projection and DPCA demo
	$(PY) scripts/demo_sas_processing.py --out assets

figures:         ## regenerate all README figures from the trained model
	$(PY) scripts/make_figures.py

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check seabed_sentinel tests scripts

all: data train eval export stream demo-sas figures
