# Results

All numbers below are produced by the code in this repository on **synthetic** data
(see `docs/DESIGN.md` §1 and the limitations in `docs/KRAKEN_RELEVANCE.md`). Re-generate with
`python scripts/write_results.py` after `make all`.

## Dataset

| split | tiles | objects | per class |
|---|---|---|---|
| train | 2400 | 3875 | cylinder 595, truncated_cone 634, wedge 612, sphere 604, rock 706, debris 724 |
| val | 300 | 500 | cylinder 77, truncated_cone 84, wedge 81, sphere 63, rock 105, debris 90 |
| test | 600 | 993 | cylinder 160, truncated_cone 149, wedge 166, sphere 144, rock 184, debris 190 |

## Full-physics SAS chain (`scripts/demo_sas_processing.py`)

* Sonar: 1.88 cm nominal range resolution, 1.88 cm theoretical along-track (d/2), 0.60 m array, PRF 7.6 Hz, max unambiguous range 98 m
* Measured point-target response: 1.91 cm along-track × 3.13 cm range at 20 m; 1.95 × 2.88 cm at 45 m (range-independent)
* DPCA micronavigation: 3 cm sinusoidal sway recovered with **0.09 mm RMS error** over 36 pings
* Image contrast: true trajectory 2.12, sway ignored 1.65, DPCA-refocused 2.10

## Detector (`runs/detector_base`, test split)

* 2.11 M parameters, 24 epochs, 56 min on 4 CPU cores
* **mAP@0.5 = 0.994**, mAP@[.5:.95] = 0.829 on 600 tiles / 983 objects (4.5 ha)
* AP@0.5 per class: cylinder 0.999, truncated_cone 1.000, wedge 0.984, sphere 1.000, rock 0.986, debris 0.996
* Operating point (score ≥ 0.3): PD = 0.994 at 714 FA/km² (all objects); mine-hunting PD = 1.000 at 402 MILCO FA/km²
* MILCO recall 0.998, NOMBO rejection 0.997, MILCO precision 0.998
* Calibration ECE = 0.072

| condition | bins | recall |
|---|---|---|
| slant range [m] | 0–40, 40–70, 70–100, 100–140, 140–260 | 0.98 (n=96), 1.00 (n=163), 1.00 (n=167), 0.99 (n=226), 0.98 (n=331) |
| grazing [deg] | 0–4, 4–7, 7–12, 12–20, 20–90 | 0.99 (n=111), 0.98 (n=369), 1.00 (n=266), 1.00 (n=130), 0.98 (n=107) |
| burial fraction | -0.01–0.001, 0.001–0.2, 0.2–0.4, 0.4–0.6, 0.6–0.76 | 0.99 (n=779), 1.00 (n=51), 1.00 (n=74), 0.99 (n=79), 0.00 (n=0) |
| aspect [deg] | 0–30, 30–60, 60–90, 90–120, 120–150, 150–180.1 | 0.99 (n=160), 1.00 (n=158), 0.99 (n=148), 0.97 (n=160), 0.99 (n=192), 1.00 (n=165) |

* Second operating point (score ≥ 0.5): PD = 0.952 at 45 FA/km²; mine-hunting PD = 0.961 at 45 MILCO FA/km²

## Out-of-distribution stress split (`test_hard`: SNR 0–10 dB, burial up to 75 %, dense rock clutter, more artefacts)

* Same checkpoint, no re-training: **mAP@0.5 = 0.964**, mAP@[.5:.95] = 0.740 on 600 tiles / 1090 visible objects
* PD = 0.953 at 1317 FA/km² (score ≥ 0.3); PD = 0.867 at 156 FA/km² (score ≥ 0.5)
* MILCO recall 0.997, NOMBO rejection 0.990; ECE 0.068
* Recall by burial fraction: -0.01–0.001: 0.96 (n=691), 0.001–0.2: 1.00 (n=64), 0.2–0.4: 0.98 (n=132), 0.4–0.6: 0.93 (n=111), 0.6–0.76: 0.79 (n=92)

## Label efficiency (same test split; 2 % = 48 tiles, 10 % = 240 tiles)

| labels | init | mAP@0.5 | mAP@[.5:.95] | PD @ 0.3 | MILCO recall |
|---|---|---|---|---|---|
| 2 % | from scratch | 0.821 | 0.480 | 0.812 | 0.996 |
| 2 % | MAE pre-trained encoder | 0.814 | 0.441 | 0.852 | 0.978 |
| 10 % | from scratch | 0.958 | 0.642 | 0.956 | 0.995 |
| 10 % | MAE pre-trained encoder | 0.951 | 0.607 | 0.958 | 0.995 |
| 100 % | from scratch (reference) | 0.994 | 0.829 | 0.994 | 0.998 |

## Streaming real-time ATR (`runs/stream`)

* Swath 1887 m² rendered as one waterfall, streamed in strips; 55 tiles of 256² with 64 px overlap
* 10/11 visible objects detected, 0 false alarms
* ONNX Runtime on 4 CPU threads: **101 tiles/s (9.9 ms/tile)** vs 62 tiles/s required for 2 × 200 m swath at 10 kn (2.29 Mpixel/s) → **real-time factor 1.6×**

