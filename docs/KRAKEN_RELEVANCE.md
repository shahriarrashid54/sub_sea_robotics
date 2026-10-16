# Why this project maps onto Kraken Robotics' work

> Written as preparation material for applying to Kraken Robotics (St. John's /
> Mount Pearl, NL). All company facts below come from public sources (press
> releases, published papers, product pages, financial reports) – see the
> references at the end. Nothing here is affiliated with or endorsed by Kraken.

## Kraken in one paragraph (public information, 2026)

Kraken Robotics designs and builds **synthetic aperture sonar** (AquaPix®
MINSAS / INSAS, and the towed **KATFISH™** system), **SeaPower™** pressure-
tolerant subsea batteries, **SeaVision®** 3D underwater laser imaging and
(since the 2025 acquisition of 3D at Depth) subsea LiDAR, and provides
robotics-as-a-service seabed surveys. In 2026 the company reported ~$327 M of
product orders year-to-date, acquired the Covelya Group (Sonardyne, EIVA,
Voyis, Forcys, Chelsea Technologies), demonstrated autonomous KATFISH launch
and recovery from SEFINE's RD-22 USV with 3 cm × 3 cm imagery streamed live to
shore, and signed an MOU to **develop automatic target recognition (ATR) for
Kraken SAS** inside SEFINE's mission-planning software for mine
countermeasures (MCM).

## Project ↔ Kraken map

| Kraken capability / need | What this project demonstrates | Where |
|---|---|---|
| **SAS physics** (AquaPix: 337 kHz, 40 kHz LFM, 3 cm → 2 cm "Ultra HD" resolution; real-time SAS processing; motion compensation without INS) | Pulse-level multi-element SAS simulation with public MINSAS-class parameters, matched filtering, back-projection image formation reproducing the range-independent *d/2* resolution (1.9 cm measured vs 1.875 cm theory), and **DPCA micronavigation** that recovers a 3 cm sway to 0.09 mm RMS and refocuses the image | `sim/acoustics.py`, `sim/sas_processor.py`, `assets/sas_image_formation.png` |
| **ATR for MCM** (SEFINE MOU; operators classifying contacts in a command centre) | Anchor-free detector with explicit **shadow supervision**, MILCO vs NOMBO reporting, **PD vs false alarms / km²**, confusion by aspect, calibration, MC-dropout uncertainty and Grad-CAM evidence maps for operator trust | `models/`, `train/evaluate.py`, `assets/*.png` |
| **Scarce labelled sonar data** | Physics-grounded synthetic data generator with pixel-exact labels, domain randomisation, and a **self-supervised MAE** pre-training stage showing label efficiency; adapters for real COCO / YOLO sonar datasets | `sim/renderer.py`, `models/ssl.py`, `docs/REAL_DATA.md` |
| **Live-streamed KATFISH imagery at up to 10 kn, 200 m per side** | `StreamingATR` consumes along-track strips with overlap buffering and merges contacts; `benchmark_realtime` compares CPU throughput with the 2 × 200 m × 10 kn area rate (≈ 2.3 Mpixel/s at 3 cm) and reports a real-time factor; ONNX export for embedded / shore-side deployment | `deploy/stream.py`, `deploy/export_onnx.py` |
| **Mission planning / autonomy** (USV + towed SAS, AUV re-acquisition, Covelya's autonomy software) | Aspect-dependent Bayesian multi-look fusion and an **information-gain re-acquisition planner** with Dubins-style turn penalties and a path budget; GeoJSON contact export for GIS / mission tools | `autonomy/`, `deploy/geo.py` |
| **Engineering practice** (towed-systems software QA roles, C++/Python, Linux) | Typed, documented Python package, 25 unit tests covering physics invariants (shadow length, PSF width, speed constraint, DPCA), CI workflow, reproducible `make all` pipeline, CLI | `tests/`, `.github/workflows/ci.yml`, `Makefile` |

## Talking points for an interview

1. **Why back-projection rather than ω-k?** – exactness under arbitrary
   motion; it isolates the effect of trajectory error, which is the point of
   the DPCA demo. Discuss when a wavenumber-domain algorithm (as used in
   real-time processors) is preferable and how DPCA + INS fusion works.
2. **Why highlight boxes but shadow supervision?** – the contact position is
   the physical object; the shadow is the strongest *shape* cue but its
   extent is a function of range and altitude, not of the object alone.
3. **Why PD vs FA/km² instead of mAP?** – MCM clearance decisions and
   re-acquisition cost are driven by false alarms per unit area at a required
   PD; mAP hides the operating point.
4. **Sim-to-real gap** – what the renderer gets right (shadow geometry,
   speckle statistics, grazing dependence), what it misses (elastic target
   response, multipath), and how MAE pre-training on unlabelled real imagery
   plus mixed-batch fine-tuning closes the gap.
5. **Aspect dependence and multi-look fusion** – a single SAS pass already
   gives multiple aspects over the aperture; the planner formalises "where
   should the AUV look next" as expected information gain per metre.
6. **Deployability** – 2.1 M parameters, ONNX, measured tiles/s versus the
   sensor's pixel rate; how the same pipeline would run shore-side on a
   streamed KATFISH feed or on-board an AUV with SeaPower energy budgets in
   mind.

## Honest limitations to state up front

* Trained and evaluated on **synthetic** imagery only (no public SAS mine
  dataset exists); absolute numbers do not transfer to real data.
* The acoustic model is point-scatterer based (no elastic / resonant target
  response, no multipath).
* DPCA here estimates sway only; yaw / surge and INS fusion are extensions.

## Public references

* Kraken Robotics, *Kraken Demonstrates KATFISH Launch From USV* (Apr 2026)
  and *Q1 2026 Financial Results* (May 2026) – krakenrobotics.com news releases.
* Kraken Robotics, *Ultra HD Upgrade for AquaPix Imaging Sonars* – raw 3.0 ×
  3.3 cm improved to 1.9 × 2.1 cm.
* Dillon, J. et al., *Shallow Water Survey with a Miniature Synthetic Aperture
  Sonar*, OCEANS 2019 – 337 kHz centre frequency, 40 kHz bandwidth.
* Shea, D. et al., *Real-time SAS processing …*, 2014 (Kraken RTSAS).
* Myers, Williams, Dillon, *Evaluation of a non-coherent SAS …*, 2018 –
  DPCA as the primary micronavigation method.
* Kraken Robotics / SEFINE SISAM MOU on ATR for Kraken SAS (May 2026).
* Kraken Robotics completes acquisition of 3D at Depth (Apr 2025) and of the
  Covelya Group (Jul 2026).
