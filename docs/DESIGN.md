# Design and physics notes

This document explains *why* the simulator and the ATR stack are built the way
they are, and derives the physics used. It is written for a sonar / robotics
engineer reviewing the project.

---

## 1. Why simulate at all?

Real synthetic aperture sonar (SAS) imagery of mine-like objects is scarce,
expensive to collect and frequently classified. Every serious ATR effort in
mine countermeasures (MCM) therefore leans on simulation for three things:

1. **Volume and label quality** – thousands of tiles with pixel-exact
   highlight / shadow masks, class, aspect, range, grazing angle and burial
   state, which no human annotator can provide consistently.
2. **Controlled stress tests** – performance as a function of range, grazing
   angle, burial, aspect and SNR, which real surveys never sample uniformly.
3. **Sim-to-real transfer** – a physically faithful renderer plus domain
   randomisation lets a model pre-trained on synthetic data fine-tune on a few
   hundred real tiles instead of tens of thousands.

Two fidelity tiers are implemented and *kept consistent with each other*:

| tier | module | what it models | cost |
|---|---|---|---|
| **Full physics** | `sim/acoustics.py`, `sim/sas_processor.py` | LFM chirp, multi-element receiver array, pulse-level echoes, matched filter, back-projection image formation, platform motion errors, DPCA micronavigation | ~1–2 min per 5 × 6 m scene |
| **Image-domain renderer** | `sim/renderer.py` | the *appearance* physics of a focused SAS image: grazing-angle backscatter, exact ray-cast shadows, coherent speckle with the system PSF, SNR / TVG, platform artefacts | ~0.4 s per 8.6 × 8.6 m tile |

The full tier takes its scatterers from the same scene rasteriser as the fast
tier (`scene_to_scatterers` samples the renderer's reflectivity × visibility
map), so the two agree on geometry, shadowing and reflectivity by construction.

---

## 2. Scene model (`sim/scene.py`)

Coordinates: `x` along-track, `y` ground range from nadir, `z` height above
the mean seabed. The sonar flies at altitude `H` above `y = 0`.

**Terrain.** Multi-octave value noise for roughness (amplitude per sediment:
mud 4 mm, sand 1 cm, gravel 3 cm), a low-frequency component for metre-scale
relief, and optional sand ripples with an asymmetric profile
`sin φ + 0.3 sin 2φ` (steep lee face) at a random orientation and wavelength
0.3–1.6 m. Ripples are important: at low grazing angles they produce the
bright/dark banding that dominates real SAS imagery of sandy seabeds and is a
major source of false alarms.

**Objects** are height functions `z(u, v)` in a local frame:

| class | MCM meaning | primitive |
|---|---|---|
| `cylinder` | bottom mine (cylindrical) | horizontal cylinder resting on the seabed, L 1.0–2.4 m, r 0.2–0.35 m |
| `truncated_cone` | Manta-type mine | truncated cone, base r 0.4–0.6 m, h 0.3–0.5 m |
| `wedge` | Rockan-type mine | tapered wedge, L 0.9–1.3 m, h 0.3–0.45 m |
| `sphere` | spherical mine / float | sphere resting on the seabed, r 0.3–0.5 m |
| `rock` | NOMBO (non-mine bottom object) | lumpy ellipsoid |
| `debris` | NOMBO | crate or tyre |

Objects can be **partially buried** (`burial` ∈ [0.1, 0.6] of their height),
which shortens both highlight and shadow – the hardest realistic case.
Mine-like objects get higher reflectivity and a specular component (smooth
man-made surfaces); rocks are rough (Lambertian only).

---

## 3. Image-domain renderer (`sim/renderer.py`)

### 3.1 Backscatter

A synthetic aperture focuses every pixel from the sub-aperture centred on that
pixel's along-track position, so the incident direction for pixel `(x, y, h)`
lies in its own `(y, z)` plane: `ŝ = (0, −y, H − h) / ‖·‖`. With the surface
normal `n = (−h_x, −h_y, 1) / ‖·‖`,

```
cos θ_i = n · ŝ = ( y·h_y + H − h ) / ( ‖n‖ · sqrt(y² + (H − h)²) )
```

For a flat seabed this reduces to `sin ψ = H / sqrt(y² + H²)` (ψ = grazing
angle), verified by `tests/test_physics.py`. Backscattered intensity follows
Lambert's law plus a Phong-like specular term for man-made objects:

```
I = μ · cos²θ_i  +  k_s · cos^p θ_i          (p ≈ 15–50 for metal, 2–6 for rock)
```

Because `cos²θ_i ≈ (H/y)²` at long range, the object-to-seabed contrast grows
quickly with range – which is why far-range targets look like saturated
blobs with very long shadows in real data, and here.

### 3.2 Exact acoustic shadows

A pixel is illuminated iff no nearer pixel on the same across-track line
blocks the ray from the sensor. With `s(y) = (h(y) − H) / y` (tangent of the
depression angle),

```
visible(y)  ⇔  s(y) > max_{y' < y} s(y')
```

which is a running maximum along range (O(N) per line, fully vectorised).
This reproduces the textbook shadow length `L = h · y_far / (H − h)` for a box
of height `h` to within one pixel (unit-tested), handles self-shadowing of
ripples and terrain, and attributes each shadow pixel to the object that
casts it by re-casting with that object removed.

### 3.3 Coherent speckle and the system PSF

SAS imagery is coherent: each resolution cell sums many random-phase
scatterers. The renderer therefore works on a complex field at 2× the pixel
rate,

```
F = sqrt(I) · CN(0, 1)  +  sqrt(N₀) · CN(0, 1)        (signal + receiver noise)
F ← F ⊛ psf,   psf(x, y) = sinc(x/δ) sinc(y/δ) · Hann
image = |F|²  (optionally averaged over L looks), block-averaged to the pixel grid
```

so speckle has the correct exponential single-look statistics *and* the
correct spatial correlation length (one resolution cell), and shadows are
filled with noise at the right SNR rather than being painted black.

### 3.4 Range-dependent gain and platform artefacts

* residual range fall-off `(y_c / y)^γ`, γ ∈ [0, 1], and a smooth TVG ripple;
* SNR at tile centre drawn from 6–24 dB;
* along-track smear (motion-compensation residual), ping-to-ping gain
  striping, dropped pings;
* log compression with a randomised display window and head-room.

### 3.5 Labels

For each object: highlight box, shadow box, union box, highlight / shadow
area, aspect angle (object axis vs. track), slant range, grazing angle,
burial, length and a `visible` flag. The detector is trained on the
**highlight box** (its centre is the physical contact position needed for
re-acquisition), while a 3-class semantic mask (seabed / highlight / shadow)
supervises the auxiliary segmentation head so the network must learn the
shadow explicitly.

---

## 4. Full-physics tier (`sim/acoustics.py`, `sim/sas_processor.py`)

Default parameters use only *public* figures for an AquaPix MINSAS-class
sonar: `f_c` = 337 kHz, LFM bandwidth 40 kHz, 60 cm receiver array
(16 × 3.75 cm), altitude 10 m, speed 2 m/s.

### 4.1 Geometry and the SAS speed constraint

Phase centres of (tx, rx_i) pairs sit at half the element offset, spaced
`d/2` apart. Advancing `Δ = (N − k) d / 2` per ping leaves `k` **redundant
phase centres** between consecutive pings (`k = 2` by default). This fixes
the PRI `= Δ / v` and the maximum unambiguous range `c · PRI / 2` (98 m at
2 m/s) – the familiar SAS area-coverage trade-off.

### 4.2 Echo synthesis

For every ping `p`, receiver `i` and scatterer `k`:

```
τ = (|r_k − tx_p| + |r_k − rx_{p,i}|) / c
s_{p,i}(t) += A_k · w_tx(θ) w_rx(θ) · (r_ref² / r_tx r_rx) · chirp(t − τ) · exp(−j 2π f_c τ)
```

with `w(θ) = sinc(d sin θ / λ)` the element directivity. Receiver noise is
added at a configurable level. The matched filter is applied in the
frequency domain (Hann-tapered replica by default).

### 4.3 Back-projection

```
img(x, y) = Σ_{p,i}  s̃_{p,i}( τ_{p,i}(x, y) ) · exp(+j 2π f_c τ_{p,i}(x, y))
```

restricted to pixels inside the real beam of each ping. The measured point
response is **1.91 cm along-track at 20 m and 1.95 cm at 45 m** against the
theoretical `d/2 = 1.875 cm` – i.e. range-independent resolution, the
defining property of SAS (a real-aperture sonar would degrade by 2.25× over
the same range change).

### 4.4 DPCA micronavigation

Coincident phase centres of pings `n` and `n+1` see the same scatterers; an
across-track displacement `δy` changes the two-way delay by
`−2 δy cos φ / c`. The delay is estimated from the up-sampled
cross-correlation peak (parabolic refinement) and refined with the carrier
phase; integrating the increments gives the sway history. On the demo scene a
3 cm sinusoidal sway is recovered with **0.09 mm RMS error**, restoring image
contrast from 1.65 (defocused) to 2.10 (truth 2.12).

---

## 5. ATR model (`models/`)

* **Encoder**: residual CNN, strides 4–32, widths 32/64/128/192.
* **Decoder**: top-down FPN to a single stride-4 map (CenterNet style).
* **Heads** (shared 3×3 neck, cheap 1×1 outputs): class heat-map (penalty-
  reduced focal loss), size, sub-pixel offset, and an auxiliary 3-class
  highlight / shadow segmentation. Anchor-free detection suits sonar: object
  scales vary smoothly with range and there are few objects per tile.
* **Uncertainty**: `Dropout2d` in the neck enables Monte-Carlo dropout at
  inference; the per-pixel std of the heat-map across passes is an epistemic
  uncertainty map, and detection scores are checked for calibration (ECE).
* **Self-supervision**: a convolutional masked auto-encoder (60 % of 16 px
  patches masked) pre-trains the same encoder on unlabelled tiles; the
  detector can be initialised from it (`--init-from runs/mae/encoder.pt`).
* 2.1 M parameters, ~8 ms per 256² tile with ONNX Runtime on 4 CPU cores.

Sonar-specific augmentation: **along-track flips only** (an across-track flip
would put shadows on the wrong side of the object), log-domain gain / contrast /
gamma jitter, speckle re-noising, along-track smear and ping drop-out.

---

## 6. Evaluation (`train/evaluate.py`)

Besides mAP (VOC-style all-point interpolation at IoU 0.5 and COCO-style
[.5:.95]), the suite reports the metrics MCM practitioners actually use:

* **PD vs. false alarms per km²** using the known pixel size (3 cm) to convert
  tiles to surveyed area; a mine-hunting variant counts only MILCO ground
  truth and treats NOMBO-called-MILCO as false alarms.
* MILCO / NOMBO confusion (mine recall, rock rejection).
* Recall by slant range, grazing angle, burial fraction and aspect angle.
* Aspect-binned confusion matrices that feed the re-acquisition planner.
* Expected calibration error of the detection score.

---

## 7. Deployment (`deploy/`)

* `export_onnx.py` – ONNX export (opset 17, dynamic shapes) with a parity check.
* `stream.py` – `StreamingATR` consumes along-track strips as a towed sonar
  would deliver them, keeps an overlap buffer, tiles with overlap, merges
  duplicates across tiles / strips with NMS and emits time-stamped contacts.
  `benchmark_realtime` compares measured tiles/s with the sensor's area rate
  (e.g. 2 × 200 m swath at 10 kn = 2 040 m²/s ≈ 2.3 Mpixel/s at 3 cm).
* `geo.py` – contact geo-referencing from towfish navigation to lat/lon and
  GeoJSON export for mission-planning / GIS tools.

---

## 8. Autonomy (`autonomy/`)

Single-look SAS classification is aspect-dependent. `AspectConfusionModel`
stores `P(pred | true, aspect bin)` (measured on the test set), `bayesian_fuse`
combines looks, and `expected_information_gain` scores a hypothetical new look.
`plan_reacquisition` greedily selects (contact, aspect) re-look legs by
information gain per metre within a path budget, then orders them with
nearest-neighbour + 2-opt including a Dubins-style turn penalty. Each leg is a
straight track at stand-off range so the contact sits in the well-focused part
of the swath at the requested aspect.

---

## 9. Known simplifications

* No multipath (surface / bottom bounces) or volume reverberation.
* Lambert + Phong is an empirical backscatter model; no elastic target
  response (resonances) – a known limitation shared by most SAS simulators
  used for ATR data generation.
* Image-domain tier assumes perfect focus apart from the smear artefact; the
  full tier models defocus explicitly.
* Only sway is estimated by DPCA (surge / yaw are straightforward extensions
  via the along-track correlation peak).
