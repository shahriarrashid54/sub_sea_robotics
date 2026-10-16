"""Collect metrics from all runs into docs/RESULTS.md and fill the README results table.

README placeholders look like ``<!--R:key-->value<!--/R-->``; the value between
the markers is replaced so the README can be regenerated after re-training.
"""
from __future__ import annotations

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(path):
    p = os.path.join(ROOT, path)
    return json.load(open(p)) if os.path.exists(p) else None


def fmt(v, nd=3):
    return "n/a" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def main():
    base = load("runs/detector_base/eval/metrics.json")
    summ = load("runs/detector_base/summary.json")
    scratch = load("runs/detector_scratch_10pct/eval/metrics.json")
    mae10 = load("runs/detector_mae_10pct/eval/metrics.json")
    stream = load("runs/stream/stream_report.json")
    sas = load("assets/sas_demo_metrics.json")
    meta = load("data/sas_synth/dataset_meta.json")
    values = {}
    lines = ["# Results", "", "All numbers below are produced by the code in this repository on **synthetic** data",
             "(see `docs/DESIGN.md` §1 and the limitations in `docs/KRAKEN_RELEVANCE.md`). Re-generate with",
             "`python scripts/write_results.py` after `make all`.", ""]
    if meta:
        s = meta["splits"]
        lines += ["## Dataset", "", "| split | tiles | objects | per class |", "|---|---|---|---|"]
        for k, v in s.items():
            pc = ", ".join(f"{c} {n}" for c, n in v["per_class"].items())
            lines.append(f"| {k} | {v['n_images']} | {v['n_annotations']} | {pc} |")
        lines.append("")
        values["n_train"] = str(s["train"]["n_images"]); values["n_test"] = str(s["test"]["n_images"])
    if sas:
        lines += ["## Full-physics SAS chain (`scripts/demo_sas_processing.py`)", "",
                  f"* Sonar: {sas['sonar']['range_resolution_cm']:.3g} cm nominal range resolution, {sas['sonar']['along_track_resolution_cm']:.3g} cm theoretical along-track (d/2), "
                  f"{sas['sonar']['array_length_m']:.2f} m array, PRF {sas['sonar']['prf_hz']:.1f} Hz, max unambiguous range {sas['sonar']['max_unambiguous_range_m']:.0f} m",
                  f"* Measured point-target response: {sas['psf']['20m']['along_track_m'] * 100:.2f} cm along-track × {sas['psf']['20m']['range_m'] * 100:.2f} cm range at 20 m; "
                  f"{sas['psf']['45m']['along_track_m'] * 100:.2f} × {sas['psf']['45m']['range_m'] * 100:.2f} cm at 45 m (range-independent)",
                  f"* DPCA micronavigation: {sas['sway_amplitude_cm']:.0f} cm sinusoidal sway recovered with **{sas['dpca_sway_rms_mm']:.2f} mm RMS error** over {sas['n_pings']} pings",
                  f"* Image contrast: true trajectory {sas['true']['contrast']:.2f}, sway ignored {sas['nominal']['contrast']:.2f}, DPCA-refocused {sas['dpca']['contrast']:.2f}", ""]
        values["psf_along_20"] = f"{sas['psf']['20m']['along_track_m'] * 100:.2f}"
        values["psf_along_45"] = f"{sas['psf']['45m']['along_track_m'] * 100:.2f}"
        values["dpca_rms"] = f"{sas['dpca_sway_rms_mm']:.2f}"
        values["contrast_nom"] = f"{sas['nominal']['contrast']:.2f}"; values["contrast_dpca"] = f"{sas['dpca']['contrast']:.2f}"; values["contrast_true"] = f"{sas['true']['contrast']:.2f}"
    if base:
        op = base["operating_point"]; mv = base["milco_vs_nombo"]
        lines += ["## Detector (`runs/detector_base`, test split)", "",
                  f"* {summ['params_M']:.2f} M parameters, {summ['epochs']} epochs, {summ['train_time_min']:.0f} min on 4 CPU cores" if summ else "",
                  f"* **mAP@0.5 = {base['mAP@0.50']:.3f}**, mAP@[.5:.95] = {base['mAP@[.5:.95]']:.3f} on {base['n_images']} tiles / {base['n_gt']} objects ({base['area_km2'] * 1e4:.1f} ha)",
                  "* AP@0.5 per class: " + ", ".join(f"{k} {v:.3f}" for k, v in base["AP_per_class@0.50"].items()),
                  f"* Operating point (score ≥ {op['score_thr']}): PD = {op['pd']:.3f} at {op['fa_per_km2']:.0f} FA/km² (all objects); "
                  f"mine-hunting PD = {op['pd_milco']:.3f} at {op['fa_per_km2_milco']:.0f} MILCO FA/km²",
                  f"* MILCO recall {mv['milco_recall']:.3f}, NOMBO rejection {mv['nombo_rejection']:.3f}, MILCO precision {mv['milco_precision']:.3f}",
                  f"* Calibration ECE = {base['calibration']['ece']:.3f}", "",
                  "| condition | bins | recall |", "|---|---|---|"]
        for key, label in (("by_range", "slant range [m]"), ("by_grazing", "grazing [deg]"), ("by_burial", "burial fraction"), ("by_aspect", "aspect [deg]")):
            d = base[key]
            e = d["edges"]
            bins = ", ".join(f"{e[i]:g}–{e[i + 1]:g}" for i in range(len(e) - 1))
            rec = ", ".join(f"{r:.2f} (n={n})" for r, n in zip(d["recall"], d["count"]))
            lines.append(f"| {label} | {bins} | {rec} |")
        lines.append("")
        values.update({"map50": f"{base['mAP@0.50']:.3f}", "map5095": f"{base['mAP@[.5:.95]']:.3f}", "pd": f"{op['pd']:.3f}",
                       "fa": f"{op['fa_per_km2']:.0f}", "pd_milco": f"{op['pd_milco']:.3f}", "fa_milco": f"{op['fa_per_km2_milco']:.0f}",
                       "milco_recall": f"{mv['milco_recall']:.3f}", "nombo_rej": f"{mv['nombo_rejection']:.3f}", "ece": f"{base['calibration']['ece']:.3f}",
                       "params": f"{summ['params_M']:.2f}" if summ else "n/a", "train_min": f"{summ['train_time_min']:.0f}" if summ else "n/a",
                       "n_gt_test": str(base["n_gt"]), "area_ha": f"{base['area_km2'] * 1e4:.1f}"})
    hard = load("runs/detector_base/eval_hard/metrics.json")
    if base:
        # second operating point (score 0.5) from the PD/FA curve
        import numpy as np
        th = np.array(base["pd_fa"]["thresholds"]); i = int(np.argmin(np.abs(th - 0.5)))
        values["pd50"] = f"{base['pd_fa']['pd'][i]:.3f}"; values["fa50"] = f"{base['pd_fa']['fa_per_km2'][i]:.0f}"
        im = int(np.argmin(np.abs(np.array(base["pd_fa_milco"]["thresholds"]) - 0.5)))
        values["pd50_milco"] = f"{base['pd_fa_milco']['pd'][im]:.3f}"; values["fa50_milco"] = f"{base['pd_fa_milco']['fa_per_km2'][im]:.0f}"
        lines += [f"* Second operating point (score ≥ 0.5): PD = {values['pd50']} at {values['fa50']} FA/km²; mine-hunting PD = {values['pd50_milco']} at {values['fa50_milco']} MILCO FA/km²", ""]
    if hard:
        import numpy as np
        op = hard["operating_point"]; mv = hard["milco_vs_nombo"]
        th = np.array(hard["pd_fa"]["thresholds"]); i = int(np.argmin(np.abs(th - 0.5)))
        lines += ["## Out-of-distribution stress split (`test_hard`: SNR 0–10 dB, burial up to 75 %, dense rock clutter, more artefacts)", "",
                  f"* Same checkpoint, no re-training: **mAP@0.5 = {hard['mAP@0.50']:.3f}**, mAP@[.5:.95] = {hard['mAP@[.5:.95]']:.3f} on {hard['n_images']} tiles / {hard['n_gt']} visible objects",
                  f"* PD = {op['pd']:.3f} at {op['fa_per_km2']:.0f} FA/km² (score ≥ 0.3); PD = {hard['pd_fa']['pd'][i]:.3f} at {hard['pd_fa']['fa_per_km2'][i]:.0f} FA/km² (score ≥ 0.5)",
                  f"* MILCO recall {mv['milco_recall']:.3f}, NOMBO rejection {mv['nombo_rejection']:.3f}; ECE {hard['calibration']['ece']:.3f}",
                  "* Recall by burial fraction: " + ", ".join(f"{hard['by_burial']['edges'][k]:g}–{hard['by_burial']['edges'][k + 1]:g}: {r:.2f} (n={n})" for k, (r, n) in enumerate(zip(hard["by_burial"]["recall"], hard["by_burial"]["count"]))), ""]
        values.update({"hard_map50": f"{hard['mAP@0.50']:.3f}", "hard_pd": f"{op['pd']:.3f}", "hard_fa": f"{op['fa_per_km2']:.0f}",
                       "hard_pd50": f"{hard['pd_fa']['pd'][i]:.3f}", "hard_fa50": f"{hard['pd_fa']['fa_per_km2'][i]:.0f}",
                       "hard_milco_recall": f"{mv['milco_recall']:.3f}", "hard_nombo_rej": f"{mv['nombo_rejection']:.3f}"})
    if scratch and mae10:
        lines += ["## Label efficiency (10 % of labels = 240 tiles)", "", "| init | mAP@0.5 | PD @ 0.3 | MILCO recall |", "|---|---|---|---|",
                  f"| from scratch | {scratch['mAP@0.50']:.3f} | {scratch['operating_point']['pd']:.3f} | {scratch['milco_vs_nombo']['milco_recall']:.3f} |",
                  f"| MAE pre-trained encoder | {mae10['mAP@0.50']:.3f} | {mae10['operating_point']['pd']:.3f} | {mae10['milco_vs_nombo']['milco_recall']:.3f} |"]
        if base:
            lines.append(f"| 100 % labels (reference) | {base['mAP@0.50']:.3f} | {base['operating_point']['pd']:.3f} | {base['milco_vs_nombo']['milco_recall']:.3f} |")
        lines.append("")
        values.update({"map_scratch10": f"{scratch['mAP@0.50']:.3f}", "map_mae10": f"{mae10['mAP@0.50']:.3f}"})
    if stream:
        lines += ["## Streaming real-time ATR (`runs/stream`)", "",
                  f"* Swath {stream['swath_m2']:.0f} m² rendered as one waterfall, streamed in strips; {stream['tiles']} tiles of 256² with 64 px overlap",
                  f"* {stream['true_positives']}/{stream['n_gt_visible']} visible objects detected, {stream['false_alarms']} false alarms",
                  f"* ONNX Runtime on 4 CPU threads: **{stream['measured_tiles_s']:.0f} tiles/s ({stream['ms_per_tile']:.1f} ms/tile)** vs {stream['required_tiles_s']:.0f} tiles/s required for "
                  f"2 × {200:.0f} m swath at 10 kn ({stream['pixel_rate_Mpix_s']:.2f} Mpixel/s) → **real-time factor {stream['realtime_factor']:.1f}×**", ""]
        values.update({"tiles_s": f"{stream['measured_tiles_s']:.0f}", "ms_tile": f"{stream['ms_per_tile']:.1f}", "rt_factor": f"{stream['realtime_factor']:.1f}",
                       "stream_tp": str(stream["true_positives"]), "stream_gt": str(stream["n_gt_visible"]), "stream_fa": str(stream["false_alarms"])})
    with open(os.path.join(ROOT, "docs", "RESULTS.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    rp = os.path.join(ROOT, "README.md")
    if os.path.exists(rp):
        txt = open(rp).read()
        n = 0
        for k, v in values.items():
            txt, c = re.subn(rf"(<!--R:{k}-->)(.*?)(<!--/R-->)", lambda m: m.group(1) + v + m.group(3), txt)
            n += c
        open(rp, "w").write(txt)
        print(f"README: {n} placeholders filled")
    print("docs/RESULTS.md written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
