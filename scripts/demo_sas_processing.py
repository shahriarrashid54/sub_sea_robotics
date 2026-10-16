"""Full-physics SAS demonstration: pulse-level echo simulation, matched filtering,
back-projection image formation and DPCA micronavigation / autofocus.

Produces ``assets/sas_image_formation.png``, ``assets/sas_psf.png`` and
``assets/sas_demo_metrics.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from seabed_sentinel.sim.acoustics import (  # noqa: E402
    SASConfig,
    platform_trajectory,
    range_compress,
    scene_to_scatterers,
    simulate_echoes,
    sinusoidal_sway,
)
from seabed_sentinel.sim.sas_processor import (  # noqa: E402
    backproject,
    dpca_estimate_sway,
    image_contrast,
    image_entropy,
    measure_resolution,
    to_db,
    trajectory_from_sway,
)
from seabed_sentinel.sim.scene import Scene, SceneObject, SonarGeometry  # noqa: E402
from seabed_sentinel.viz.plots import sonar_cmap  # noqa: E402


def build_scene(cfg: SASConfig) -> Scene:
    geom = SonarGeometry(pixel_m=0.02, altitude_m=cfg.altitude, range_start_m=24.0, n_along=250, n_range=300, fine_factor=2)
    objs = [
        SceneObject("cylinder", x=2.6, y=26.6, yaw_deg=65.0, dims={"radius": 0.25, "length": 1.6},
                    reflectivity=3.5, specular=3.0, spec_power=25.0, seed=1),
        SceneObject("rock", x=1.2, y=28.9, yaw_deg=20.0, dims={"a": 0.45, "b": 0.3, "height": 0.3},
                    reflectivity=2.2, specular=0.3, spec_power=4.0, seed=2),
    ]
    terrain = {"roughness": 0.008, "ripple_amp": 0.02, "ripple_wavelength": 0.9, "ripple_dir_deg": 30.0,
               "mu": 1.0, "mu_texture": 0.3, "large_scale_amp": 0.0}
    return Scene(geometry=geom, sediment="sand", objects=objs, clutter=[], terrain_params=terrain, seed=11)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="assets")
    ap.add_argument("--max-scatterers", type=int, default=22000)
    ap.add_argument("--sway-cm", type=float, default=3.0)
    ap.add_argument("--pixel-cm", type=float, default=2.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cfg = SASConfig()
    rng = np.random.default_rng(0)
    scene = build_scene(cfg)
    g = scene.geometry
    print("scene -> scatterers ...", flush=True)
    scat = scene_to_scatterers(scene, max_scatterers=args.max_scatterers, rng=rng)
    print(f"  {len(scat['x'])} scatterers ({scat['is_object'].sum()} on objects)")

    # aperture: cover the scene plus one synthetic aperture length on each side
    r_far = np.hypot(g.range_start_m + g.range_extent_m, cfg.altitude)
    L = cfg.synthetic_aperture_length(r_far)
    n_pings = int(np.ceil((g.along_extent_m + L) / cfg.ping_advance)) + 2
    x0 = -L / 2
    sway_true = sinusoidal_sway(n_pings, amplitude=args.sway_cm / 100, period_pings=n_pings / 1.3, phase=0.4)
    traj_true = platform_trajectory(cfg, n_pings, x0=x0, sway=sway_true)
    traj_nominal = platform_trajectory(cfg, n_pings, x0=x0)
    print(f"simulating {n_pings} pings x {cfg.n_rx} receivers ...", flush=True)
    t0 = time.time()
    ech = simulate_echoes(cfg, scat, traj_true, rng=rng, verbose=True)
    comp = range_compress(ech, cfg)
    t_sim = time.time() - t0
    print(f"  echo simulation + matched filter: {t_sim:.1f}s  raw data {comp.shape}")

    dx = args.pixel_cm / 100
    xg = np.arange(0, g.along_extent_m, dx)
    yg = np.arange(g.range_start_m, g.range_start_m + g.range_extent_m, dx)
    print("DPCA micronavigation ...", flush=True)
    est = dpca_estimate_sway(comp, ech["t0"], cfg)
    sway_est = est["sway"] - est["sway"][0] + sway_true[0]
    traj_est = trajectory_from_sway(cfg, n_pings, sway_est, x0=x0)
    rms_mm = float(np.sqrt(np.mean((sway_est - sway_true) ** 2)) * 1e3)
    print(f"  sway RMS error {rms_mm:.2f} mm (true amplitude {args.sway_cm} cm), mean coherence {est['coherence'][1:].mean():.3f}")

    images = {}
    for name, tr in (("true", traj_true), ("nominal", traj_nominal), ("dpca", traj_est)):
        t0 = time.time()
        images[name] = backproject(comp, ech["t0"], cfg, tr, xg, yg)
        print(f"  back-projection [{name}]: {time.time() - t0:.1f}s", flush=True)
    metrics = {name: {"contrast": image_contrast(im), "entropy": image_entropy(im)} for name, im in images.items()}
    metrics["dpca_sway_rms_mm"] = rms_mm
    metrics["sway_amplitude_cm"] = args.sway_cm
    metrics["n_pings"] = n_pings
    metrics["n_scatterers"] = int(len(scat["x"]))
    metrics["sonar"] = cfg.summary()

    # ---------------- figure 1: image formation -----------------------------------
    fine = scat["render"].meta["_fine"]
    truth = fine["intensity"] * fine["visibility"]
    cmap = sonar_cmap()
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    ext = [yg[0], yg[-1], xg[0], xg[-1]]
    ax = axes[0, 0]
    ax.imshow(10 * np.log10(truth + 1e-6), cmap=cmap, origin="lower", extent=[fine["Y"].min(), fine["Y"].max(), fine["X"].min(), fine["X"].max()], vmin=-25, vmax=5)
    ax.set_title("(a) Scene truth: reflectivity × visibility (ray-cast shadows)", fontsize=9)
    ax = axes[0, 1]
    mid = cfg.n_rx // 2
    rc = np.abs(comp[:, mid, :])
    tau = ech["t0"] + np.arange(comp.shape[-1]) / cfg.fs
    rg = np.sqrt(np.clip((cfg.c * tau / 2) ** 2 - cfg.altitude**2, 0, None))
    ax.imshow(20 * np.log10(rc / rc.max() + 1e-6), cmap=cmap, origin="lower", aspect="auto",
              extent=[rg[0], rg[-1], traj_true["tx"][0, 0], traj_true["tx"][-1, 0]], vmin=-50, vmax=0)
    ax.set_xlim(yg[0] - 1, yg[-1] + 1)
    ax.set_title("(b) Range-compressed echoes, single receiver (unfocused, real-aperture)", fontsize=9)
    ax.set_xlabel("ground range [m]"); ax.set_ylabel("platform along-track [m]")
    ax = axes[0, 2]
    ax.imshow(to_db(images["true"], -45), cmap=cmap, origin="lower", extent=ext, vmin=-45, vmax=0)
    ax.set_title(f"(c) SAS back-projection, true trajectory ({args.pixel_cm:.0f} cm pixels)", fontsize=9)
    ax = axes[1, 0]
    ax.imshow(to_db(images["nominal"], -45), cmap=cmap, origin="lower", extent=ext, vmin=-45, vmax=0)
    ax.set_title(f"(d) {args.sway_cm:.0f} cm sway ignored (straight-line assumption): defocused", fontsize=9)
    ax = axes[1, 1]
    ax.imshow(to_db(images["dpca"], -45), cmap=cmap, origin="lower", extent=ext, vmin=-45, vmax=0)
    ax.set_title(f"(e) Refocused with DPCA-estimated trajectory (RMS err {rms_mm:.1f} mm)", fontsize=9)
    for a in (axes[0, 0], axes[0, 2], axes[1, 0], axes[1, 1]):
        a.set_xlabel("ground range [m]"); a.set_ylabel("along-track [m]")
    ax = axes[1, 2]
    n = np.arange(n_pings)
    ax.plot(n, sway_true * 100, "k-", lw=2, label="true sway")
    ax.plot(n, sway_est * 100, "r--", lw=1.5, label="DPCA estimate")
    ax.plot(n, (sway_est - sway_true) * 1000, "b:", lw=1, label="error [mm]")
    ax.set_xlabel("ping"); ax.set_ylabel("sway [cm]  /  error [mm]")
    ax.set_title("(f) DPCA micronavigation: redundant phase-centre correlation", fontsize=9)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle(f"Full-physics SAS chain: {cfg.fc / 1e3:.0f} kHz, {cfg.bandwidth / 1e3:.0f} kHz LFM, {cfg.n_rx}×{cfg.d_rx * 100:.2f} cm array, "
                 f"{n_pings} pings, {len(scat['x'])} scatterers", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "sas_image_formation.png"), dpi=110)

    # ---------------- figure 2: point-spread function -----------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    psf_metrics = {}
    for k, y_t in enumerate((20.0, 45.0)):
        Lk = cfg.synthetic_aperture_length(np.hypot(y_t, cfg.altitude))
        npk = int(np.ceil(Lk / cfg.ping_advance)) + 6
        trk = platform_trajectory(cfg, npk, x0=-npk * cfg.ping_advance / 2)
        pt = {"x": np.array([0.0]), "y": np.array([y_t]), "z": np.array([0.0]), "amp": np.array([1.0 + 0j])}
        e = simulate_echoes(cfg, pt, trk, rng=np.random.default_rng(1))
        c = range_compress(e, cfg)
        xs = np.arange(-0.12, 0.12, 0.002); ys = y_t + np.arange(-0.12, 0.12, 0.002)
        im = backproject(c, e["t0"], cfg, trk, xs, ys)
        r = measure_resolution(im, xs, ys)
        psf_metrics[f"{y_t:.0f}m"] = r
        ax = axes[k]
        ax.imshow(to_db(im, -40), cmap=cmap, origin="lower", extent=[(ys[0] - y_t) * 100, (ys[-1] - y_t) * 100, xs[0] * 100, xs[-1] * 100], vmin=-40, vmax=0)
        ax.set_title(f"Point target at {y_t:.0f} m: {r['along_track_m'] * 100:.2f} cm along × {r['range_m'] * 100:.2f} cm range (-3 dB)", fontsize=9)
        ax.set_xlabel("range offset [cm]"); ax.set_ylabel("along-track offset [cm]")
        p = np.abs(im) ** 2
        i, j = np.unravel_index(np.argmax(p), p.shape)
        axes[2].plot(xs * 100, 10 * np.log10(p[:, j] / p.max()), label=f"along-track cut @ {y_t:.0f} m")
    axes[2].axhline(-3, color="gray", ls="--", lw=1)
    axes[2].axvline(-cfg.along_track_resolution * 50, color="k", ls=":", lw=1); axes[2].axvline(cfg.along_track_resolution * 50, color="k", ls=":", lw=1, label="theory d/2")
    axes[2].set_ylim(-40, 1); axes[2].set_xlabel("along-track offset [cm]"); axes[2].set_ylabel("dB"); axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)
    axes[2].set_title("Range-independent along-track resolution", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "sas_psf.png"), dpi=110)
    metrics["psf"] = psf_metrics
    with open(os.path.join(args.out, "sas_demo_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1, default=float)
    print(json.dumps({k: v for k, v in metrics.items() if k != "sonar"}, indent=1, default=float))


if __name__ == "__main__":
    main()
