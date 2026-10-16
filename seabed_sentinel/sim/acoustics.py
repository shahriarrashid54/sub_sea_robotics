"""Pulse-level acoustic echo simulation for a multi-element synthetic aperture sonar.

This is the *full-physics* tier of the simulator. A seabed scene is converted
into a cloud of point scatterers (with exact ray-cast shadowing), a linear-FM
chirp is transmitted from a moving platform, and the complex baseband echo at
every receiver element on every ping is synthesised by coherent superposition.
The result is range-compressed by a matched filter and can be focused with the
back-projection beamformer in :mod:`seabed_sentinel.sim.sas_processor`.

Default parameters are chosen to be representative of a commercial miniature
SAS of the AquaPix MINSAS class (337 kHz centre frequency, 40 kHz LFM
bandwidth, 60 cm receiver array) using only publicly available figures.

Conventions
-----------
* Platform travels along +x at height ``altitude`` above the seabed (z = 0).
* The transmitter sits at the along-track centre of the receiver array.
* Receiver element ``i`` is at along-track offset ``(i - (N-1)/2) * d_rx``.
* The phase centre of the pair (tx, rx_i) is at half that offset, so phase
  centres are spaced ``d_rx / 2`` apart. Advancing the platform by
  ``(N - overlap) * d_rx / 2`` per ping makes the last ``overlap`` phase
  centres of ping ``n`` coincide with the first ``overlap`` of ping ``n+1``
  (redundant phase centres used by DPCA micronavigation).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from scipy import signal

from .renderer import RenderOptions, render_scene
from .scene import Scene


@dataclass
class SASConfig:
    fc: float = 337e3            # centre frequency [Hz]
    bandwidth: float = 40e3      # LFM bandwidth [Hz]
    pulse_length: float = 2e-3   # [s]
    c: float = 1500.0            # sound speed [m/s]
    fs: float = 60e3             # complex baseband sample rate [Hz]
    n_rx: int = 16               # receiver elements
    d_rx: float = 0.0375         # receiver element length [m] (16 x 3.75 cm = 60 cm array)
    d_tx: float = 0.0375         # transmitter length [m]
    speed: float = 2.0           # platform speed [m/s]
    altitude: float = 10.0       # [m]
    overlap_pc: int = 2          # redundant phase centres per ping
    noise_db: float = -25.0      # receiver noise power relative to mean echo power
    range_taper: bool = True     # Hann window on the matched filter (sidelobe control)

    @property
    def wavelength(self) -> float:
        return self.c / self.fc

    @property
    def range_resolution(self) -> float:
        """Nominal slant-range resolution c / (2B)."""
        return self.c / (2 * self.bandwidth)

    @property
    def along_track_resolution(self) -> float:
        """Theoretical SAS along-track resolution d / 2 (independent of range)."""
        return self.d_rx / 2

    @property
    def ping_advance(self) -> float:
        return (self.n_rx - self.overlap_pc) * self.d_rx / 2

    @property
    def pri(self) -> float:
        return self.ping_advance / self.speed

    @property
    def max_unambiguous_range(self) -> float:
        return self.c * self.pri / 2

    @property
    def beamwidth_rad(self) -> float:
        return self.wavelength / self.d_rx

    def rx_offsets(self) -> np.ndarray:
        return (np.arange(self.n_rx) - (self.n_rx - 1) / 2) * self.d_rx

    def phase_centre_offsets(self) -> np.ndarray:
        return self.rx_offsets() / 2

    def synthetic_aperture_length(self, slant_range: float) -> float:
        return slant_range * self.beamwidth_rad

    def summary(self) -> Dict[str, float]:
        return {
            "wavelength_mm": self.wavelength * 1e3,
            "range_resolution_cm": self.range_resolution * 1e2,
            "along_track_resolution_cm": self.along_track_resolution * 1e2,
            "array_length_m": self.n_rx * self.d_rx,
            "ping_advance_m": self.ping_advance,
            "pri_s": self.pri,
            "prf_hz": 1 / self.pri,
            "max_unambiguous_range_m": self.max_unambiguous_range,
            "beamwidth_deg": np.degrees(self.beamwidth_rad),
        }


# ---------------------------------------------------------------------------------
# Waveform
# ---------------------------------------------------------------------------------

def chirp_baseband(cfg: SASConfig) -> np.ndarray:
    """Complex baseband LFM chirp sweeping -B/2 .. +B/2 over the pulse length."""
    n = int(round(cfg.pulse_length * cfg.fs))
    t = np.arange(n) / cfg.fs
    k = cfg.bandwidth / cfg.pulse_length
    return np.exp(1j * np.pi * k * (t - cfg.pulse_length / 2) ** 2)


# ---------------------------------------------------------------------------------
# Scene -> scatterers
# ---------------------------------------------------------------------------------

def scene_to_scatterers(scene: Scene, max_scatterers: int = 30000, rng: Optional[np.random.Generator] = None,
                        keep_all_object_pixels: bool = True) -> Dict[str, np.ndarray]:
    """Sample a shadow-aware point-scatterer cloud from a scene.

    Diffuse seabed pixels are randomly sub-sampled (with power conserved) while
    object pixels are kept in full so that highlight structure is preserved.
    """
    rng = rng or np.random.default_rng(scene.seed)
    res = render_scene(scene, RenderOptions(speckle=False, p_dropout=0, p_striping=0, p_smear=0,
                                            snr_db_range=(60, 60), tvg_residual_max=0,
                                            range_falloff_gamma_range=(0, 0)), return_fine=True)
    fine = res.meta["_fine"]
    I, vis, h, X, Y, lab = (fine[k] for k in ("intensity", "visibility", "height", "X", "Y", "label"))
    amp = np.sqrt(np.clip(I, 0, None)) * vis
    is_obj = lab != -1
    idx_obj = np.flatnonzero(is_obj & (amp > 0))
    idx_bg = np.flatnonzero(~is_obj & (amp > 0))
    n_bg = max(0, max_scatterers - (len(idx_obj) if keep_all_object_pixels else 0))
    if len(idx_bg) > n_bg:
        sel = rng.choice(idx_bg, n_bg, replace=False)
        scale_bg = np.sqrt(len(idx_bg) / n_bg)
    else:
        sel, scale_bg = idx_bg, 1.0
    idx = np.concatenate([idx_obj, sel]) if keep_all_object_pixels else sel
    a = amp.ravel()[idx].astype(np.float64)
    a[len(idx_obj):] *= scale_bg
    phase = np.exp(1j * rng.uniform(0, 2 * np.pi, len(idx)))
    return {"x": X.ravel()[idx], "y": Y.ravel()[idx], "z": h.ravel()[idx], "amp": a * phase,
            "is_object": np.r_[np.ones(len(idx_obj), bool), np.zeros(len(sel), bool)] if keep_all_object_pixels
            else np.zeros(len(sel), bool), "render": res}


# ---------------------------------------------------------------------------------
# Platform trajectory
# ---------------------------------------------------------------------------------

def platform_trajectory(cfg: SASConfig, n_pings: int, x0: float = 0.0, sway: Optional[np.ndarray] = None,
                        surge_err: Optional[np.ndarray] = None, heave: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Nominal straight-line trajectory plus optional motion errors (metres)."""
    n = np.arange(n_pings)
    x = x0 + n * cfg.ping_advance + (surge_err if surge_err is not None else 0.0)
    y = np.zeros(n_pings) + (sway if sway is not None else 0.0)
    z = np.full(n_pings, cfg.altitude) + (heave if heave is not None else 0.0)
    tx = np.stack([x, y, z], 1)                                   # [P, 3]
    rx = tx[:, None, :] + np.stack([cfg.rx_offsets(), np.zeros(cfg.n_rx), np.zeros(cfg.n_rx)], 1)[None]  # [P, N, 3]
    return {"tx": tx, "rx": rx}


def sinusoidal_sway(n_pings: int, amplitude: float = 0.03, period_pings: float = 40.0, phase: float = 0.3,
                    drift: float = 0.0) -> np.ndarray:
    n = np.arange(n_pings)
    return amplitude * np.sin(2 * np.pi * n / period_pings + phase) + drift * n


# ---------------------------------------------------------------------------------
# Echo synthesis
# ---------------------------------------------------------------------------------

def simulate_echoes(cfg: SASConfig, scat: Dict[str, np.ndarray], traj: Dict[str, np.ndarray],
                    rng: Optional[np.random.Generator] = None, r_ref: Optional[float] = None,
                    chunk: int = 4000, verbose: bool = False) -> Dict[str, np.ndarray]:
    """Synthesise complex baseband echoes for every (ping, receiver).

    Returns ``raw`` [P, N, M] complex64, the sample-time origin ``t0`` and the
    chirp replica. Each scatterer contributes
    ``A_k * w_k * p(t - tau_k) * exp(-j 2 pi fc tau_k)`` where ``w_k`` is the
    two-way element directivity and ``A_k`` includes spherical spreading
    (normalised to a reference range, i.e. ideal TVG).
    """
    rng = rng or np.random.default_rng(0)
    tx, rx = traj["tx"], traj["rx"]
    P, N = rx.shape[:2]
    px, py, pz, amp = scat["x"], scat["y"], scat["z"], scat["amp"]
    # time axis covering all two-way delays plus one pulse
    d_all = np.sqrt((px[None] - tx[:, 0:1]) ** 2 + (py[None] - tx[:, 1:2]) ** 2 + (pz[None] - tx[:, 2:3]) ** 2)
    r_min, r_max = d_all.min() - cfg.n_rx * cfg.d_rx, d_all.max() + cfg.n_rx * cfg.d_rx
    t0 = 2 * r_min / cfg.c - cfg.pulse_length
    t_end = 2 * r_max / cfg.c + cfg.pulse_length
    M = int(np.ceil((t_end - t0) * cfg.fs)) + 1
    replica = chirp_baseband(cfg)
    Lp = len(replica)
    r_ref = r_ref or float(np.median(d_all))
    k_chirp = cfg.bandwidth / cfg.pulse_length
    raw = np.zeros((P, N, M), dtype=np.complex64)
    lam = cfg.wavelength
    m_local = np.arange(Lp)
    for p in range(P):
        r_tx = np.sqrt((px - tx[p, 0]) ** 2 + (py - tx[p, 1]) ** 2 + (pz - tx[p, 2]) ** 2)
        s_tx = (px - tx[p, 0]) / r_tx
        w_tx = np.sinc(cfg.d_tx * s_tx / lam)
        for i in range(N):
            r_rx = np.sqrt((px - rx[p, i, 0]) ** 2 + (py - rx[p, i, 1]) ** 2 + (pz - rx[p, i, 2]) ** 2)
            s_rx = (px - rx[p, i, 0]) / r_rx
            w = w_tx * np.sinc(cfg.d_rx * s_rx / lam)
            tau = (r_tx + r_rx) / cfg.c
            A = amp * w * (r_ref**2 / (r_tx * r_rx)) * np.exp(-2j * np.pi * cfg.fc * tau)
            acc_re = np.zeros(M + Lp + 2)
            acc_im = np.zeros(M + Lp + 2)
            for c0 in range(0, len(tau), chunk):
                sl = slice(c0, c0 + chunk)
                start = (tau[sl] - t0) * cfg.fs                       # fractional start sample
                m0 = np.ceil(start).astype(int)
                frac = (m0 - start) / cfg.fs                           # time from chirp start to first sample
                tt = frac[:, None] + m_local[None, :] / cfg.fs        # [K, Lp] times within the pulse
                valid = tt < cfg.pulse_length
                pulse = np.exp(1j * np.pi * k_chirp * (tt - cfg.pulse_length / 2) ** 2) * valid
                vals = A[sl][:, None] * pulse
                idx = (m0[:, None] + m_local[None, :]).ravel()
                ok = (idx >= 0) & (idx < M)
                acc_re += np.bincount(idx[ok], weights=vals.real.ravel()[ok], minlength=M + Lp + 2)
                acc_im += np.bincount(idx[ok], weights=vals.imag.ravel()[ok], minlength=M + Lp + 2)
            raw[p, i] = (acc_re[:M] + 1j * acc_im[:M]).astype(np.complex64)
        if verbose and (p % 10 == 0):
            print(f"  ping {p + 1}/{P}", flush=True)
    sig_pow = float(np.mean(np.abs(raw) ** 2))
    n_pow = sig_pow * 10 ** (cfg.noise_db / 10)
    raw += (np.sqrt(n_pow / 2) * (rng.standard_normal(raw.shape) + 1j * rng.standard_normal(raw.shape))).astype(np.complex64)
    return {"raw": raw, "t0": t0, "replica": replica, "fs": cfg.fs}


def range_compress(echoes: Dict[str, np.ndarray], cfg: SASConfig) -> np.ndarray:
    """Matched filter (pulse compression). Output sample m corresponds to two-way
    delay ``t0 + m / fs`` and retains the carrier phase ``exp(-j 2 pi fc tau)``."""
    raw, replica = echoes["raw"], echoes["replica"]
    Lp = len(replica)
    if cfg.range_taper:
        replica = replica * np.hanning(Lp)
    h = np.conj(replica[::-1])
    P, N, M = raw.shape
    flat = raw.reshape(P * N, M)
    out = signal.fftconvolve(flat, h[None, :], mode="full", axes=1)[:, Lp - 1: Lp - 1 + M]
    return (out / np.sqrt(np.sum(np.abs(replica) ** 2))).reshape(P, N, M).astype(np.complex64)
