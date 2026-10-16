"""SAS image formation (time-domain back-projection) and DPCA micronavigation.

Back-projection is the exact, geometry-agnostic beamformer: for every image
pixel and every (ping, receiver) pair the two-way delay is computed from the
*estimated* platform trajectory, the range-compressed echo is interpolated at
that delay, phase-rotated back to baseband and coherently accumulated. It is
slower than wavenumber-domain algorithms but tolerates arbitrary motion, which
is what makes it the reference algorithm for motion-compensation studies.

DPCA (Displaced Phase Centre Antenna) micronavigation exploits the redundant
phase centres between consecutive pings: echoes recorded at coincident phase
centres differ only by the platform's unknown inter-ping displacement, which
is estimated by cross-correlation and integrated into a trajectory estimate.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .acoustics import SASConfig


def backproject(compressed: np.ndarray, t0: float, cfg: SASConfig, traj: Dict[str, np.ndarray],
                x_grid: np.ndarray, y_grid: np.ndarray, z0: float = 0.0, beam_limit: float = 1.0,
                verbose: bool = False) -> np.ndarray:
    """Time-domain back-projection onto a ground-plane grid.

    Parameters
    ----------
    compressed : [P, N, M] range-compressed complex data.
    traj       : trajectory *used for focusing* (truth, nominal or estimated).
    x_grid, y_grid : 1-D along-track / ground-range pixel coordinates (metres).
    beam_limit : include a pixel only if its angle off broadside is within
        ``beam_limit * beamwidth / 2`` of the element pattern (processing aperture).
    """
    tx, rx = traj["tx"], traj["rx"]
    P, N, M = compressed.shape
    X, Y = np.meshgrid(x_grid, y_grid, indexing="ij")
    img = np.zeros(X.shape, dtype=np.complex128)
    sample_idx = np.arange(M)
    half_beam = beam_limit * cfg.beamwidth_rad / 2
    for p in range(P):
        dy = Y - tx[p, 1]
        dz = z0 - tx[p, 2]
        r_tx = np.sqrt((X - tx[p, 0]) ** 2 + dy**2 + dz**2)
        # processing aperture: pixels within the beam of this ping
        in_beam = np.abs(np.arctan2(X - tx[p, 0], np.sqrt(dy**2 + dz**2))) <= half_beam
        if not in_beam.any():
            continue
        rows = np.where(in_beam.any(axis=1))[0]
        rs = slice(rows[0], rows[-1] + 1)
        Xs, r_txs, dys = X[rs], r_tx[rs], dy[rs]
        for i in range(N):
            r_rx = np.sqrt((Xs - rx[p, i, 0]) ** 2 + dys**2 + dz**2)
            tau = (r_txs + r_rx) / cfg.c
            m = (tau - t0) * cfg.fs
            sig = compressed[p, i]
            re = np.interp(m.ravel(), sample_idx, sig.real, left=0.0, right=0.0)
            im = np.interp(m.ravel(), sample_idx, sig.imag, left=0.0, right=0.0)
            contrib = (re + 1j * im).reshape(m.shape) * np.exp(2j * np.pi * cfg.fc * tau)
            img[rs] += contrib * in_beam[rs]
        if verbose and p % 10 == 0:
            print(f"  focus ping {p + 1}/{P}", flush=True)
    return img


def to_db(img: np.ndarray, floor_db: float = -60.0) -> np.ndarray:
    p = np.abs(img) ** 2
    p = p / max(p.max(), 1e-30)
    return np.clip(10 * np.log10(p + 1e-30), floor_db, 0)


# ---------------------------------------------------------------------------------
# DPCA micronavigation
# ---------------------------------------------------------------------------------

def _xcorr_delay(a: np.ndarray, b: np.ndarray, fs: float, upsample: int = 16) -> Tuple[float, float]:
    """Delay of ``b`` relative to ``a`` (seconds) from the up-sampled cross-correlation
    peak with parabolic refinement. Returns (delay, normalised peak coherence)."""
    n = len(a)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    A = np.fft.fft(a, nfft)
    B = np.fft.fft(b, nfft)
    X = B * np.conj(A)
    # zero-pad the spectrum for interpolation
    Xp = np.zeros(nfft * upsample, dtype=complex)
    h = nfft // 2
    Xp[:h] = X[:h]
    Xp[-h:] = X[-h:]
    xc = np.fft.ifft(Xp) * upsample
    mag = np.abs(xc)
    k = int(np.argmax(mag))
    # parabolic interpolation on the magnitude
    km, kp = (k - 1) % len(mag), (k + 1) % len(mag)
    denom = mag[km] - 2 * mag[k] + mag[kp]
    delta = 0.5 * (mag[km] - mag[kp]) / denom if abs(denom) > 1e-12 else 0.0
    lag = k + delta
    if lag > len(mag) / 2:
        lag -= len(mag)
    coh = mag[k] / max(np.sqrt(np.sum(np.abs(a) ** 2) * np.sum(np.abs(b) ** 2)), 1e-30)
    return lag / (fs * upsample), float(coh)


def dpca_estimate_sway(compressed: np.ndarray, t0: float, cfg: SASConfig, gate: Optional[Tuple[int, int]] = None,
                       fine_phase: bool = True) -> Dict[str, np.ndarray]:
    """Estimate inter-ping sway from redundant phase centres.

    For consecutive pings n and n+1 the last ``overlap_pc`` phase centres of
    ping n coincide (nominally) with the first ``overlap_pc`` of ping n+1. Any
    across-track displacement dy between the pings changes the two-way delay of
    the coincident echoes by ``-2 dy cos(phi) / c`` where ``phi`` is the angle
    between the across-track axis and the slant direction at the range gate.

    Returns the per-ping sway increment, the integrated sway (relative to ping
    0) and the mean coherence of each estimate.
    """
    P, N, M = compressed.shape
    k = cfg.overlap_pc
    if gate is None:
        gate = (M // 4, 3 * M // 4)
    g0, g1 = gate
    # slant geometry at the gate centre
    tau_c = t0 + 0.5 * (g0 + g1) / cfg.fs
    r_c = cfg.c * tau_c / 2
    y_c = np.sqrt(max(r_c**2 - cfg.altitude**2, 1e-6))
    cos_phi = y_c / r_c
    inc = np.zeros(P)
    coh = np.zeros(P)
    for n in range(P - 1):
        d_est, c_est = [], []
        for j in range(k):
            a = compressed[n, N - k + j, g0:g1]
            b = compressed[n + 1, j, g0:g1]
            dt, c = _xcorr_delay(a, b, cfg.fs)
            if fine_phase:
                # refine with the carrier phase: wrap the residual onto the coarse estimate
                phase = np.angle(np.sum(b * np.conj(a) * np.exp(2j * np.pi * cfg.fc * dt)))
                dt = dt - phase / (2 * np.pi * cfg.fc)
            d_est.append(dt)
            c_est.append(c)
        dt_mean = float(np.mean(d_est))
        inc[n + 1] = -cfg.c * dt_mean / (2 * cos_phi)
        coh[n + 1] = float(np.mean(c_est))
    sway = np.cumsum(inc)
    return {"increment": inc, "sway": sway, "coherence": coh, "cos_phi": cos_phi}


def trajectory_from_sway(cfg: SASConfig, n_pings: int, sway: np.ndarray, x0: float = 0.0) -> Dict[str, np.ndarray]:
    from .acoustics import platform_trajectory
    return platform_trajectory(cfg, n_pings, x0=x0, sway=sway)


# ---------------------------------------------------------------------------------
# Image quality metrics
# ---------------------------------------------------------------------------------

def image_contrast(img: np.ndarray) -> float:
    p = np.abs(img) ** 2
    return float(p.std() / max(p.mean(), 1e-30))


def image_entropy(img: np.ndarray) -> float:
    p = np.abs(img) ** 2
    p = p / max(p.sum(), 1e-30)
    return float(-np.sum(p * np.log(p + 1e-30)))


def measure_resolution(img: np.ndarray, x_grid: np.ndarray, y_grid: np.ndarray) -> Dict[str, float]:
    """-3 dB widths (metres) of the brightest point response along both axes."""
    p = np.abs(img) ** 2
    i, j = np.unravel_index(np.argmax(p), p.shape)
    half = p[i, j] / 2

    def width(profile, coords):
        above = profile >= half
        # contiguous region around the peak
        k = int(np.argmax(profile))
        lo = k
        while lo > 0 and above[lo - 1]:
            lo -= 1
        hi = k
        while hi < len(profile) - 1 and above[hi + 1]:
            hi += 1
        d = coords[1] - coords[0]
        # linear interpolation at the edges
        def edge(a, b):
            if b < 0 or b >= len(profile) or profile[a] == profile[b]:
                return coords[a]
            f = (profile[a] - half) / (profile[a] - profile[b])
            return coords[a] + f * (coords[b] - coords[a])
        return float(edge(hi, hi + 1) - edge(lo, lo - 1)) if hi > lo else float(d)

    return {"along_track_m": width(p[:, j], x_grid), "range_m": width(p[i, :], y_grid),
            "peak_x": float(x_grid[i]), "peak_y": float(y_grid[j])}
