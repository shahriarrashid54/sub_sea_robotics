"""Procedural noise utilities (fractal / multi-octave value noise)."""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def value_noise(shape, rng: np.random.Generator, cells: int = 8) -> np.ndarray:
    """Smooth random field by bilinear up-sampling a coarse Gaussian grid."""
    h, w = shape
    ch, cw = max(2, int(round(h / shape[0] * cells))), max(2, int(round(w / shape[1] * cells)))
    coarse = rng.standard_normal((ch + 1, cw + 1))
    zoom = (h / (ch + 1), w / (cw + 1))
    fine = ndimage.zoom(coarse, zoom, order=3, mode="nearest")
    return fine[:h, :w]


def fractal_noise(shape, rng: np.random.Generator, octaves: int = 5, persistence: float = 0.55,
                  base_cells: int = 4) -> np.ndarray:
    """Sum of octaves of value noise, normalised to unit standard deviation."""
    out = np.zeros(shape, dtype=np.float64)
    amp, cells = 1.0, base_cells
    for _ in range(octaves):
        out += amp * value_noise(shape, rng, cells=cells)
        amp *= persistence
        cells *= 2
    out -= out.mean()
    std = out.std()
    return out / std if std > 0 else out


def smooth_1d(n: int, rng: np.random.Generator, sigma: float) -> np.ndarray:
    """Smooth zero-mean, unit-std 1-D random profile."""
    x = rng.standard_normal(n)
    x = ndimage.gaussian_filter1d(x, sigma, mode="reflect")
    x -= x.mean()
    s = x.std()
    return x / s if s > 0 else x
