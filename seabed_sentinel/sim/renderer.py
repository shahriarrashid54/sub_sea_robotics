"""Fast physics-based image-domain SAS renderer.

The renderer is the *image-domain approximation* of the full acoustic model in
:mod:`seabed_sentinel.sim.acoustics`: it keeps the physics that dominate the
appearance of SAS imagery (Lambertian + specular backscatter as a function of
the local grazing angle, exact ray-cast acoustic shadows, coherent speckle with
the correct resolution-cell correlation, range-dependent SNR / TVG residuals,
and typical platform artefacts) but skips pulse-level echo simulation so that
thousands of labelled tiles can be produced per minute on a CPU.

Pipeline (per tile, on a ``fine_factor``-times oversampled grid)::

    terrain + objects -> heightmap h(x,y)
    -> local grazing geometry (sensor at y=0, altitude H)
    -> backscatter I = mu * cos^2(theta_i) + k_s * cos^p(theta_i)   (Lambert + Phong)
    -> visibility by ray casting along each across-track line (exact shadows)
    -> complex field  F = sqrt(I) * CN(0,1) + noise
    -> convolution with the system point-spread function (sinc x sinc)
    -> |F|^2, incoherent multi-look averaging, block down-sample to pixel grid
    -> TVG residual, ping drop-outs, gain striping, motion-residual smear
    -> log-compression and dynamic-range normalisation
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage, signal

from .noise import fractal_noise, smooth_1d
from .scene import Scene, SceneObject


@dataclass
class RenderOptions:
    speckle: bool = True
    looks_choices: Tuple[int, ...] = (1, 1, 1, 2, 3)
    snr_db_range: Tuple[float, float] = (6.0, 24.0)
    tvg_residual_max: float = 0.3
    range_falloff_gamma_range: Tuple[float, float] = (0.0, 1.0)
    p_dropout: float = 0.15
    p_striping: float = 0.35
    p_smear: float = 0.25
    smear_sigma_px: Tuple[float, float] = (0.6, 2.0)
    ambient: float = 0.0
    shadow_eps: float = 1e-6
    normalise_percentiles: Tuple[float, float] = (0.5, 99.7)
    headroom_db_range: Tuple[float, float] = (2.0, 10.0)   # extra dynamic range above the upper percentile


@dataclass
class Annotation:
    cls: str
    class_id: int
    is_milco: bool
    bbox_highlight: Tuple[int, int, int, int]     # x, y, w, h in pixels (x = range col, y = along row)
    bbox_shadow: Optional[Tuple[int, int, int, int]]
    bbox_union: Tuple[int, int, int, int]
    x_m: float
    y_m: float
    range_m: float
    grazing_deg: float
    aspect_deg: float                              # object axis relative to along-track, [0, 180)
    burial: float
    length_m: float
    highlight_area_px: int
    shadow_area_px: int
    visible: bool


@dataclass
class RenderResult:
    image: np.ndarray             # float32 in [0, 1], log-compressed & normalised
    db: np.ndarray                # float32 intensity in dB (un-normalised)
    intensity: np.ndarray         # float32 linear intensity (pixel grid)
    height: np.ndarray            # heightmap on pixel grid (m)
    label_map: np.ndarray         # int16: -1 bg, i = object i highlight
    shadow_map: np.ndarray        # int16: -1 none, i = shadow cast by object i
    visibility: np.ndarray        # bool on pixel grid
    annotations: List[Annotation]
    meta: Dict[str, float] = field(default_factory=dict)

    def seg_mask(self) -> np.ndarray:
        """3-class semantic mask: 0 = seabed, 1 = highlight, 2 = shadow."""
        seg = np.zeros(self.image.shape, dtype=np.uint8)
        seg[self.shadow_map >= 0] = 2
        seg[self.label_map >= 0] = 1
        return seg


# ---------------------------------------------------------------------------------
# Heightmap construction
# ---------------------------------------------------------------------------------

def _terrain(scene: Scene, X: np.ndarray, Y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    p = scene.terrain_params
    shape = X.shape
    h = p["roughness"] * fractal_noise(shape, rng, octaves=5, persistence=0.6, base_cells=6)
    if p.get("large_scale_amp", 0) > 0:
        h += p["large_scale_amp"] * fractal_noise(shape, rng, octaves=2, persistence=0.5, base_cells=2)
    if p.get("ripple_amp", 0) > 0:
        th = np.radians(p["ripple_dir_deg"])
        phase = 2 * np.pi * (X * np.cos(th) + Y * np.sin(th)) / p["ripple_wavelength"]
        # asymmetric ripple profile (steep lee side) typical of current-formed sand waves
        prof = np.sin(phase) + 0.3 * np.sin(2 * phase)
        wobble = 0.25 * fractal_noise(shape, rng, octaves=2, base_cells=3)
        h += p["ripple_amp"] * (prof + wobble)
    return h


def _object_window(obj: SceneObject, X: np.ndarray, Y: np.ndarray, dx: float) -> Tuple[slice, slice]:
    """Index window (rows, cols) that bounds the object's footprint."""
    L = obj.length_m
    d = obj.dims
    extent = max(L, 2 * d.get("a", 0), 2 * d.get("b", 0), d.get("width", 0), 2 * d.get("base_radius", 0),
                 2 * d.get("radius", 0)) / 2 + 0.3
    x0, x1 = obj.x - extent, obj.x + extent
    y0, y1 = obj.y - extent, obj.y + extent
    xs, ys = X[:, 0], Y[0, :]
    r0 = int(np.clip(np.searchsorted(xs, x0), 0, X.shape[0]))
    r1 = int(np.clip(np.searchsorted(xs, x1), 0, X.shape[0]))
    c0 = int(np.clip(np.searchsorted(ys, y0), 0, X.shape[1]))
    c1 = int(np.clip(np.searchsorted(ys, y1), 0, X.shape[1]))
    return slice(r0, r1), slice(c0, c1)


def rasterise_objects(scene: Scene, X: np.ndarray, Y: np.ndarray, terrain: np.ndarray,
                      dx: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Place objects on the terrain. Returns (height, label_map, mu_map, spec_map, spec_pow_map)."""
    height = terrain.copy()
    label = np.full(X.shape, -1, dtype=np.int16)
    objs = scene.all_objects()
    mu = np.ones(X.shape, dtype=np.float64)
    spec = np.zeros(X.shape, dtype=np.float64)
    spow = np.full(X.shape, 10.0, dtype=np.float64)
    for i, obj in enumerate(objs):
        rs, cs = _object_window(obj, X, Y, dx)
        if rs.stop <= rs.start or cs.stop <= cs.start:
            continue
        xl, yl = X[rs, cs] - obj.x, Y[rs, cs] - obj.y
        th = np.radians(obj.yaw_deg)
        u = xl * np.cos(th) + yl * np.sin(th)
        v = -xl * np.sin(th) + yl * np.cos(th)
        z = obj.height_fn(u, v)
        on = z > 1e-4
        if not on.any():
            continue
        # object rests on the terrain at its centre
        rc = min(max(int(np.searchsorted(X[:, 0], obj.x)), 0), X.shape[0] - 1)
        cc = min(max(int(np.searchsorted(Y[0, :], obj.y)), 0), X.shape[1] - 1)
        base = terrain[rc, cc]
        hz = base + z
        sub_h = height[rs, cs]
        upd = on & (hz > sub_h)
        sub_h[upd] = hz[upd]
        height[rs, cs] = sub_h
        sub_l = label[rs, cs]
        sub_l[upd] = i if i < len(scene.objects) else -2  # -2 = clutter (unlabelled)
        label[rs, cs] = sub_l
        for arr, val in ((mu, obj.reflectivity), (spec, obj.specular), (spow, obj.spec_power)):
            sub = arr[rs, cs]
            sub[upd] = val
            arr[rs, cs] = sub
    return height, label, mu, spec, spow


# ---------------------------------------------------------------------------------
# Acoustic geometry
# ---------------------------------------------------------------------------------

def visibility_raycast(height: np.ndarray, ranges: np.ndarray, altitude: float, eps: float = 1e-6) -> np.ndarray:
    """Exact geometric shadowing by ray casting along each across-track line.

    A pixel at ground range y with height h is illuminated iff the tangent of
    its depression angle seen from the sensor, s = (h - H) / y, is not exceeded
    by any nearer pixel.  Implemented as a running maximum along range.
    """
    s = (height - altitude) / ranges[None, :]
    prev = np.maximum.accumulate(s, axis=1)
    prev = np.concatenate([np.full((s.shape[0], 1), -np.inf), prev[:, :-1]], axis=1)
    return s > prev - eps


def grazing_cosine(height: np.ndarray, ranges: np.ndarray, altitude: float, d: float) -> np.ndarray:
    """cos(theta_i) = n . s_hat between local surface normal and direction to sensor.

    For a flat seabed this equals sin(grazing angle) = H / sqrt(y^2 + H^2).
    """
    hx, hy = np.gradient(height, d, d)
    Y = ranges[None, :]
    num = Y * hy + (altitude - height)
    nn = np.sqrt(1 + hx**2 + hy**2)
    sn = np.sqrt(Y**2 + (altitude - height) ** 2)
    return np.clip(num / (nn * sn), 0.0, 1.0)


def system_psf(res_px: float, taper: float = 1.0, half_width: int = 8) -> np.ndarray:
    """Separable sinc point-spread function with a Hann taper (resolution in pixels)."""
    n = np.arange(-half_width * int(np.ceil(res_px)), half_width * int(np.ceil(res_px)) + 1)
    k = np.sinc(n / res_px)
    w = 0.5 * (1 + np.cos(np.pi * n / (n[-1] + 1)))
    k = k * (taper * w + (1 - taper))
    return k / np.sqrt(np.sum(k**2))


# ---------------------------------------------------------------------------------
# Main render routine
# ---------------------------------------------------------------------------------

def _bbox_from_mask(m: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    if not m.any():
        return None
    rows = np.where(m.any(axis=1))[0]
    cols = np.where(m.any(axis=0))[0]
    r0, r1, c0, c1 = rows[0], rows[-1], cols[0], cols[-1]
    return int(c0), int(r0), int(c1 - c0 + 1), int(r1 - r0 + 1)


def _union(a, b):
    if b is None:
        return a
    x0, y0 = min(a[0], b[0]), min(a[1], b[1])
    x1, y1 = max(a[0] + a[2], b[0] + b[2]), max(a[1] + a[3], b[1] + b[3])
    return int(x0), int(y0), int(x1 - x0), int(y1 - y0)


def _block_reduce(a: np.ndarray, f: int, how: str = "mean") -> np.ndarray:
    H, W = a.shape[0] // f * f, a.shape[1] // f * f
    b = a[:H, :W].reshape(H // f, f, W // f, f)
    if how == "mean":
        return b.mean(axis=(1, 3))
    if how == "max":
        return b.max(axis=(1, 3))
    if how == "any":
        return b.any(axis=(1, 3))
    raise ValueError(how)


def _label_reduce(label: np.ndarray, f: int) -> np.ndarray:
    """Down-sample an integer label map by majority of non-negative labels (else -1)."""
    H, W = label.shape[0] // f * f, label.shape[1] // f * f
    b = label[:H, :W].reshape(H // f, f, W // f, f).transpose(0, 2, 1, 3).reshape(H // f, W // f, f * f)
    out = np.full((H // f, W // f), -1, dtype=np.int16)
    mx = b.max(axis=2)
    out[mx >= 0] = mx[mx >= 0]
    return out


def render_scene(scene: Scene, opts: Optional[RenderOptions] = None, seed: Optional[int] = None,
                 return_fine: bool = False) -> RenderResult:
    opts = opts or RenderOptions()
    geom = scene.geometry
    rng = np.random.default_rng(scene.seed if seed is None else seed)
    f = geom.fine_factor
    d = geom.pixel_m / f
    xs, ys = geom.alongs(fine=True), geom.ranges(fine=True)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    H = geom.altitude_m

    # 1. geometry -------------------------------------------------------------------
    terrain = _terrain(scene, X, Y, rng)
    height, label, mu_obj, spec, spow = rasterise_objects(scene, X, Y, terrain, d)

    # 2. reflectivity ---------------------------------------------------------------
    p = scene.terrain_params
    tex = 1 + p["mu_texture"] * fractal_noise(X.shape, rng, octaves=4, base_cells=3)
    mu = p["mu"] * np.clip(tex, 0.2, None)
    is_obj = label != -1
    mu = np.where(is_obj, mu_obj * p["mu"], mu)

    # 3. grazing / backscatter -----------------------------------------------------
    cos_i = grazing_cosine(height, ys, H, d)
    intensity = mu * cos_i**2 + spec * np.power(cos_i, spow) * is_obj
    # 4. shadows -------------------------------------------------------------------
    vis_all = visibility_raycast(height, ys, H, opts.shadow_eps)
    shadow = np.full(X.shape, -1, dtype=np.int16)
    for i in range(len(scene.objects)):
        mask_i = label == i
        if not mask_i.any():
            continue
        h_wo = np.where(mask_i, terrain, height)
        vis_wo = visibility_raycast(h_wo, ys, H, opts.shadow_eps)
        sh = vis_wo & ~vis_all & ~mask_i
        shadow[sh & (shadow == -1)] = i
    intensity = intensity * vis_all + opts.ambient * intensity.mean()

    # 5. range-dependent gain and noise -------------------------------------------
    y_c = ys.mean()
    gamma = rng.uniform(*opts.range_falloff_gamma_range)
    gain_r = (y_c / ys) ** gamma
    eps_tvg = rng.uniform(0, opts.tvg_residual_max)
    gain_r = gain_r * (1 + eps_tvg * smooth_1d(len(ys), rng, sigma=len(ys) / 6))
    gain_r = np.clip(gain_r, 0.2, None)
    snr_db = rng.uniform(*opts.snr_db_range)
    seabed_mean = float(np.mean(intensity[vis_all & ~is_obj])) if (vis_all & ~is_obj).any() else float(intensity.mean())
    noise_pow = seabed_mean / (10 ** (snr_db / 10))

    # 6. coherent speckle + system PSF ---------------------------------------------
    looks = int(rng.choice(opts.looks_choices)) if opts.speckle else 1
    psf = system_psf(res_px=float(f), taper=1.0)
    psf2 = np.outer(psf, psf)
    amp = np.sqrt(np.clip(intensity, 0, None) * gain_r[None, :])
    acc = np.zeros(X.shape, dtype=np.float64)
    for _ in range(looks):
        if opts.speckle:
            cn = (rng.standard_normal(X.shape) + 1j * rng.standard_normal(X.shape)) / np.sqrt(2)
            nz = (rng.standard_normal(X.shape) + 1j * rng.standard_normal(X.shape)) / np.sqrt(2)
            field_c = amp * cn + np.sqrt(noise_pow * gain_r[None, :]) * nz
            field_c = signal.fftconvolve(field_c, psf2, mode="same")
            acc += np.abs(field_c) ** 2
        else:
            acc += signal.fftconvolve(amp**2 + noise_pow * gain_r[None, :], psf2**2, mode="same")
    I_fine = acc / looks

    # 7. platform artefacts (along-track = rows) ----------------------------------
    meta = {"snr_db": float(snr_db), "looks": looks, "gamma": float(gamma), "tvg_eps": float(eps_tvg),
            "altitude_m": H, "range_start_m": geom.range_start_m, "sediment": scene.sediment,
            "dropout": 0, "striping": 0, "smear_px": 0.0}
    if rng.random() < opts.p_smear:
        s = float(rng.uniform(*opts.smear_sigma_px)) * f
        I_fine = ndimage.gaussian_filter1d(I_fine, s, axis=0)
        meta["smear_px"] = s / f
    if rng.random() < opts.p_striping:
        g = 1 + 0.12 * rng.standard_normal(I_fine.shape[0] // f)
        I_fine *= np.repeat(np.clip(g, 0.5, 1.5), f)[: I_fine.shape[0], None]
        meta["striping"] = 1
    if rng.random() < opts.p_dropout:
        n_drop = int(rng.integers(1, 4))
        for _ in range(n_drop):
            r = int(rng.integers(0, I_fine.shape[0] // f)) * f
            I_fine[r: r + f, :] *= rng.uniform(0.0, 0.2)
        meta["dropout"] = n_drop

    # 8. down-sample to the pixel grid --------------------------------------------
    I_px = _block_reduce(I_fine, f, "mean").astype(np.float32)
    height_px = _block_reduce(height, f, "mean").astype(np.float32)
    vis_px = _block_reduce(vis_all.astype(np.float64), f, "mean") > 0.5
    label_px = _label_reduce(np.where(label == -2, -1, label), f)
    shadow_px = _label_reduce(shadow, f)
    shadow_px[label_px >= 0] = -1

    # 9. log compression ------------------------------------------------------------
    db = 10 * np.log10(I_px + 1e-12)
    lo, hi = np.percentile(db, opts.normalise_percentiles)
    lo += rng.uniform(-2, 2)
    hi += rng.uniform(*opts.headroom_db_range)
    img = np.clip((db - lo) / max(hi - lo, 1e-3), 0, 1).astype(np.float32)

    # 10. annotations ---------------------------------------------------------------
    anns: List[Annotation] = []
    for i, obj in enumerate(scene.objects):
        hm = label_px == i
        sm = shadow_px == i
        bh = _bbox_from_mask(hm)
        bs = _bbox_from_mask(sm)
        visible = bh is not None and (hm & vis_px).sum() >= 2
        if bh is None:
            # fully buried / sub-pixel: keep a tiny box at the true location for bookkeeping
            cx = int(np.clip((obj.y - geom.range_start_m) / geom.pixel_m, 0, geom.n_range - 1))
            cy = int(np.clip(obj.x / geom.pixel_m, 0, geom.n_along - 1))
            bh = (cx, cy, 1, 1)
        anns.append(Annotation(
            cls=obj.cls, class_id=obj.class_id, is_milco=obj.is_milco,
            bbox_highlight=bh, bbox_shadow=bs, bbox_union=_union(bh, bs),
            x_m=obj.x, y_m=obj.y, range_m=float(np.hypot(obj.y, H)),
            grazing_deg=geom.grazing_deg(obj.y), aspect_deg=float(obj.yaw_deg % 180.0),
            burial=obj.burial, length_m=obj.length_m,
            highlight_area_px=int(hm.sum()), shadow_area_px=int(sm.sum()), visible=bool(visible)))

    res = RenderResult(image=img, db=db.astype(np.float32), intensity=I_px, height=height_px,
                       label_map=label_px, shadow_map=shadow_px, visibility=vis_px,
                       annotations=anns, meta=meta)
    if return_fine:
        res.meta["_fine"] = {"intensity": intensity, "visibility": vis_all, "height": height,
                             "X": X, "Y": Y, "label": label}
    return res
