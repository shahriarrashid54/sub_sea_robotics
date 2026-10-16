"""Seabed scene description and target primitives.

Coordinate system (right-handed):
    x : along-track (direction of platform travel), metres
    y : across-track ground range from the sonar nadir line, metres (y > 0)
    z : height above the local mean seabed, metres (positive up)

The sonar platform (towfish / AUV) flies at altitude ``H`` above the mean
seabed at ground range ``y = 0``. Because a synthetic aperture focuses every
pixel from the aperture centred on that pixel's along-track position, the
effective look direction for a pixel is contained in its own (y, z) plane.

All object primitives are expressed as *height functions* z(u, v) in a local
frame (u along the object's main axis, v perpendicular) and then rasterised
onto the scene heightmap. This is what lets the same scene feed both the fast
image-domain renderer and the full acoustic point-scatterer simulator.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# Class taxonomy -----------------------------------------------------------------
# MILCO = MIne-Like COntact, NOMBO = NOn-Mine-like Bottom Object (NATO MCM terms).
CLASSES: List[str] = ["cylinder", "truncated_cone", "wedge", "sphere", "rock", "debris"]
MILCO_CLASSES = {"cylinder", "truncated_cone", "wedge", "sphere"}
NOMBO_CLASSES = {"rock", "debris"}
CLASS_TO_ID: Dict[str, int] = {c: i for i, c in enumerate(CLASSES)}

SEDIMENTS = ["mud", "sand", "gravel"]


@dataclass
class SonarGeometry:
    """Imaging geometry for one tile."""

    pixel_m: float = 0.03          # pixel size (both axes), metres
    altitude_m: float = 12.0       # sonar altitude above mean seabed
    range_start_m: float = 30.0    # ground range of the first (near) column
    n_along: int = 288             # rows   (along-track samples)
    n_range: int = 288             # cols   (across-track samples)
    fine_factor: int = 2           # oversampling for coherent speckle rendering

    @property
    def along_extent_m(self) -> float:
        return self.n_along * self.pixel_m

    @property
    def range_extent_m(self) -> float:
        return self.n_range * self.pixel_m

    def ranges(self, fine: bool = False) -> np.ndarray:
        n = self.n_range * (self.fine_factor if fine else 1)
        d = self.pixel_m / (self.fine_factor if fine else 1)
        return self.range_start_m + (np.arange(n) + 0.5) * d

    def alongs(self, fine: bool = False) -> np.ndarray:
        n = self.n_along * (self.fine_factor if fine else 1)
        d = self.pixel_m / (self.fine_factor if fine else 1)
        return (np.arange(n) + 0.5) * d

    def grazing_deg(self, y: float) -> float:
        return float(np.degrees(np.arctan2(self.altitude_m, y)))


@dataclass
class SceneObject:
    """A single object placed on the seabed."""

    cls: str
    x: float                      # along-track centre (m)
    y: float                      # ground range centre (m)
    yaw_deg: float                # orientation of main axis relative to +x
    dims: Dict[str, float]        # primitive-specific dimensions (m)
    reflectivity: float = 3.0     # relative to local sediment
    specular: float = 0.0         # specular highlight strength (man-made = high)
    spec_power: float = 20.0      # specular exponent (smoothness)
    burial: float = 0.0           # fraction of max height buried (0 = proud)
    seed: int = 0

    @property
    def is_milco(self) -> bool:
        return self.cls in MILCO_CLASSES

    @property
    def class_id(self) -> int:
        return CLASS_TO_ID[self.cls]

    @property
    def length_m(self) -> float:
        d = self.dims
        return d.get("length", 2 * d.get("radius", d.get("base_radius", 0.5)))

    def height_fn(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Height above seabed as a function of local coords (u, v) in metres."""
        d = self.dims
        rng = np.random.default_rng(self.seed)
        if self.cls == "cylinder":
            r, L = d["radius"], d["length"]
            inside = (np.abs(u) <= L / 2) & (np.abs(v) <= r)
            z = np.where(inside, r + np.sqrt(np.clip(r**2 - v**2, 0, None)), 0.0)
        elif self.cls == "truncated_cone":
            rb, rt, h = d["base_radius"], d["top_radius"], d["height"]
            rho = np.hypot(u, v)
            z = h * np.clip((rb - rho) / max(rb - rt, 1e-6), 0, 1)
        elif self.cls == "wedge":
            L, W, h = d["length"], d["width"], d["height"]
            inside = (np.abs(u) <= L / 2) & (np.abs(v) <= W / 2 * (0.6 + 0.4 * (0.5 - u / L)))
            z = np.where(inside, h * np.clip(0.5 - u / L, 0, 1), 0.0)
        elif self.cls == "sphere":
            r = d["radius"]
            rho2 = u**2 + v**2
            z = np.where(rho2 <= r**2, r + np.sqrt(np.clip(r**2 - rho2, 0, None)), 0.0)
        elif self.cls == "rock":
            a, b, c = d["a"], d["b"], d["height"]
            q = 1 - (u / a) ** 2 - (v / b) ** 2
            base = c * np.sqrt(np.clip(q, 0, None))
            # low-frequency random lumpiness so rocks are not perfect ellipsoids
            lump = np.ones_like(base)
            for _ in range(3):
                kx, ky = rng.uniform(0.6, 1.8, 2)
                ph = rng.uniform(0, 2 * np.pi, 2)
                lump += 0.18 * np.sin(kx * np.pi * u / a + ph[0] + 1.3 * np.cos(ky * np.pi * v / b + ph[1]))
            z = base * np.clip(lump, 0.35, 1.6)
        elif self.cls == "debris":
            kind = d.get("kind", 0)
            if kind == 0:  # crate / box with dents
                a, b, h = d["a"], d["b"], d["height"]
                inside = (np.abs(u) <= a) & (np.abs(v) <= b)
                ph = rng.uniform(0, 2 * np.pi)
                dent = 1 - 0.2 * (np.sin(3.5 * u / a + ph) ** 2) * (np.abs(v) < b / 2) - 0.15 * (np.abs(u) > 0.8 * a)
                z = np.where(inside, h * dent, 0.0)
            else:  # tyre (torus lying flat)
                R, r = d["a"], d["b"]
                rho = np.hypot(u, v)
                q = r**2 - (rho - R) ** 2
                z = np.where(q > 0, r + np.sqrt(np.clip(q, 0, None)), 0.0)
        else:
            raise ValueError(self.cls)
        if self.burial > 0:
            z = np.clip(z - self.burial * np.max(z, initial=0.0), 0, None)
        return z

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SceneConfig:
    """Stochastic ranges used by :func:`make_random_scene`."""

    max_objects: int = 3
    p_empty: float = 0.18
    p_milco: float = 0.62
    p_burial: float = 0.3
    max_burial: float = 0.6
    ripple_prob: float = 0.55
    rock_field_prob: float = 0.25
    small_rock_density: float = 0.08   # small clutter rocks per m^2 in rock fields
    pixel_m: float = 0.03
    n_along: int = 288
    n_range: int = 288
    altitude_range: Tuple[float, float] = (8.0, 20.0)
    range_start_range: Tuple[float, float] = (15.0, 190.0)
    sediments: Tuple[str, ...] = tuple(SEDIMENTS)


@dataclass
class Scene:
    geometry: SonarGeometry
    sediment: str
    objects: List[SceneObject]
    clutter: List[SceneObject]
    terrain_params: Dict[str, float] = field(default_factory=dict)
    seed: int = 0

    def all_objects(self) -> List[SceneObject]:
        return list(self.objects) + list(self.clutter)


# Random generation --------------------------------------------------------------

def _random_object(cls: str, rng: np.random.Generator, geom: SonarGeometry, cfg: SceneConfig,
                   margin: float = 1.5) -> SceneObject:
    x = rng.uniform(margin, geom.along_extent_m - margin)
    y = rng.uniform(geom.range_start_m + margin, geom.range_start_m + geom.range_extent_m - margin)
    yaw = rng.uniform(0, 180)
    spec, spec_p, refl, burial = 0.0, 20.0, 1.0, 0.0
    if cls == "cylinder":
        dims = {"radius": rng.uniform(0.2, 0.35), "length": rng.uniform(1.0, 2.4)}
        refl, spec, spec_p = rng.uniform(2.5, 4.5), rng.uniform(2, 5), rng.uniform(15, 40)
    elif cls == "truncated_cone":
        rb = rng.uniform(0.4, 0.6)
        dims = {"base_radius": rb, "top_radius": rb * rng.uniform(0.35, 0.6), "height": rng.uniform(0.3, 0.5)}
        refl, spec, spec_p = rng.uniform(2.5, 4.5), rng.uniform(1, 4), rng.uniform(10, 30)
    elif cls == "wedge":
        dims = {"length": rng.uniform(0.9, 1.3), "width": rng.uniform(0.6, 0.9), "height": rng.uniform(0.3, 0.45)}
        refl, spec, spec_p = rng.uniform(2.5, 4.5), rng.uniform(1, 4), rng.uniform(10, 30)
    elif cls == "sphere":
        dims = {"radius": rng.uniform(0.3, 0.5)}
        refl, spec, spec_p = rng.uniform(2.5, 4.5), rng.uniform(3, 6), rng.uniform(20, 50)
    elif cls == "rock":
        a = rng.uniform(0.35, 1.3)
        dims = {"a": a, "b": a * rng.uniform(0.5, 1.0), "height": rng.uniform(0.2, 0.8)}
        refl, spec, spec_p = rng.uniform(1.5, 3.5), rng.uniform(0, 0.8), rng.uniform(2, 6)
    elif cls == "debris":
        kind = int(rng.integers(0, 2))
        if kind == 0:
            dims = {"kind": 0, "a": rng.uniform(0.3, 0.7), "b": rng.uniform(0.25, 0.6), "height": rng.uniform(0.2, 0.6)}
        else:
            dims = {"kind": 1, "a": rng.uniform(0.3, 0.5), "b": rng.uniform(0.08, 0.14)}
        refl, spec, spec_p = rng.uniform(1.5, 3.5), rng.uniform(0.5, 2.5), rng.uniform(5, 20)
    else:
        raise ValueError(cls)
    if cls in MILCO_CLASSES and rng.random() < cfg.p_burial:
        burial = float(rng.uniform(0.1, cfg.max_burial))
    return SceneObject(cls=cls, x=float(x), y=float(y), yaw_deg=float(yaw), dims=dims,
                       reflectivity=float(refl), specular=float(spec), spec_power=float(spec_p),
                       burial=burial, seed=int(rng.integers(0, 2**31 - 1)))


def make_random_scene(seed: int, cfg: Optional[SceneConfig] = None) -> Scene:
    """Draw a random seabed scene (geometry, sediment, terrain, objects)."""
    cfg = cfg or SceneConfig()
    rng = np.random.default_rng(seed)
    geom = SonarGeometry(
        pixel_m=cfg.pixel_m,
        altitude_m=float(rng.uniform(*cfg.altitude_range)),
        range_start_m=float(rng.uniform(*cfg.range_start_range)),
        n_along=cfg.n_along,
        n_range=cfg.n_range,
    )
    sediment = str(rng.choice(list(cfg.sediments)))
    terrain = {
        "roughness": {"mud": 0.004, "sand": 0.01, "gravel": 0.03}[sediment] * rng.uniform(0.5, 1.5),
        "ripple_amp": float(rng.uniform(0.01, 0.05)) if (sediment != "mud" and rng.random() < cfg.ripple_prob) else 0.0,
        "ripple_wavelength": float(rng.uniform(0.3, 1.6)),
        "ripple_dir_deg": float(rng.uniform(0, 180)),
        "mu": {"mud": 0.35, "sand": 1.0, "gravel": 1.8}[sediment] * rng.uniform(0.8, 1.25),
        "mu_texture": float(rng.uniform(0.1, 0.5)),
        "large_scale_amp": float(rng.uniform(0.0, 0.12)),
    }
    objects: List[SceneObject] = []
    n_obj = 0 if rng.random() < cfg.p_empty else int(rng.integers(1, cfg.max_objects + 1))
    for _ in range(n_obj):
        if rng.random() < cfg.p_milco:
            cls = str(rng.choice(sorted(MILCO_CLASSES)))
        else:
            cls = str(rng.choice(sorted(NOMBO_CLASSES)))
        for _attempt in range(20):
            obj = _random_object(cls, rng, geom, cfg)
            if all(np.hypot(obj.x - o.x, obj.y - o.y) > 2.5 for o in objects):
                objects.append(obj)
                break
    clutter: List[SceneObject] = []
    if rng.random() < cfg.rock_field_prob or sediment == "gravel":
        area = geom.along_extent_m * geom.range_extent_m
        n_cl = int(rng.poisson(cfg.small_rock_density * area * (0.5 if sediment != "gravel" else 1.0)))
        for _ in range(n_cl):
            a = rng.uniform(0.12, 0.35)
            clutter.append(SceneObject(
                cls="rock", x=float(rng.uniform(0, geom.along_extent_m)),
                y=float(rng.uniform(geom.range_start_m, geom.range_start_m + geom.range_extent_m)),
                yaw_deg=float(rng.uniform(0, 180)),
                dims={"a": a, "b": a * rng.uniform(0.5, 1.0), "height": rng.uniform(0.05, 0.25)},
                reflectivity=float(rng.uniform(1.3, 2.5)), specular=0.0, spec_power=3.0,
                seed=int(rng.integers(0, 2**31 - 1))))
    return Scene(geometry=geom, sediment=sediment, objects=objects, clutter=clutter,
                 terrain_params=terrain, seed=seed)
