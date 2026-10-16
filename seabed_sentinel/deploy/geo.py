"""Geo-referencing of waterfall detections into contact reports.

Given the towfish navigation (position + heading per ping) and the imaging
geometry, an image pixel (row = ping, col = across-track sample) is mapped to
ground range and then to latitude / longitude via a local east-north-up
tangent plane. Contact lists are exported as GeoJSON for GIS / mission
planning tools.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

R_EARTH = 6378137.0


@dataclass
class TowfishNav:
    """Navigation samples per ping (arrays of equal length)."""

    lat: np.ndarray            # degrees
    lon: np.ndarray            # degrees
    heading_deg: np.ndarray    # course over ground, degrees clockwise from north
    altitude_m: np.ndarray     # towfish altitude above seabed

    @classmethod
    def straight_line(cls, n: int, lat0: float, lon0: float, heading_deg: float, ping_advance_m: float,
                      altitude_m: float = 12.0, sway_m: Optional[np.ndarray] = None) -> "TowfishNav":
        d = np.arange(n) * ping_advance_m
        h = np.radians(heading_deg)
        east = d * np.sin(h)
        north = d * np.cos(h)
        if sway_m is not None:
            east += sway_m * np.cos(h)
            north -= sway_m * np.sin(h)
        lat, lon = enu_to_latlon(east, north, lat0, lon0)
        return cls(lat=lat, lon=lon, heading_deg=np.full(n, heading_deg, float), altitude_m=np.full(n, altitude_m, float))


def enu_to_latlon(east: np.ndarray, north: np.ndarray, lat0: float, lon0: float) -> Tuple[np.ndarray, np.ndarray]:
    lat = lat0 + np.degrees(north / R_EARTH)
    lon = lon0 + np.degrees(east / (R_EARTH * np.cos(np.radians(lat0))))
    return lat, lon


def latlon_to_enu(lat: np.ndarray, lon: np.ndarray, lat0: float, lon0: float) -> Tuple[np.ndarray, np.ndarray]:
    north = np.radians(lat - lat0) * R_EARTH
    east = np.radians(lon - lon0) * R_EARTH * np.cos(np.radians(lat0))
    return east, north


def pixel_to_latlon(row: float, col: float, nav: TowfishNav, pixel_m: float, range_start_m: float,
                    side: str = "starboard", rows_per_ping: float = 1.0) -> Tuple[float, float, float]:
    """Map a waterfall pixel to (lat, lon, ground_range_m).

    ``row`` indexes along-track samples (``rows_per_ping`` image rows per
    navigation sample), ``col`` indexes across-track ground-range samples
    starting at ``range_start_m``. Starboard is +90 deg from heading.
    """
    n = len(nav.lat)
    p = np.clip(row / rows_per_ping, 0, n - 1)
    i0 = int(np.floor(p)); i1 = min(i0 + 1, n - 1); f = p - i0
    lat_p = (1 - f) * nav.lat[i0] + f * nav.lat[i1]
    lon_p = (1 - f) * nav.lon[i0] + f * nav.lon[i1]
    hdg = np.radians((1 - f) * nav.heading_deg[i0] + f * nav.heading_deg[i1])
    gr = range_start_m + (col + 0.5) * pixel_m
    sign = 1.0 if side == "starboard" else -1.0
    bearing = hdg + sign * np.pi / 2
    east, north = gr * np.sin(bearing), gr * np.cos(bearing)
    lat, lon = enu_to_latlon(np.array([east]), np.array([north]), lat_p, lon_p)
    return float(lat[0]), float(lon[0]), float(gr)


def contacts_to_geojson(contacts: Sequence, nav: TowfishNav, pixel_m: float, range_start_m: float,
                        class_names: Sequence[str], side: str = "starboard", rows_per_ping: float = 1.0,
                        path: Optional[str] = None) -> Dict:
    feats = []
    for c in contacts:
        lat, lon, gr = pixel_to_latlon(c.ping_index, c.range_px, nav, pixel_m, range_start_m, side, rows_per_ping)
        c.latlon = [lat, lon]
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {"contact_id": c.id, "class": class_names[c.label] if c.label < len(class_names) else str(c.label),
                           "score": round(c.score, 4), "ground_range_m": round(gr, 2), "side": side,
                           "along_track_row": c.ping_index, "box_px": [round(v, 1) for v in c.box],
                           "length_m_est": round((c.box[3] - c.box[1]) * pixel_m, 2),
                           "width_m_est": round((c.box[2] - c.box[0]) * pixel_m, 2)},
        })
    gj = {"type": "FeatureCollection", "features": feats}
    if path:
        with open(path, "w") as f:
            json.dump(gj, f, indent=1)
    return gj
