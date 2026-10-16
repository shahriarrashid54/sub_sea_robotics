"""Physics sanity tests for the simulator and the SAS processing chain."""
import numpy as np
import pytest

from seabed_sentinel.sim.acoustics import (
    SASConfig,
    chirp_baseband,
    platform_trajectory,
    range_compress,
    simulate_echoes,
    sinusoidal_sway,
)
from seabed_sentinel.sim.renderer import grazing_cosine, visibility_raycast
from seabed_sentinel.sim.sas_processor import backproject, dpca_estimate_sway, measure_resolution


def test_shadow_length_matches_geometry():
    """A box of height h whose far edge is at ground range y casts a shadow of
    length h * y / (H - h) on a flat seabed."""
    d = 0.01
    H = 10.0
    y = 20.0 + (np.arange(1200) + 0.5) * d
    height = np.zeros((1, len(y)))
    h = 0.5
    y0, y1 = 25.0, 25.6
    height[0, (y >= y0) & (y <= y1)] = h
    vis = visibility_raycast(height, y, H)
    shadow = (~vis[0]) & (y > y1)
    L_meas = shadow.sum() * d
    L_theory = h * y1 / (H - h)
    assert abs(L_meas - L_theory) < 3 * d
    # nothing before the object is shadowed and the object top is lit
    assert vis[0, y < y0].all()
    assert vis[0, (y >= y0) & (y <= y1)].all()


def test_flat_seabed_grazing_cosine_equals_sin_grazing():
    y = np.linspace(20, 80, 200)
    H = 12.0
    cos_i = grazing_cosine(np.zeros((3, len(y))), y, H, 0.03)
    expected = H / np.sqrt(y**2 + H**2)
    assert np.allclose(cos_i[1], expected, atol=1e-6)


def test_facing_slope_is_brighter_than_lee_slope():
    y = np.linspace(30, 40, 500)
    H = 10.0
    slope = 0.1
    # slope rising with range faces the sensor (brighter); falling slope faces away
    rising = np.tile(slope * (y - y[0]), (3, 1))
    falling = -rising
    c_up = grazing_cosine(rising, y, H, y[1] - y[0])[1].mean()
    c_down = grazing_cosine(falling, y, H, y[1] - y[0])[1].mean()
    assert c_up > c_down


def test_chirp_compression_resolution():
    cfg = SASConfig(range_taper=False)
    p = chirp_baseband(cfg)
    ac = np.abs(np.correlate(p, p, mode="full"))
    ac /= ac.max()
    width_samples = (ac >= 0.5).sum()
    width_m = width_samples / cfg.fs * cfg.c / 2
    # -6 dB width of the matched-filter output ~ c/(2B) (within a factor ~1.5 incl. sampling)
    assert 0.5 * cfg.range_resolution < width_m < 1.6 * cfg.range_resolution


def _point_target_psf(cfg: SASConfig, y_t: float):
    L = cfg.synthetic_aperture_length(np.hypot(y_t, cfg.altitude))
    n_pings = int(np.ceil(L / cfg.ping_advance)) + 4
    traj = platform_trajectory(cfg, n_pings, x0=-n_pings * cfg.ping_advance / 2)
    scat = {"x": np.array([0.0]), "y": np.array([y_t]), "z": np.array([0.0]), "amp": np.array([1.0 + 0j])}
    ech = simulate_echoes(cfg, scat, traj, rng=np.random.default_rng(0))
    comp = range_compress(ech, cfg)
    xg = np.arange(-0.12, 0.12, 0.0025)
    yg = y_t + np.arange(-0.12, 0.12, 0.0025)
    img = backproject(comp, ech["t0"], cfg, traj, xg, yg)
    return measure_resolution(img, xg, yg)


def test_sas_along_track_resolution_is_d_over_2_and_range_independent():
    cfg = SASConfig(n_rx=8, noise_db=-60)
    r20 = _point_target_psf(cfg, 20.0)
    r35 = _point_target_psf(cfg, 35.0)
    theory = cfg.along_track_resolution
    for r in (r20, r35):
        assert 0.7 * theory < r["along_track_m"] < 1.6 * theory, r
        assert 0.6 * cfg.range_resolution < r["range_m"] < 2.5 * cfg.range_resolution, r
    # range independence (real-aperture sonar would degrade by 35/20 = 1.75x)
    assert abs(r35["along_track_m"] - r20["along_track_m"]) / r20["along_track_m"] < 0.25


def test_speed_constraint_and_ambiguity():
    cfg = SASConfig()
    # the platform must not advance more than the redundant-phase-centre span per ping
    assert cfg.speed * cfg.pri == pytest.approx(cfg.ping_advance)
    assert cfg.ping_advance <= cfg.n_rx * cfg.d_rx / 2
    assert cfg.max_unambiguous_range > 50


def test_dpca_recovers_sway():
    cfg = SASConfig(n_rx=8, overlap_pc=2, noise_db=-40)
    rng = np.random.default_rng(3)
    n_scat = 1500
    scat = {"x": rng.uniform(-3, 3, n_scat), "y": rng.uniform(22, 27, n_scat), "z": np.zeros(n_scat),
            "amp": (rng.standard_normal(n_scat) + 1j * rng.standard_normal(n_scat))}
    n_pings = 14
    sway = sinusoidal_sway(n_pings, amplitude=0.02, period_pings=9.0)
    traj = platform_trajectory(cfg, n_pings, x0=-n_pings * cfg.ping_advance / 2, sway=sway)
    ech = simulate_echoes(cfg, scat, traj, rng=rng)
    comp = range_compress(ech, cfg)
    est = dpca_estimate_sway(comp, ech["t0"], cfg)
    err = (est["sway"] - est["sway"][0]) - (sway - sway[0])
    assert np.sqrt(np.mean(err**2)) < 0.004, f"RMS sway error {np.sqrt(np.mean(err**2)) * 1e3:.2f} mm"
