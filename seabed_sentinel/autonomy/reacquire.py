"""Information-driven re-acquisition (re-look) planning for an AUV / USV.

After a first survey pass the ATR produces contacts with class posteriors.
Uncertain contacts should be re-imaged from the aspect that is expected to
reduce classification uncertainty the most, within a path-length budget.
The planner:

1. scores candidate re-look legs for each contact by expected information gain
   (from the aspect-dependent confusion model),
2. greedily selects (contact, aspect) pairs by gain-per-metre until the budget
   is exhausted, and
3. orders the selected legs with a nearest-neighbour tour improved by 2-opt,
   respecting a minimum turn radius via a Dubins-style penalty on heading
   changes.

Each re-look leg is a straight track at a stand-off range from the contact
(so the contact sits in the well-focused part of the swath) oriented so that
the contact is seen at the requested aspect angle.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .aspect_fusion import AspectConfusionModel, entropy


@dataclass
class Contact2D:
    id: int
    x: float                       # east (m)
    y: float                       # north (m)
    posterior: np.ndarray          # class posterior from first pass
    first_aspect_deg: float        # aspect of the first look
    est_heading_deg: float = 0.0   # estimated object axis heading (deg from north), unknown -> 0


@dataclass
class Leg:
    contact_id: int
    aspect_deg: float
    start: Tuple[float, float]
    end: Tuple[float, float]
    gain_bits: float
    heading_deg: float


@dataclass
class ReacquisitionPlan:
    legs: List[Leg]
    order: List[int]
    path: np.ndarray              # [K, 2] way-points (east, north)
    total_length_m: float
    total_gain_bits: float
    skipped: List[int] = field(default_factory=list)

    def summary(self) -> Dict[str, float]:
        return {"n_legs": len(self.legs), "path_length_m": self.total_length_m,
                "expected_gain_bits": self.total_gain_bits, "n_skipped": len(self.skipped)}


def make_leg(c: Contact2D, aspect_deg: float, standoff_m: float, leg_len_m: float) -> Leg:
    """Straight leg at ``standoff_m`` from the contact so the object axis makes
    ``aspect_deg`` with the track. Track heading = object heading - aspect."""
    hdg = np.radians(c.est_heading_deg - aspect_deg)
    dirv = np.array([np.sin(hdg), np.cos(hdg)])          # unit along-track (east, north)
    normal = np.array([np.cos(hdg), -np.sin(hdg)])       # starboard normal
    centre = np.array([c.x, c.y]) - standoff_m * normal  # contact on starboard side
    start = centre - dirv * leg_len_m / 2
    end = centre + dirv * leg_len_m / 2
    return Leg(c.id, aspect_deg, tuple(start), tuple(end), 0.0, float(np.degrees(hdg) % 360))


def _dist(a, b) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _tour_length(legs: Sequence[Leg], order: Sequence[int], start: Tuple[float, float], turn_radius: float) -> float:
    L, pos, hdg = 0.0, start, None
    for k in order:
        leg = legs[k]
        L += _dist(pos, leg.start)
        if hdg is not None:
            dh = abs((leg.heading_deg - hdg + 180) % 360 - 180)
            L += turn_radius * np.radians(dh)          # Dubins-style turn penalty
        L += _dist(leg.start, leg.end)
        pos, hdg = leg.end, leg.heading_deg
    return L


def _two_opt(legs, order, start, turn_radius, iters: int = 200):
    best = list(order)
    best_len = _tour_length(legs, best, start, turn_radius)
    improved, it = True, 0
    while improved and it < iters:
        improved = False
        it += 1
        for i in range(len(best) - 1):
            for j in range(i + 1, len(best)):
                cand = best[:i] + best[i: j + 1][::-1] + best[j + 1:]
                L = _tour_length(legs, cand, start, turn_radius)
                if L < best_len - 1e-6:
                    best, best_len, improved = cand, L, True
    return best, best_len


def plan_reacquisition(contacts: Sequence[Contact2D], model: AspectConfusionModel, budget_m: float,
                       start: Tuple[float, float] = (0.0, 0.0), standoff_m: float = 40.0, leg_len_m: float = 60.0,
                       turn_radius_m: float = 15.0, candidate_aspects: Sequence[float] = (0, 30, 60, 90, 120, 150),
                       min_entropy_bits: float = 0.3, max_legs_per_contact: int = 2) -> ReacquisitionPlan:
    """Greedy gain-per-metre selection of re-look legs followed by tour optimisation."""
    # candidate legs
    cands: List[Leg] = []
    for c in contacts:
        if entropy(c.posterior) < min_entropy_bits:
            continue
        for a in candidate_aspects:
            if abs(((a - c.first_aspect_deg) + 90) % 180 - 90) < 15:
                continue  # same aspect as already observed: little new information
            leg = make_leg(c, a, standoff_m, leg_len_m)
            leg.gain_bits = model.expected_information_gain(c.posterior, a)
            cands.append(leg)
    selected: List[Leg] = []
    per_contact: Dict[int, int] = {}
    post_by_contact = {c.id: np.asarray(c.posterior, float).copy() for c in contacts}
    pos = start
    remaining = budget_m
    used = set()
    while True:
        best, best_score, best_cost = None, 0.0, 0.0
        for idx, leg in enumerate(cands):
            if idx in used or per_contact.get(leg.contact_id, 0) >= max_legs_per_contact:
                continue
            # re-evaluate gain with the *current* fused posterior for that contact
            gain = model.expected_information_gain(post_by_contact[leg.contact_id], leg.aspect_deg)
            cost = _dist(pos, leg.start) + leg_len_m
            if cost > remaining:
                continue
            score = gain / max(cost, 1.0)
            if score > best_score:
                best, best_score, best_cost, best_idx = leg, score, cost, idx
        if best is None:
            break
        best.gain_bits = best_score * best_cost
        selected.append(best)
        used.add(best_idx)
        per_contact[best.contact_id] = per_contact.get(best.contact_id, 0) + 1
        remaining -= best_cost
        pos = best.end
        # expected posterior after the look (approximate: sharpen towards the MAP class)
        L = model.likelihood(best.aspect_deg)
        p = post_by_contact[best.contact_id]
        p_obs = p @ L
        o = int(np.argmax(p_obs))
        p = p * L[:, o]
        post_by_contact[best.contact_id] = p / p.sum()
    if not selected:
        return ReacquisitionPlan([], [], np.array([start]), 0.0, 0.0, skipped=[c.id for c in contacts])
    order = list(range(len(selected)))
    order, length = _two_opt(selected, order, start, turn_radius_m)
    # the greedy stage budgets straight-line distance; enforce the budget on the
    # final tour (which includes turn penalties) by dropping the lowest-gain legs
    while length > budget_m and len(selected) > 1:
        drop = int(np.argmin([leg.gain_bits for leg in selected]))
        per_contact[selected[drop].contact_id] -= 1
        selected.pop(drop)
        order, length = _two_opt(selected, list(range(len(selected))), start, turn_radius_m)
    per_contact = {k: v for k, v in per_contact.items() if v > 0}
    path = [start]
    for k in order:
        path += [selected[k].start, selected[k].end]
    skipped = [c.id for c in contacts if c.id not in per_contact]
    return ReacquisitionPlan(selected, order, np.array(path), float(length), float(sum(l.gain_bits for l in selected)),
                             skipped=skipped)
