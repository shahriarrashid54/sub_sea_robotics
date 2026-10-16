"""Multi-aspect Bayesian classification fusion.

A single SAS look at a seabed object is ambiguous: a cylinder seen end-on looks
like a truncated cone, a wedge seen from the wrong side looks like a rock. The
ATR classifier's confusion depends strongly on the *aspect angle* (object
axis relative to the sonar track). This module models that dependence with
aspect-binned confusion matrices (estimated from the evaluation suite) and
fuses successive looks with Bayes' rule:

    P(class | looks) ∝ P(class) * Π_k P(observed_k | class, aspect_k)

It also provides the *expected information gain* of a hypothetical new look at
a given aspect, which drives the re-acquisition planner.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np


def entropy(p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1)
    p = p / p.sum()
    return float(-(p * np.log2(p)).sum())


@dataclass
class AspectConfusionModel:
    """P(predicted | true, aspect_bin) with Laplace smoothing."""

    class_names: Sequence[str]
    aspect_edges: np.ndarray = field(default_factory=lambda: np.array([0, 30, 60, 90, 120, 150, 180.001]))
    counts: Optional[np.ndarray] = None  # [bins, C, C]
    alpha: float = 0.5

    def __post_init__(self):
        C = len(self.class_names)
        B = len(self.aspect_edges) - 1
        if self.counts is None:
            self.counts = np.zeros((B, C, C))

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    def aspect_bin(self, aspect_deg: float) -> int:
        a = float(aspect_deg) % 180.0
        return int(np.clip(np.searchsorted(self.aspect_edges, a, side="right") - 1, 0, len(self.aspect_edges) - 2))

    def add(self, true_cls: int, pred_cls: int, aspect_deg: float) -> None:
        self.counts[self.aspect_bin(aspect_deg), true_cls, pred_cls] += 1

    def likelihood(self, aspect_deg: float) -> np.ndarray:
        """Row-normalised confusion P(pred | true) for this aspect bin."""
        m = self.counts[self.aspect_bin(aspect_deg)] + self.alpha
        return m / m.sum(axis=1, keepdims=True)

    @classmethod
    def synthetic(cls, class_names: Sequence[str], base_acc: float = 0.8, seed: int = 0) -> "AspectConfusionModel":
        """A plausible prior model when no evaluation data is available: broadside
        looks (aspect ~90 deg) are most informative for elongated shapes."""
        rng = np.random.default_rng(seed)
        m = cls(class_names)
        C = m.n_classes
        for b in range(len(m.aspect_edges) - 1):
            centre = 0.5 * (m.aspect_edges[b] + m.aspect_edges[b + 1])
            acc = base_acc - 0.25 * abs(np.sin(np.radians(centre - 90))) ** 2
            for t in range(C):
                row = rng.dirichlet(np.ones(C) * 0.6) * (1 - acc)
                row[t] += acc
                m.counts[b, t] = row * 200
        return m

    def expected_information_gain(self, posterior: np.ndarray, aspect_deg: float) -> float:
        """E_obs[ H(post) - H(post | obs) ] for one new look at this aspect."""
        L = self.likelihood(aspect_deg)                 # [true, pred]
        p_obs = posterior @ L                            # predictive distribution of the observation
        h0 = entropy(posterior)
        exp_h = 0.0
        for o in range(self.n_classes):
            if p_obs[o] < 1e-12:
                continue
            post_o = posterior * L[:, o]
            post_o /= post_o.sum()
            exp_h += p_obs[o] * entropy(post_o)
        return h0 - exp_h


def bayesian_fuse(prior: np.ndarray, observations: Sequence[Dict], model: AspectConfusionModel) -> np.ndarray:
    """Fuse observations ``{"pred": class_id, "aspect_deg": a}`` (or soft ``{"probs": [...], "aspect_deg": a}``)."""
    post = np.asarray(prior, dtype=float).copy()
    for obs in observations:
        L = model.likelihood(obs["aspect_deg"])
        if "probs" in obs:
            # soft observation: marginalise the classifier output over its own confusion
            like = L @ np.asarray(obs["probs"], dtype=float)
        else:
            like = L[:, int(obs["pred"])]
        post *= like
        post /= max(post.sum(), 1e-300)
    return post


def milco_probability(posterior: np.ndarray, class_names: Sequence[str], milco: Sequence[str]) -> float:
    return float(sum(p for p, c in zip(posterior, class_names) if c in milco))
