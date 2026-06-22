"""Data and prediction drift detection for deployed models.

Implements the checks you actually run in production monitoring:
  - PSI  (Population Stability Index)  for distribution shift
  - KS test                            for continuous feature drift
  - Chi-square                         for categorical feature drift

Pure numpy/scipy. Compares a frozen *reference* window (e.g. training or the
last validated production window) against a *current* window.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

# Conventional PSI thresholds (credit-risk / MRM practice):
#   < 0.1  no significant shift, 0.1-0.25 moderate, > 0.25 significant.
PSI_MODERATE = 0.10
PSI_SIGNIFICANT = 0.25


@dataclass
class DriftResult:
    feature: str
    metric: str
    value: float
    drifted: bool
    detail: str


def population_stability_index(reference: np.ndarray, current: np.ndarray,
                               bins: int = 10) -> float:
    """PSI between two continuous distributions using reference quantile bins."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    quantiles = np.quantile(ref, np.linspace(0, 1, bins + 1))
    quantiles[0], quantiles[-1] = -np.inf, np.inf
    edges = np.unique(quantiles)
    ref_pct = np.histogram(ref, bins=edges)[0] / len(ref)
    cur_pct = np.histogram(cur, bins=edges)[0] / len(cur)
    eps = 1e-6
    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def ks_drift(reference: np.ndarray, current: np.ndarray, alpha: float = 0.05) -> DriftResult:
    stat, p = stats.ks_2samp(reference, current)
    return DriftResult("", "ks", float(p), bool(p < alpha),
                       f"KS stat={stat:.3f}, p={p:.4f}")


def chi2_drift(reference: np.ndarray, current: np.ndarray, alpha: float = 0.05) -> DriftResult:
    cats = np.unique(np.concatenate([reference, current]))
    ref_c = np.array([(reference == c).sum() for c in cats], dtype=float)
    cur_c = np.array([(current == c).sum() for c in cats], dtype=float)
    # Scale reference to current's total so chi-square compares shapes.
    ref_scaled = ref_c / ref_c.sum() * cur_c.sum()
    stat, p = stats.chisquare(f_obs=np.clip(cur_c, 1e-6, None),
                              f_exp=np.clip(ref_scaled, 1e-6, None))
    return DriftResult("", "chi2", float(p), bool(p < alpha),
                       f"chi2={stat:.3f}, p={p:.4f}")


def check_feature(name: str, reference: np.ndarray, current: np.ndarray,
                  categorical: bool = False) -> list[DriftResult]:
    """Run PSI + the appropriate statistical test for one feature."""
    results = []
    if categorical:
        r = chi2_drift(reference, current)
    else:
        psi = population_stability_index(reference, current)
        results.append(DriftResult(name, "psi", psi, psi > PSI_SIGNIFICANT,
                                    f"PSI={psi:.3f} (>{PSI_SIGNIFICANT} = significant)"))
        r = ks_drift(reference, current)
    r.feature = name
    results.append(r)
    return results
