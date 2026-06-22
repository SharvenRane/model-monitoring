"""Tests: no-drift on same distribution, drift on shifted distribution."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from drift import population_stability_index, ks_drift, check_feature
from monitor import DriftMonitor

RNG = np.random.default_rng(0)


def test_psi_low_for_same_distribution():
    a = RNG.normal(0, 1, 5000)
    b = RNG.normal(0, 1, 5000)
    assert population_stability_index(a, b) < 0.1


def test_psi_high_for_shifted_distribution():
    a = RNG.normal(0, 1, 5000)
    b = RNG.normal(1.5, 1, 5000)  # mean shift
    assert population_stability_index(a, b) > 0.25


def test_ks_flags_shift():
    a = RNG.normal(0, 1, 2000)
    b = RNG.normal(0, 1, 2000)
    assert not ks_drift(a, b).drifted
    c = RNG.normal(0.8, 1, 2000)
    assert ks_drift(a, c).drifted


def test_monitor_reports_drift():
    ref = {"age": RNG.normal(50, 10, 3000), "pred_score": RNG.beta(2, 5, 3000)}
    mon = DriftMonitor(ref)

    # same distribution -> no drift
    same = {"age": RNG.normal(50, 10, 3000), "pred_score": RNG.beta(2, 5, 3000)}
    assert mon.report(same)["drift_detected"] is False

    # shifted score distribution -> drift
    shifted = {"age": RNG.normal(50, 10, 3000), "pred_score": RNG.beta(5, 2, 3000)}
    rep = mon.report(shifted)
    assert rep["drift_detected"] is True
    assert "pred_score" in rep["drifted_features"]


def test_categorical_drift():
    a = RNG.choice(["A", "B", "C"], size=2000, p=[0.6, 0.3, 0.1])
    b = RNG.choice(["A", "B", "C"], size=2000, p=[0.2, 0.3, 0.5])
    res = check_feature("region", a, b, categorical=True)
    assert any(r.drifted for r in res)


if __name__ == "__main__":
    for fn in [test_psi_low_for_same_distribution, test_psi_high_for_shifted_distribution,
               test_ks_flags_shift, test_monitor_reports_drift, test_categorical_drift]:
        fn()
    print("all tests passed")
