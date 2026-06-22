"""DriftMonitor: track a deployed model's input features and output scores
against a frozen reference window, and emit a drift report.

This is the runtime piece that, in the PCCP/governance story, triggers
re-evaluation when production data shifts beyond threshold.
"""
from __future__ import annotations

import json

import numpy as np

from drift import check_feature, DriftResult


class DriftMonitor:
    def __init__(self, reference: dict[str, np.ndarray], categorical: set[str] | None = None):
        """reference: {feature_name -> 1D array} from the validated window.
        Include the model's output score as a feature (e.g. 'pred_score')."""
        self.reference = {k: np.asarray(v) for k, v in reference.items()}
        self.categorical = categorical or set()

    def check(self, current: dict[str, np.ndarray]) -> list[DriftResult]:
        out: list[DriftResult] = []
        for name, ref in self.reference.items():
            if name not in current:
                continue
            out += check_feature(name, ref, np.asarray(current[name]),
                                 categorical=name in self.categorical)
        return out

    def report(self, current: dict[str, np.ndarray]) -> dict:
        results = self.check(current)
        drifted = [r for r in results if r.drifted]
        return {
            "drift_detected": len(drifted) > 0,
            "n_features_checked": len(self.reference),
            "n_signals_drifted": len(drifted),
            "drifted_features": sorted({r.feature for r in drifted}),
            "results": [vars(r) for r in results],
        }

    def report_json(self, current: dict[str, np.ndarray]) -> str:
        return json.dumps(self.report(current), indent=2)
