"""Calibration metric. Macro-F1 and the block bootstrap live in suite.py."""
from __future__ import annotations

import numpy as np


def expected_calibration_error(proba: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    conf, pred = proba.max(axis=1), proba.argmax(axis=1)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(ece)
