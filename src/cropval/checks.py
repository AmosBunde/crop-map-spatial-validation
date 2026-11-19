"""Data checks from the validation plan that are run before modelling."""
from __future__ import annotations

import numpy as np
import pandas as pd


def cloud_table(long: pd.DataFrame) -> pd.DataFrame:
    """Mean cloud fraction over fields, per tile and date."""
    return long.pivot_table(index="date", columns="tile", values="cloud_frac", aggfunc="mean").round(3)


def semivariogram(x_m, y_m, values, max_km: float = 12.0, bins: int = 24, pairs: int = 400_000, seed: int = 0):
    """Empirical semivariogram from random field pairs.

    Semivariance at lag h is half the mean squared difference between values h apart.
    It rises with distance while nearby fields are alike, and flattens (the "range")
    once they are no more alike than any two fields.
    """
    rng = np.random.default_rng(seed)
    n = len(values)
    i, j = rng.integers(0, n, pairs), rng.integers(0, n, pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    d = np.hypot(x_m[i] - x_m[j], y_m[i] - y_m[j]) / 1000.0
    g = 0.5 * (values[i] - values[j]) ** 2
    edges = np.linspace(0, max_km, bins + 1)
    idx = np.digitize(d, edges) - 1
    out = []
    for b in range(bins):
        m = idx == b
        if m.sum() >= 30:
            out.append({"lag_km": float((edges[b] + edges[b + 1]) / 2), "semivariance": float(g[m].mean()), "pairs": int(m.sum())})
    return {"bins": out, "sill": float(np.var(values))}
