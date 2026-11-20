"""Describe a field's season by the parameters of a growth curve.

For one vegetation index, a field's series is modelled as

    index(t) = base + amplitude * bell(t; peak, rise, fall),   amplitude >= 0

where ``bell`` is a Gaussian with one width before the peak and another after it.
``(peak, rise, fall)`` is chosen from a fixed grid; for each candidate, ``base`` and
``amplitude`` are the least-squares solution in closed form. The fit uses the field's own
series only, so it can be computed for every field before any train/test split.
"""
from __future__ import annotations

import numpy as np

PEAKS = np.arange(-30.0, 181.0, 10.0)            # days after the first acquisition
WIDTHS = np.array([15.0, 25.0, 40.0, 60.0, 90.0])
INDICES = ("NDVI", "NDRE", "NDWI")
PARAMETERS = ("base", "amplitude", "peak", "rise", "fall", "rmse")


def candidates() -> np.ndarray:
    """Every (peak, rise, fall) on the grid, shape (C, 3)."""
    p, r, f = np.meshgrid(PEAKS, WIDTHS, WIDTHS, indexing="ij")
    return np.stack([p.ravel(), r.ravel(), f.ravel()], axis=1)


def bells(days: np.ndarray, cand: np.ndarray) -> np.ndarray:
    """The bell curve of each candidate at each day, shape (C, T)."""
    dt = days[None, :] - cand[:, [0]]
    width = np.where(dt < 0, cand[:, [1]], cand[:, [2]])
    return np.exp(-((dt / width) ** 2))


def fit_curve(series: np.ndarray, days: np.ndarray) -> np.ndarray:
    """Fit every row of ``series`` (N, T). Returns (N, 6): base, amplitude, peak, rise, fall, rmse.

    A row containing NaN returns NaN. A row no bell improves on (amplitude would be
    negative for every candidate) returns amplitude 0 and the first candidate's shape.
    """
    series = np.asarray(series, dtype=np.float64)
    cand = candidates()
    b = bells(np.asarray(days, dtype=np.float64), cand)
    bc = b - b.mean(axis=1, keepdims=True)
    ss_b = (bc ** 2).sum(axis=1)
    ok = np.isfinite(series).all(axis=1)
    y = np.where(ok[:, None], series, 0.0)
    yc = y - y.mean(axis=1, keepdims=True)
    ss_y = (yc ** 2).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        amp = np.where(ss_b > 1e-12, (yc @ bc.T) / ss_b, 0.0)        # (N, C)
    amp = np.maximum(amp, 0.0)
    sse = ss_y[:, None] - amp ** 2 * ss_b[None, :]
    best = sse.argmin(axis=1)
    rows = np.arange(len(y))
    a = amp[rows, best]
    base = y.mean(axis=1) - a * b.mean(axis=1)[best]
    rmse = np.sqrt(np.maximum(sse[rows, best], 0.0) / series.shape[1])
    out = np.column_stack([base, a, cand[best, 0], cand[best, 1], cand[best, 2], rmse])
    out[~ok] = np.nan
    return out


def phenology_parameters(x: np.ndarray, days: np.ndarray, columns: list[str]) -> np.ndarray:
    """(fields, dates, features) -> (fields, 1, 18): six curve parameters for each of three indices.

    The middle axis of length one keeps the shape the models expect.
    """
    parts = [fit_curve(x[:, :, columns.index(name)], days) for name in INDICES]
    return np.concatenate(parts, axis=1)[:, None, :].astype(np.float32)
