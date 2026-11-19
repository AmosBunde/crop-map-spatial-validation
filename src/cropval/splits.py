"""Three ways to hold out fields, from most to least optimistic.

random   Stratified k-fold over fields. Neighbouring fields land on both
         sides of the split, so spatial autocorrelation leaks into the score.
block    Fields are grouped into square blocks (default 2 km); whole blocks
         are held out. An optional buffer drops training fields that sit
         within a set distance of any test field.
tile     Leave one Sentinel-2 tile out: train on three, test on the fourth.
         The closest thing here to mapping a district with no local labels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def block_ids(fields: pd.DataFrame, block_m: float = 2000.0) -> np.ndarray:
    """Blocks are cut on the shared map grid, so they can straddle tile edges."""
    bx = np.floor(fields.x_m.to_numpy() / block_m).astype(np.int64)
    by = np.floor(fields.y_m.to_numpy() / block_m).astype(np.int64)
    return pd.factorize(pd.Series(list(zip(bx, by))))[0]


def random_folds(fields: pd.DataFrame, k: int = 5, seed: int = 0):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    yield from skf.split(np.zeros(len(fields)), fields.label)


def block_folds(fields: pd.DataFrame, k: int = 5, seed: int = 0, block_m: float = 2000.0, buffer_m: float = 0.0):
    groups = block_ids(fields, block_m)
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    xy = fields[["x_m", "y_m"]].to_numpy()
    for train, test in sgkf.split(np.zeros(len(fields)), fields.label, groups):
        if buffer_m > 0:
            near = cKDTree(xy[test]).query(xy[train], distance_upper_bound=buffer_m)[0]
            train = train[~np.isfinite(near)]
        yield train, test


def tile_folds(fields: pd.DataFrame):
    tiles = fields.tile.to_numpy()
    for t in sorted(set(tiles)):
        yield np.flatnonzero(tiles != t), np.flatnonzero(tiles == t)


def folds(fields: pd.DataFrame, scheme: str, **kw):
    if scheme == "random":
        return random_folds(fields, kw.get("k", 5), kw.get("seed", 0))
    if scheme == "block":
        return block_folds(fields, kw.get("k", 5), kw.get("seed", 0), kw.get("block_m", 2000.0), kw.get("buffer_m", 0.0))
    if scheme == "tile":
        return tile_folds(fields)
    raise ValueError(f"unknown split scheme: {scheme}")
