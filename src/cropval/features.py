"""Per-field, per-date band statistics from the Sentinel-2 rasters.

Every band is on the same grid as the field-id raster, so a field's mean
reflectance is a grouped sum over pixels (np.bincount), with no resampling.
Cloudy pixels are excluded using the dataset's cloud-probability band (CLD);
the share of cloudy pixels is kept as a feature and as a quality flag.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from . import BANDS


def tile_dates(data_dir: Path, tile: str) -> list[str]:
    return sorted(p.name for p in (data_dir / "data" / tile).iterdir() if p.is_dir())


def _read(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float64)


def tile_features(data_dir: Path, tile: str, cloud_threshold: float = 50.0) -> pd.DataFrame:
    """Long table: one row per (field, date) with cloud-masked band means."""
    base = data_dir / "data" / tile
    with rasterio.open(base / f"{tile}_field_id.tif") as src:
        ids = src.read(1).astype(np.int64).ravel()
    n = int(ids.max()) + 1
    field_ids = np.flatnonzero(np.bincount(ids, minlength=n)[1:]) + 1
    n_pix = np.bincount(ids, minlength=n)[field_ids]

    frames = []
    for date in tile_dates(data_dir, tile):
        cld_path = base / date / f"{tile}_CLD_{date}.tif"
        cloud = _read(cld_path).ravel() if cld_path.exists() else np.zeros(ids.shape)
        clear = (cloud < cloud_threshold).astype(np.float64)
        n_clear = np.bincount(ids, weights=clear, minlength=n)[field_ids]
        row = {
            "tile": tile,
            "field_id": field_ids,
            "date": date,
            "cloud_frac": 1.0 - n_clear / n_pix,
        }
        for band in BANDS:
            values = _read(base / date / f"{tile}_{band}_{date}.tif").ravel()
            total = np.bincount(ids, weights=values * clear, minlength=n)[field_ids]
            with np.errstate(invalid="ignore", divide="ignore"):
                row[band] = np.where(n_clear > 0, total / n_clear, np.nan)
        frames.append(pd.DataFrame(row))
    return pd.concat(frames, ignore_index=True)


def add_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Vegetation and water indices commonly used for crop discrimination."""
    out = df.copy()
    eps = 1e-6
    out["NDVI"] = (out.B08 - out.B04) / (out.B08 + out.B04 + eps)
    out["NDRE"] = (out.B8A - out.B05) / (out.B8A + out.B05 + eps)
    out["NDWI"] = (out.B8A - out.B11) / (out.B8A + out.B11 + eps)
    return out


def to_cube(long: pd.DataFrame, columns: list[str]):
    """Pivot the long table to (fields, dates, features).

    Cloud gaps are filled inside each field's own series, by linear interpolation
    over the real acquisition dates (the gaps range from 5 to 30 days), holding the
    first and last clear value at the ends. A field with no clear observation of a
    feature on any date stays NaN: that is imputed later, inside each training fold,
    so no statistic of held-out fields can reach the model.
    Returns (keys DataFrame[tile, field_id], dates list, float32 array).
    """
    dates = sorted(long.date.unique())
    days = pd.to_datetime(pd.Series(dates), format="%Y%m%d")
    days = ((days - days.iloc[0]).dt.days).to_numpy(dtype=float)
    keys = long[["tile", "field_id"]].drop_duplicates().sort_values(["tile", "field_id"]).reset_index(drop=True)
    index = pd.MultiIndex.from_frame(keys)
    cube = np.empty((len(keys), len(dates), len(columns)), dtype=np.float32)
    for j, col in enumerate(columns):
        wide = long.pivot_table(index=["tile", "field_id"], columns="date", values=col, dropna=False)
        values = wide.reindex(index=index, columns=dates).to_numpy(dtype=float).copy()
        for i in np.flatnonzero(np.isnan(values).any(axis=1)):
            ok = ~np.isnan(values[i])
            if ok.any():
                values[i] = np.interp(days, days[ok], values[i, ok])
        cube[:, :, j] = values
    return keys, dates, cube
