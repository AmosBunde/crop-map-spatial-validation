"""Raster to vector: turn the field-id raster into one polygon per field.

The dataset ships fields as rasters (a field-id image and a label image per
tile). Spatial validation needs geometry, so we polygonise the field-id raster
and attach the crop label, pixel count, area and centroid to each field.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from shapely.geometry import shape
from shapely.ops import unary_union

from . import CLASSES


def read_tile(data_dir: Path, tile: str):
    """Return (field_id array, label array, transform, crs) for one tile."""
    base = data_dir / "data" / tile
    with rasterio.open(base / f"{tile}_field_id.tif") as src:
        ids = src.read(1)
        transform, crs = src.transform, src.crs
    if crs is None:  # the published rasters are not georeferenced; see georef.py
        from . import georef

        transform, crs = georef.transform_for(tile), georef.CRS
    with rasterio.open(base / f"{tile}_label.tif") as src:
        labels = src.read(1)
    if ids.shape != labels.shape:
        raise ValueError(f"tile {tile}: field-id and label rasters differ in shape")
    return ids, labels, transform, crs


def polygonise(data_dir: Path, tile: str) -> gpd.GeoDataFrame:
    """One row per field: geometry, label, pixel count. Label 0 means unlabelled."""
    ids, labels, transform, crs = read_tile(data_dir, tile)
    ids = ids.astype(np.int32)
    parts: dict[int, list] = {}
    for geom, value in features.shapes(ids, mask=ids > 0, transform=transform):
        parts.setdefault(int(value), []).append(shape(geom))

    flat_ids, flat_lab = ids.ravel(), labels.ravel().astype(np.int64)
    n = int(flat_ids.max()) + 1
    counts = np.bincount(flat_ids, minlength=n)
    # A field's label is the most common non-zero label among its pixels.
    label_of = np.zeros(n, dtype=np.int64)
    purity = np.zeros(n)
    for c in CLASSES:
        votes = np.bincount(flat_ids, weights=(flat_lab == c), minlength=n)
        better = votes > purity
        label_of[better], purity[better] = c, votes[better]

    rows = [
        {
            "tile": tile,
            "field_id": fid,
            "label": int(label_of[fid]),
            "n_pixels": int(counts[fid]),
            "label_purity": float(purity[fid] / counts[fid]) if counts[fid] else 0.0,
            "geometry": unary_union(geoms),
        }
        for fid, geoms in parts.items()
    ]
    gdf = gpd.GeoDataFrame(rows, crs=crs)
    metric = gdf if gdf.crs.is_projected else gdf.to_crs(gdf.estimate_utm_crs())
    gdf["area_m2"] = metric.area
    gdf["x_m"], gdf["y_m"] = metric.centroid.x, metric.centroid.y
    return gdf


def build(data_dir: Path, tiles: list[str]) -> gpd.GeoDataFrame:
    import pandas as pd

    frames = [polygonise(data_dir, t) for t in tiles]
    crs = frames[0].crs
    return gpd.GeoDataFrame(pd.concat([f.to_crs(crs) for f in frames], ignore_index=True), crs=crs)
