"""Recover the map coordinates the dataset was published without.

The CV4A Kenya rasters carry no CRS or geotransform. Two facts restore them:

1. The four tiles are one contiguous mosaic. Edge rows and columns of
   neighbouring tiles correlate at 0.97-0.99 (0.90 for adjacent columns
   inside a tile), giving the layout   1 | 3
                                       --+--
                                       0 | 2
2. Tile 3 overlaps the three tiles of the georeferenced sibling dataset
   (african-crops-kenya-01, EPSG:32636, 10 m). For the same acquisition date
   (2019-06-06, band B04) a 400 x 400 px chip from each sibling tile matches
   tile 3 with a normalised cross-correlation of 1.000 at a single offset.

The three matches place tile 3's upper-left corner within 2.6 m of each other
in x and 0.7 m in y, well under one 10 m pixel. We use their median.
`register()` reproduces both checks from the raw files.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from rasterio.transform import Affine

CRS = "EPSG:32636"  # WGS 84 / UTM zone 36N
PIXEL_M = 10.0
TILE_SHAPE = (3035, 2016)  # rows, cols
LAYOUT = {"1": (0, 0), "3": (0, 1), "0": (1, 0), "2": (1, 1)}  # tile -> (row, col) in the mosaic
ORIGIN_TILE3 = (633898.97, 79157.78)  # upper-left x, y in metres; median of three registrations

SIBLING = "https://data.source.coop/radiantearth/african-crops-kenya-01/imagery/{t}/{t}_{d}/{t}_{d}_B04_10m.tif"


def transform_for(tile: str) -> Affine:
    """Geotransform of a tile, derived from tile 3's origin and the mosaic layout."""
    rows, cols = TILE_SHAPE
    r3, c3 = LAYOUT["3"]
    r, c = LAYOUT[tile]
    x0 = ORIGIN_TILE3[0] + (c - c3) * cols * PIXEL_M
    y0 = ORIGIN_TILE3[1] - (r - r3) * rows * PIXEL_M
    return Affine(PIXEL_M, 0.0, x0, 0.0, -PIXEL_M, y0)


def ncc(image: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Normalised cross-correlation of `template` at every valid position in `image`."""
    from scipy.signal import fftconvolve

    t = template - template.mean()
    ones = np.ones_like(t)
    num = fftconvolve(image, t[::-1, ::-1], mode="valid")
    s1 = fftconvolve(image, ones, mode="valid")
    s2 = fftconvolve(image**2, ones, mode="valid")
    den = np.sqrt(np.maximum(s2 - s1**2 / t.size, 1e-12)) * np.sqrt((t**2).sum())
    return num / den


def register(data_dir: Path, date: str = "20190606", chip: int = 400) -> dict:
    """Re-derive the layout and tile 3's origin. Needs band B04 of `date` for all four tiles."""
    import tempfile
    import urllib.request

    import rasterio

    band = {}
    for tile in LAYOUT:
        with rasterio.open(data_dir / "data" / tile / date / f"{tile}_B04_{date}.tif") as src:
            band[tile] = src.read(1).astype(np.float64)

    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1])

    edges = {
        "1 above 0": corr(band["1"][-1, :], band["0"][0, :]),
        "3 above 2": corr(band["3"][-1, :], band["2"][0, :]),
        "1 left of 3": corr(band["1"][:, -1], band["3"][:, 0]),
        "0 left of 2": corr(band["0"][:, -1], band["2"][:, 0]),
        "baseline (adjacent columns inside tile 0)": corr(band["0"][:, 1000], band["0"][:, 1001]),
    }

    origins = []
    with tempfile.TemporaryDirectory() as tmp:
        for sib in ("00", "01", "02"):
            path = Path(tmp) / f"{sib}.tif"
            urllib.request.urlretrieve(SIBLING.format(t=sib, d=date), path)
            with rasterio.open(path) as src:
                ref, tr = src.read(1).astype(np.float64), src.transform
            h, w = ref.shape
            r0, c0 = h // 2 - chip // 2, w // 2 - chip // 2
            score = ncc(band["3"], ref[r0:r0 + chip, c0:c0 + chip])
            r, c = np.unravel_index(score.argmax(), score.shape)
            origins.append({
                "sibling_tile": sib,
                "peak_ncc": float(score.max()),
                "x": tr.c + (c0 - c) * PIXEL_M,
                "y": tr.f - (r0 - r) * PIXEL_M,
            })
    xs, ys = [o["x"] for o in origins], [o["y"] for o in origins]
    return {
        "edge_correlations": edges,
        "registrations": origins,
        "origin_tile3_median": [float(np.median(xs)), float(np.median(ys))],
        "spread_m": [float(max(xs) - min(xs)), float(max(ys) - min(ys))],
    }
