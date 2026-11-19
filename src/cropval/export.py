"""Write the small JSON files the D3 page reads (docs/data/)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import CLASSES, georef
from .checks import cloud_table, semivariogram
from .features import add_indices


def export(fields_path: Path, features_path: Path, results_dir: Path, out_dir: Path) -> dict:
    import geopandas as gpd
    from shapely import wkt
    from shapely.geometry import box

    from . import postgis

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads((results_dir / "summary.json").read_text())
    oof = pd.read_parquet(results_dir / "oof.parquet")

    gdf = gpd.read_file(fields_path)
    lab = gdf[gdf.label > 0].sort_values(["tile", "field_id"]).reset_index(drop=True)
    pts = gpd.GeoSeries(gpd.points_from_xy(lab.x_m, lab.y_m), crs=gdf.crs).to_crs(4326)
    key = pd.MultiIndex.from_arrays([lab.tile, lab.field_id])

    # The map and the headline tiles show whichever model scores higher on the random split.
    scores = {r["model"]: r["macro_f1"] for r in summary["main"] if r["split"] == "random" and r["model"] != "majority"}
    headline = max(scores, key=scores.get)

    def predictions(split):
        q = oof[(oof.group == "main") & (oof.split == split) & (oof.model == headline) & (oof.seed == 0)]
        q = q.set_index(["tile", "field_id"]).reindex(key)
        if q.pred.isna().any():
            raise ValueError(f"the '{split}' {headline} run (seed 0) does not cover every labelled field")
        return (q.pred == q.y).astype(int).tolist(), q.pred.astype(int).tolist()

    ok_random, _ = predictions("random")
    ok_block, _ = predictions("block")
    ok_tile, pred_tile = predictions("tile")
    fields_json = {
        "classes": [CLASSES[i] for i in sorted(CLASSES)],
        "lon": [round(p.x, 5) for p in pts], "lat": [round(p.y, 5) for p in pts],
        "label": (lab.label - 1).tolist(), "tile": lab.tile.tolist(), "pixels": lab.n_pixels.tolist(),
        "ok_random": ok_random, "ok_block": ok_block, "ok_tile": ok_tile, "pred_tile": pred_tile,
    }
    (out_dir / "fields.json").write_text(json.dumps(fields_json, separators=(",", ":")))

    rows, cols = georef.TILE_SHAPE
    tiles = {}
    for t in georef.LAYOUT:
        tr = georef.transform_for(t)
        outline = gpd.GeoSeries([box(tr.c, tr.f - rows * 10, tr.c + cols * 10, tr.f)], crs=georef.CRS).to_crs(4326).iloc[0]
        tiles[t] = [[round(x, 5), round(y, 5)] for x, y in outline.exterior.coords]

    long = add_indices(pd.read_parquet(features_path))
    peak = long.groupby("date").NDVI.mean().idxmax()
    ndvi = long[long.date == peak].set_index(["tile", "field_id"]).NDVI.reindex(
        pd.MultiIndex.from_arrays([lab.tile, lab.field_id])).to_numpy()
    good = np.isfinite(ndvi)
    vario = semivariogram(lab.x_m.to_numpy()[good], lab.y_m.to_numpy()[good], ndvi[good])
    vario["date"] = peak

    study = {**summary, "headline_model": headline, "tiles": tiles, "variogram": vario,
             "cloud_by_date": {d: {t: float(v) for t, v in row.items()} for d, row in cloud_table(long).iterrows()},
             "field_pixels": {"median": float(lab.n_pixels.median()), "p10": float(lab.n_pixels.quantile(0.1)),
                              "p90": float(lab.n_pixels.quantile(0.9)), "share_under_4": float((lab.n_pixels < 4).mean())}}
    if postgis.available():
        postgis.load(gdf)
        study["neighbours"] = postgis.neighbour_agreement(postgis.neighbours())
        blocks = postgis.block_summary(2000.0)
        study["blocks"] = [{"n": int(r.n_fields), "classes": int(r.n_classes),
                            "ring": [[round(x, 5), round(y, 5)] for x, y in wkt.loads(r.outline_wkt).exterior.coords]}
                           for r in blocks.itertuples()]
    (out_dir / "study.json").write_text(json.dumps(study, separators=(",", ":")))
    return study
