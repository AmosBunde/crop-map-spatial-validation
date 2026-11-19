"""Command line: download -> fields -> features -> experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import BANDS, CLASSES, TILES


def cmd_download(a):
    from .download import download

    n = download(Path(a.data), a.tiles, a.dates)
    print(f"downloaded {n} new files into {a.data}")


def cmd_georeference(a):
    from .georef import ORIGIN_TILE3, register

    report = register(Path(a.data))
    print(json.dumps(report, indent=2))
    dx = abs(report["origin_tile3_median"][0] - ORIGIN_TILE3[0])
    dy = abs(report["origin_tile3_median"][1] - ORIGIN_TILE3[1])
    print(f"difference from the origin stored in georef.py: {dx:.2f} m, {dy:.2f} m")


def cmd_fields(a):
    from .fields import build

    gdf = build(Path(a.data), a.tiles or TILES)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out, driver="GPKG")
    lab = gdf[gdf.label > 0]
    print(f"{len(gdf)} fields ({len(lab)} labelled) -> {out}")
    print(lab.label.map(CLASSES).value_counts().to_string())


def cmd_postgis(a):
    import geopandas as gpd

    from . import postgis

    if a.action == "load":
        n = postgis.load(gpd.read_file(a.fields))
        print(f"loaded {n} fields into PostGIS")
    elif a.action == "neighbours":
        print(json.dumps(postgis.neighbour_agreement(postgis.neighbours()), indent=2))
    elif a.action == "blocks":
        df = postgis.block_summary(a.block_m)
        print(f"{len(df)} blocks of {a.block_m:.0f} m; fields per block: "
              f"median {df.n_fields.median():.0f}, max {df.n_fields.max()}, "
              f"{(df.n_fields == 1).sum()} blocks hold a single field")


def cmd_features(a):
    from .features import tile_features

    df = pd.concat([tile_features(Path(a.data), t, a.cloud_threshold) for t in (a.tiles or TILES)], ignore_index=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"{len(df)} field-date rows, {df.date.nunique()} dates -> {out}")


def cmd_experiment(a):
    from .suite import load, macro_f1, run

    fields, x, dates = load(Path(a.fields), Path(a.features))
    y = fields.label.to_numpy() - 1
    proba, n_train = run(fields, x, a.split, a.model, a.seed, a.block_m, a.buffer_m, a.k)
    pred = proba.argmax(axis=1)
    result = {"split": a.split, "model": a.model, "seed": a.seed, "block_m": a.block_m, "buffer_m": a.buffer_m,
              "n_fields": int(len(y)), "n_dates": len(dates), "mean_train_size": n_train,
              "macro_f1": macro_f1(y, pred), "accuracy": float((pred == y).mean())}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def cmd_suite(a):
    from .suite import run_suite, summarise

    run_suite(Path(a.fields), Path(a.features), Path(a.out), a.only)
    s = summarise(Path(a.fields), Path(a.features), Path(a.out))
    for row in s["main"]:
        print(f"{row['split']:6s} {row['model']:8s} macro-F1 {row['macro_f1']:.3f} {row['macro_f1_ci95']}")
    for g in s["gaps"]:
        print(f"{g['model']:4s} {g['comparison']:15s} gap {g['macro_f1_gap']:+.3f} {g['ci95']}")


def cmd_export(a):
    from .export import export

    study = export(Path(a.fields), Path(a.features), Path(a.results), Path(a.out))
    print(f"wrote {a.out}/fields.json and study.json ({study['n_labelled_fields']} labelled fields)")


def cmd_checks(a):
    from .checks import cloud_table

    long = pd.read_parquet(a.features)
    print("mean cloud fraction per date and tile:")
    print(cloud_table(long).to_string())


def main(argv=None):
    p = argparse.ArgumentParser(prog="cropval", description=__doc__)
    sub = p.add_subparsers(required=True)

    d = sub.add_parser("download", help="fetch the dataset (resumable)")
    d.add_argument("--data", default="data")
    d.add_argument("--tiles", nargs="*", choices=TILES)
    d.add_argument("--dates", nargs="*", help="YYYYMMDD; default all 13")
    d.set_defaults(fn=cmd_download)

    r = sub.add_parser("georeference", help="re-derive the map coordinates the rasters lack")
    r.add_argument("--data", default="data")
    r.set_defaults(fn=cmd_georeference)

    f = sub.add_parser("fields", help="polygonise field rasters to a GeoPackage")
    f.add_argument("--data", default="data")
    f.add_argument("--tiles", nargs="*", choices=TILES)
    f.add_argument("--out", default="results/fields.gpkg")
    f.set_defaults(fn=cmd_fields)

    pg = sub.add_parser("postgis", help="load fields into PostGIS and query them")
    pg.add_argument("action", choices=["load", "neighbours", "blocks"])
    pg.add_argument("--fields", default="results/fields.gpkg")
    pg.add_argument("--block-m", type=float, default=2000.0)
    pg.set_defaults(fn=cmd_postgis)

    g = sub.add_parser("features", help="per-field, per-date band means")
    g.add_argument("--data", default="data")
    g.add_argument("--tiles", nargs="*", choices=TILES)
    g.add_argument("--cloud-threshold", type=float, default=50.0)
    g.add_argument("--out", default="results/features.parquet")
    g.set_defaults(fn=cmd_features)

    e = sub.add_parser("experiment", help="cross-validate one model under one split scheme")
    e.add_argument("--fields", default="results/fields.gpkg")
    e.add_argument("--features", default="results/features.parquet")
    e.add_argument("--split", choices=["random", "block", "tile"], required=True)
    e.add_argument("--model", choices=["rf", "tcn"], default="rf")
    e.add_argument("--k", type=int, default=5)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--block-m", type=float, default=2000.0)
    e.add_argument("--buffer-m", type=float, default=0.0)
    e.add_argument("--out", required=True)
    e.set_defaults(fn=cmd_experiment)

    su = sub.add_parser("suite", help="run the whole planned matrix (resumable) and summarise")
    su.add_argument("--fields", default="results/fields.gpkg")
    su.add_argument("--features", default="results/features.parquet")
    su.add_argument("--out", default="results/suite")
    su.add_argument("--only", choices=["main", "buffer", "buffer_control", "block_size"])
    su.set_defaults(fn=cmd_suite)

    ex = sub.add_parser("export", help="write the JSON the D3 page reads")
    ex.add_argument("--fields", default="results/fields.gpkg")
    ex.add_argument("--features", default="results/features.parquet")
    ex.add_argument("--results", default="results/suite")
    ex.add_argument("--out", default="docs/data")
    ex.set_defaults(fn=cmd_export)

    ck = sub.add_parser("checks", help="data checks from the validation plan")
    ck.add_argument("--features", default="results/features.parquet")
    ck.set_defaults(fn=cmd_checks)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
