"""PostGIS backend for the field layer.

The fields live in PostGIS as polygons in EPSG:32636. The block assignment and the
buffer exclusion that define the spatial splits are also written as SQL (see sql/),
and the tests check that they return exactly what the in-memory versions in
splits.py return. PostGIS additionally answers the questions that are awkward in
NumPy: nearest-neighbour label agreement and per-block inventories.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

DSN = os.environ.get("CROPVAL_DSN", "postgresql://cropval:cropval@127.0.0.1:5433/cropval")
SQL = Path(__file__).resolve().parents[2] / "sql"


def connect(dsn: str = DSN):
    import psycopg

    return psycopg.connect(dsn)


def _sql(name: str) -> str:
    return (SQL / name).read_text()


def load(gdf, dsn: str = DSN) -> int:
    """(Re)create the schema and insert every field. Returns the row count."""
    from shapely import to_wkb
    from shapely.geometry import MultiPolygon

    epsg = gdf.crs.to_epsg()
    if epsg != 32636:
        gdf = gdf.to_crs(32636)
    rows = [
        (r.tile, int(r.field_id), int(r.label), int(r.n_pixels), float(r.area_m2),
         to_wkb(r.geometry if r.geometry.geom_type == "MultiPolygon" else MultiPolygon([r.geometry])))
        for r in gdf.itertuples()
    ]
    with connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(_sql("schema.sql"))
        cur.executemany(
            "INSERT INTO fields (tile, field_id, label, n_pixels, area_m2, geom) "
            "VALUES (%s, %s, %s, %s, %s, ST_SetSRID(ST_GeomFromWKB(%s), 32636))",
            rows,
        )
        cur.execute("ANALYZE fields")
        cur.execute("SELECT count(*) FROM fields")
        return cur.fetchone()[0]


def block_table(block_m: float, dsn: str = DSN) -> pd.DataFrame:
    with connect(dsn) as conn:
        rows = conn.execute(_sql("blocks.sql"), {"block_m": block_m}).fetchall()
    return pd.DataFrame(rows, columns=["tile", "field_id", "bx", "by"])


def buffer_exclusions(train: pd.DataFrame, test: pd.DataFrame, buffer_m: float, dsn: str = DSN) -> set[tuple[str, int]]:
    """Keys of training fields within `buffer_m` of any test field."""
    params = {
        "buffer_m": buffer_m,
        "train_tiles": train.tile.tolist(), "train_ids": train.field_id.astype(int).tolist(),
        "test_tiles": test.tile.tolist(), "test_ids": test.field_id.astype(int).tolist(),
    }
    with connect(dsn) as conn:
        return {(t, int(i)) for t, i in conn.execute(_sql("buffer.sql"), params).fetchall()}


def neighbours(dsn: str = DSN) -> pd.DataFrame:
    with connect(dsn) as conn:
        rows = conn.execute(_sql("neighbours.sql")).fetchall()
    return pd.DataFrame(rows, columns=["tile", "field_id", "label", "neighbour_label", "distance_m"])


def neighbour_agreement(df: pd.DataFrame) -> dict:
    """Share of fields whose nearest labelled neighbour grows the same crop, against chance."""
    observed = float((df.label == df.neighbour_label).mean())
    share = df.label.value_counts(normalize=True)
    chance = float((share**2).sum())  # two fields drawn at random share a label
    return {
        "n_fields": int(len(df)),
        "same_crop_as_nearest_neighbour": observed,
        "expected_by_chance": chance,
        "ratio": observed / chance if chance else float("nan"),
        "median_neighbour_distance_m": float(df.distance_m.median()),
        "p90_neighbour_distance_m": float(df.distance_m.quantile(0.9)),
    }


def block_summary(block_m: float, dsn: str = DSN) -> pd.DataFrame:
    with connect(dsn) as conn:
        rows = conn.execute(_sql("block_summary.sql"), {"block_m": block_m}).fetchall()
    return pd.DataFrame(rows, columns=["bx", "by", "n_fields", "n_classes", "majority_label", "outline_wkt"])


def available(dsn: str = DSN) -> bool:
    try:
        with connect(dsn) as conn:
            conn.execute("SELECT PostGIS_Version()")
        return True
    except Exception:
        return False
