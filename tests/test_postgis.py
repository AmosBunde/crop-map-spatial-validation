"""PostGIS must agree with the in-memory split code. Skipped when no database is up."""
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from cropval import postgis
from cropval.splits import block_ids, folds

pytestmark = pytest.mark.skipif(not postgis.available(), reason="PostGIS not reachable (docker compose up -d)")

# The tests load synthetic fields, so they use their own database and never touch
# the one holding the real field layer.
TEST_DSN = postgis.DSN.rsplit("/", 1)[0] + "/cropval_test"


@pytest.fixture(scope="module", autouse=True)
def test_database():
    import psycopg

    with psycopg.connect(postgis.DSN, autocommit=True) as conn:
        if not conn.execute("SELECT 1 FROM pg_database WHERE datname = 'cropval_test'").fetchone():
            conn.execute("CREATE DATABASE cropval_test")


@pytest.fixture(scope="module")
def fields(test_database):
    rng = np.random.default_rng(1)
    n = 300
    x = rng.uniform(620000, 640000, n)
    y = rng.uniform(20000, 45000, n)
    gdf = gpd.GeoDataFrame(
        {"tile": rng.choice(["0", "1"], n), "field_id": np.arange(1, n + 1), "label": rng.integers(1, 4, n),
         "n_pixels": 9, "area_m2": 900.0},
        geometry=[box(a - 15, b - 15, a + 15, b + 15) for a, b in zip(x, y)], crs="EPSG:32636",
    )
    gdf["x_m"], gdf["y_m"] = gdf.centroid.x, gdf.centroid.y
    assert postgis.load(gdf, TEST_DSN) == n
    return gdf


def test_block_assignment_matches_python(fields):
    db = postgis.block_table(2000, TEST_DSN).merge(fields[["tile", "field_id"]].reset_index(), on=["tile", "field_id"]).sort_values("index")
    from_db = pd.factorize(pd.Series(list(zip(db.bx, db.by))))[0]
    from_py = block_ids(fields, 2000)
    # Same partition of fields into blocks, whatever the block numbering.
    assert len(set(zip(from_db, from_py))) == len(set(from_db)) == len(set(from_py))


def test_buffer_exclusion_matches_python(fields):
    plain = list(folds(fields, "block", k=5, block_m=2000))
    buffered = list(folds(fields, "block", k=5, block_m=2000, buffer_m=750))
    for (train, test), (kept, _) in zip(plain, buffered):
        dropped_py = {(fields.tile[i], int(fields.field_id[i])) for i in set(train) - set(kept)}
        dropped_db = postgis.buffer_exclusions(fields.iloc[train], fields.iloc[test], 750, TEST_DSN)
        assert dropped_db == dropped_py


def test_neighbour_agreement_is_near_chance_for_random_labels(fields):
    stats = postgis.neighbour_agreement(postgis.neighbours(TEST_DSN))
    assert stats["n_fields"] == len(fields)
    assert 0.6 < stats["ratio"] < 1.5
