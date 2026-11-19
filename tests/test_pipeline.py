"""End-to-end checks on a tiny synthetic tile, so the tests need no download."""
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

from cropval import BANDS
from cropval.evaluate import expected_calibration_error
from cropval.suite import impute, macro_f1
from cropval.features import add_indices, tile_features, to_cube
from cropval.fields import polygonise
from cropval.splits import block_ids, folds


def write(path, array, dtype):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=array.shape[0], width=array.shape[1], count=1,
                       dtype=dtype, crs="EPSG:32636", transform=from_origin(600000, 50000, 10, 10)) as dst:
        dst.write(array.astype(dtype), 1)


@pytest.fixture
def tile(tmp_path):
    ids = np.zeros((40, 40), dtype=np.int32)
    labels = np.zeros((40, 40), dtype=np.uint8)
    fid = 0
    for r in range(0, 40, 10):
        for c in range(0, 40, 10):
            fid += 1
            ids[r + 1:r + 9, c + 1:c + 9] = fid
            labels[r + 1:r + 9, c + 1:c + 9] = 1 + fid % 2
    labels[ids == 16] = 0  # one unlabelled (held-back) field
    write(tmp_path / "data/0/0_field_id.tif", ids, "int32")
    write(tmp_path / "data/0/0_label.tif", labels, "uint8")
    for date in ("20190606", "20190701"):
        cloud = np.zeros((40, 40))
        if date == "20190701":
            cloud[ids == 1] = 100  # field 1 fully clouded on the second date
        write(tmp_path / f"data/0/{date}/0_CLD_{date}.tif", cloud, "float32")
        for i, band in enumerate(BANDS):
            write(tmp_path / f"data/0/{date}/0_{band}_{date}.tif", ids * 10.0 + i, "float32")
    return tmp_path


def test_polygonise_gives_one_polygon_per_field(tile):
    gdf = polygonise(tile, "0")
    assert len(gdf) == 16
    assert (gdf.n_pixels == 64).all()
    assert np.allclose(gdf.area_m2, 64 * 100)
    assert gdf.set_index("field_id").label[16] == 0
    assert gdf.set_index("field_id").label[3] == 2


def test_features_are_cloud_masked_field_means(tile):
    df = tile_features(tile, "0")
    assert len(df) == 16 * 2
    row = df[(df.field_id == 5) & (df.date == "20190606")].iloc[0]
    assert row.B01 == pytest.approx(50.0) and row.B02 == pytest.approx(51.0)
    clouded = df[(df.field_id == 1) & (df.date == "20190701")].iloc[0]
    assert clouded.cloud_frac == 1.0 and np.isnan(clouded.B04)
    keys, dates, cube = to_cube(add_indices(df), BANDS + ["NDVI"])
    assert cube.shape == (16, 2, len(BANDS) + 1) and not np.isnan(cube).any()


def _fields(n=400, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"tile": rng.choice(["0", "1"], n), "x_m": rng.uniform(0, 20000, n),
                         "y_m": rng.uniform(0, 20000, n), "label": rng.integers(1, 4, n)})


def test_block_split_never_shares_a_block(tile):
    f = _fields()
    blocks = block_ids(f, 2000)
    for train, test in folds(f, "block", k=5, block_m=2000):
        assert not set(blocks[train]) & set(blocks[test])


def test_buffer_removes_training_fields_near_test_fields():
    f = _fields()
    xy = f[["x_m", "y_m"]].to_numpy()
    for train, test in folds(f, "block", k=5, block_m=2000, buffer_m=500):
        for t in set(f.tile):
            a, b = train[f.tile.to_numpy()[train] == t], test[f.tile.to_numpy()[test] == t]
            if len(a) and len(b):
                d = np.linalg.norm(xy[a][:, None] - xy[b][None], axis=2).min()
                assert d >= 500


def test_tile_split_holds_out_whole_tiles():
    f = _fields()
    for train, test in folds(f, "tile"):
        assert not set(f.tile.to_numpy()[train]) & set(f.tile.to_numpy()[test])


def test_mosaic_layout_places_tiles_edge_to_edge():
    from cropval import georef

    rows, cols = georef.TILE_SHAPE
    t = {k: georef.transform_for(k) for k in "0123"}
    assert t["3"].c - t["1"].c == cols * 10 and t["3"].f == t["1"].f      # 1 | 3
    assert t["1"].f - t["0"].f == rows * 10 and t["1"].c == t["0"].c      # 1 over 0
    assert (t["2"].c, t["2"].f) == (t["3"].c, t["0"].f)


def test_ncc_finds_a_planted_chip():
    from cropval.georef import ncc

    rng = np.random.default_rng(0)
    image = rng.normal(size=(120, 150))
    score = ncc(image, image[40:70, 55:95])
    assert np.unravel_index(score.argmax(), score.shape) == (40, 55)
    assert score.max() == pytest.approx(1.0, abs=1e-6)


def test_macro_f1_matches_sklearn_and_skips_absent_classes():
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(0)
    y, pred = rng.integers(0, 7, 500), rng.integers(0, 7, 500)
    assert macro_f1(y, pred) == pytest.approx(f1_score(y, pred, average="macro"))
    # A class with no true examples is left out of the average, not scored as zero.
    y2, pred2 = np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1])
    assert macro_f1(y2, pred2) == 1.0
    assert macro_f1(np.array([0, 0, 1, 1]), np.array([0, 5, 1, 1])) == pytest.approx((2 / 3 + 1.0) / 2)


def test_imputation_uses_training_rows_only():
    train = np.array([[[1.0], [np.nan]], [[3.0], [5.0]]])
    test = np.array([[[np.nan], [np.nan]]])
    a, b = impute(train, test)
    assert a[0, 1, 0] == 5.0 and b[0, 0, 0] == 2.0 and b[0, 1, 0] == 5.0
    huge = np.full((1, 2, 1), 1e9)
    assert np.array_equal(impute(train, huge)[0], a)


def test_gap_filling_stays_inside_each_field(tile):
    df = tile_features(tile, "0")
    df.loc[df.field_id == 2, BANDS] = np.nan  # field 2 never seen clearly
    keys, dates, cube = to_cube(add_indices(df), BANDS)
    assert np.isnan(cube[1]).all() and not np.isnan(np.delete(cube, 1, axis=0)).any()


def test_calibration_error_is_zero_for_perfect_confident_predictions():
    y = np.array([0, 1, 2, 0, 1, 2])
    assert expected_calibration_error(np.eye(3)[y], y) == pytest.approx(0.0)
