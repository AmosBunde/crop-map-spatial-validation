"""Run the planned experiment matrix and summarise it.

Every run stores its out-of-fold prediction for every field. All reported
numbers, intervals and paired differences are then computed from those stored
predictions, on one shared set of bootstrap resamples.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import BANDS, CLASSES
from .evaluate import expected_calibration_error
from .features import add_indices, to_cube
from .models import MODELS
from .phenology import phenology_parameters
from .splits import block_ids, folds

COLUMNS = BANDS + ["NDVI", "NDRE", "NDWI", "cloud_frac"]
SEEDS = [0, 1, 2, 3, 4]
BUFFERS = [0, 250, 500, 1000, 2000]
N_CLASSES = len(CLASSES)
KEY = ["group", "split", "model", "seed", "block_m", "buffer_m"]


def load(fields_path: Path, features_path: Path):
    """Labelled fields (sorted by tile, field id) and their (dates x features) series."""
    import geopandas as gpd

    fields = gpd.read_file(fields_path).drop(columns="geometry")
    fields = fields[fields.label > 0]
    keys, dates, cube = to_cube(add_indices(pd.read_parquet(features_path)), COLUMNS)
    fields = keys.reset_index().merge(fields, on=["tile", "field_id"]).sort_values(["tile", "field_id"]).reset_index(drop=True)
    return fields, cube[fields["index"].to_numpy()], dates


def days_since_first(dates: list[str]) -> np.ndarray:
    d = pd.to_datetime(pd.Series(dates), format="%Y%m%d")
    return (d - d.iloc[0]).dt.days.to_numpy(dtype=float)


def impute(x_train: np.ndarray, x_test: np.ndarray):
    """Fill values a field never observed (cloud on every date) with the training median."""
    if not (np.isnan(x_train).any() or np.isnan(x_test).any()):
        return x_train, x_test
    med = np.nan_to_num(np.nanmedian(x_train, axis=0), nan=0.0)
    return np.where(np.isnan(x_train), med, x_train), np.where(np.isnan(x_test), med, x_test)


def run(fields, x, split, model, seed, block_m=2000.0, buffer_m=0.0, k=5, match_buffer_m=0.0):
    """Out-of-fold class probabilities for every field.

    `match_buffer_m` is the size-matched control for the buffer experiment: train
    with no buffer, but on a random subset as large as the buffered training set.
    """
    y = fields.label.to_numpy() - 1
    proba = np.zeros((len(y), N_CLASSES), dtype=np.float32)
    sizes = []
    plain = folds(fields, split, k=k, seed=seed, block_m=block_m, buffer_m=buffer_m)
    reference = folds(fields, split, k=k, seed=seed, block_m=block_m, buffer_m=match_buffer_m) if match_buffer_m else None
    rng = np.random.default_rng(seed)
    for train, test in plain:
        if reference is not None:
            ref_train, ref_test = next(reference)
            assert np.array_equal(ref_test, test)
            train = rng.choice(train, size=len(ref_train), replace=False)
        sizes.append(len(train))
        if model == "majority":
            proba[test, np.bincount(y[train], minlength=N_CLASSES).argmax()] = 1.0
        else:
            x_train, x_test = impute(x[train], x[test])
            proba[test] = MODELS[model](x_train, y[train], x_test, N_CLASSES, seed=seed)
    return proba, float(np.mean(sizes))


def matrix():
    """(group, split, model, seed, block_m, buffer_m) for every planned run."""
    out = []
    for seed in SEEDS:
        for split in ("random", "block", "tile"):
            for model in ("majority", "rf", "tcn"):
                out.append(("main", split, model, seed, 2000.0, 0.0))
        for buffer_m in BUFFERS[1:]:
            for model in ("rf", "tcn"):
                out.append(("buffer", "block", model, seed, 2000.0, float(buffer_m)))
                out.append(("buffer_control", "block", model, seed, 2000.0, float(buffer_m)))
        for block_m in (1000.0, 5000.0):
            out.append(("block_size", "block", "rf", seed, block_m, 0.0))
        for split in ("random", "block"):                      # follow-up study, see FOLLOW_UP_PLAN.md
            out.append(("phenology", split, "pheno", seed, 2000.0, 0.0))
    return out


def run_suite(fields_path: Path, features_path: Path, out_dir: Path, only: str | None = None):
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_path, oof_path = out_dir / "runs.parquet", out_dir / "oof.parquet"
    runs = pd.read_parquet(runs_path) if runs_path.exists() else pd.DataFrame()
    oof = pd.read_parquet(oof_path) if oof_path.exists() else pd.DataFrame()
    done = set(map(tuple, runs[KEY].itertuples(index=False))) if len(runs) else set()
    todo = [m for m in matrix() if m not in done and (only is None or m[0] == only)]
    if not todo:
        return
    fields, x, dates = load(fields_path, features_path)
    y = fields.label.to_numpy() - 1
    x_pheno = None
    for i, (group, split, model, seed, block_m, buffer_m) in enumerate(todo):
        t0 = time.time()
        control = group == "buffer_control"
        if model == "pheno" and x_pheno is None:
            x_pheno = phenology_parameters(x, days_since_first(dates), COLUMNS)
        proba, n_train = run(fields, x_pheno if model == "pheno" else x, split, model, seed, block_m,
                             0.0 if control else buffer_m,
                             match_buffer_m=buffer_m if control else 0.0)
        pred = proba.argmax(axis=1)
        key = dict(group=group, split=split, model=model, seed=seed, block_m=block_m, buffer_m=buffer_m)
        row = dict(**key, n_fields=len(y), mean_train_size=n_train, seconds=time.time() - t0,
                   macro_f1=macro_f1(y, pred), accuracy=float((pred == y).mean()),
                   ece=expected_calibration_error(proba, y))
        part = pd.DataFrame(dict(**key, tile=fields.tile.to_numpy(), field_id=fields.field_id.to_numpy(),
                                 y=y, pred=pred, conf=proba.max(axis=1)))
        # Predictions are written before the run is marked done, so an interrupted
        # suite can never hold a "finished" run with no predictions behind it.
        oof = pd.concat([oof, part], ignore_index=True)
        oof.to_parquet(oof_path, index=False)
        runs = pd.concat([runs, pd.DataFrame([row])], ignore_index=True)
        runs.to_parquet(runs_path, index=False)
        print(f"[{i + 1}/{len(todo)}] {group:14s} {split:6s} {model:8s} seed {seed} block {block_m:.0f} "
              f"buffer {buffer_m:.0f}: macro-F1 {row['macro_f1']:.3f} ({row['seconds']:.0f}s)", flush=True)


def macro_f1(y: np.ndarray, pred: np.ndarray) -> float:
    """Macro-F1 over the classes present in `y`.

    A class with no true examples in the sample has no defined recall, so it is
    left out of the average instead of being scored as zero. On the full data
    every class is present; this matters for bootstrap resamples and subsets.
    """
    cm = np.bincount(y * N_CLASSES + pred, minlength=N_CLASSES * N_CLASSES).reshape(N_CLASSES, N_CLASSES)
    tp = np.diag(cm).astype(float)
    support, predicted = cm.sum(axis=1), cm.sum(axis=0)
    present = support > 0
    f1 = np.where(present & (tp > 0), 2 * tp / np.maximum(support + predicted, 1), 0.0)
    return float(f1[present].mean())


def summarise(fields_path: Path, features_path: Path, out_dir: Path, n_boot: int = 1000) -> dict:
    runs = pd.read_parquet(out_dir / "runs.parquet")
    oof = pd.read_parquet(out_dir / "oof.parquet").sort_values(KEY + ["tile", "field_id"])
    fields, _, _ = load(fields_path, features_path)
    y = fields.label.to_numpy() - 1
    blocks = block_ids(fields, 2000.0)
    n = len(y)

    preds, confs = {}, {}
    for key, g in oof.groupby(KEY, sort=False):
        assert len(g) == n and np.array_equal(g.y.to_numpy(), y), f"run {key} does not cover the labelled fields"
        preds[key], confs[key] = g.pred.to_numpy(), g.conf.to_numpy()

    def seeds_of(group, split, model, block_m=2000.0, buffer_m=0.0):
        return [preds[(group, split, model, s, block_m, float(buffer_m))] for s in SEEDS
                if (group, split, model, s, block_m, float(buffer_m)) in preds]

    # One shared set of block resamples for every statistic, so differences are paired.
    rng = np.random.default_rng(0)
    uniq = np.unique(blocks)
    members = [np.flatnonzero(blocks == u) for u in uniq]
    draws = [np.concatenate([members[j] for j in rng.integers(0, len(uniq), len(uniq))]) for _ in range(n_boot)]

    def boot(pred_list, mask=None):
        """Seed-averaged macro-F1 on each resample (optionally restricted to a field subset)."""
        out = np.empty(n_boot)
        for b, idx in enumerate(draws):
            if mask is not None:
                idx = idx[mask[idx]]
            out[b] = np.mean([macro_f1(y[idx], p[idx]) for p in pred_list])
        return out

    def point(pred_list, mask=None):
        m = slice(None) if mask is None else mask
        return [macro_f1(y[m], p[m]) for p in pred_list]

    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    def row_stats(group, split, model, block_m=2000.0, buffer_m=0.0):
        pl = seeds_of(group, split, model, block_m, buffer_m)
        if not pl:
            return None
        pts = point(pl)
        q = runs[(runs.group == group) & (runs.split == split) & (runs.model == model)
                 & (runs.block_m == block_m) & (runs.buffer_m == float(buffer_m))]
        return {"macro_f1": float(np.mean(pts)), "macro_f1_seed_sd": float(np.std(pts, ddof=1)) if len(pts) > 1 else 0.0,
                "macro_f1_ci95": ci(boot(pl)), "accuracy": float(q.accuracy.mean()), "ece": float(q.ece.mean()),
                "n_seeds": len(pl), "mean_train_size": float(q.mean_train_size.mean())}

    main = []
    for split in ("random", "block", "tile"):
        for model in ("majority", "rf", "tcn"):
            st = row_stats("main", split, model)
            if st:
                main.append({"split": split, "model": model, **st})

    def gap(model, other, mask=None):
        a, b = seeds_of("main", "random", model), seeds_of("main", other, model)
        if not a or not b:
            return None
        d = boot(a, mask) - boot(b, mask)
        return {"model": model, "comparison": f"random - {other}",
                "macro_f1_gap": float(np.mean(point(a, mask)) - np.mean(point(b, mask))),
                "ci95": ci(d), "share_of_resamples_above_zero": float((d > 0).mean()), "n_seeds": min(len(a), len(b))}

    gaps = [g for model in ("rf", "tcn") for other in ("block", "tile") if (g := gap(model, other))]

    # H3: does the neural model lose more than the forest? Difference of the two gaps, paired.
    h3 = None
    if all(seeds_of("main", s, m) for s in ("random", "tile") for m in ("rf", "tcn")):
        d = (boot(seeds_of("main", "random", "tcn")) - boot(seeds_of("main", "tile", "tcn"))) - \
            (boot(seeds_of("main", "random", "rf")) - boot(seeds_of("main", "tile", "rf")))
        h3 = {"tcn_gap_minus_rf_gap": float(np.mean(d)), "ci95": ci(d)}

    buffer = []
    for model in ("rf", "tcn"):
        for buffer_m in BUFFERS:
            st = row_stats("main" if buffer_m == 0 else "buffer", "block", model, buffer_m=buffer_m)
            if st:
                buffer.append({"model": model, "buffer_m": buffer_m, **st})
    # Size-matched control and the paired difference "control - buffered": what the
    # gap itself costs, once the smaller training set is accounted for.
    control = []
    for model in ("rf", "tcn"):
        for buffer_m in BUFFERS[1:]:
            st = row_stats("buffer_control", "block", model, buffer_m=buffer_m)
            if not st:
                continue
            c, b = seeds_of("buffer_control", "block", model, buffer_m=buffer_m), seeds_of("buffer", "block", model, buffer_m=buffer_m)
            d = boot(c) - boot(b)
            control.append({"model": model, "buffer_m": buffer_m, **st,
                            "cost_of_gap": float(np.mean(point(c)) - np.mean(point(b))), "cost_of_gap_ci95": ci(d)})

    block_size = []
    for block_m in (1000.0, 2000.0, 5000.0):
        st = row_stats("main" if block_m == 2000.0 else "block_size", "block", "rf", block_m=block_m)
        if st:
            block_size.append({"block_m": block_m, **st})

    # Field-size sensitivity: the SAME predictions, scored on the larger fields only.
    field_size = []
    for min_px in (1, 4, 9):
        mask = fields.n_pixels.to_numpy() >= min_px
        g = gap("rf", "block", mask)
        if g:
            field_size.append({"min_pixels": min_px, "n_fields": int(mask.sum()),
                               "random": float(np.mean(point(seeds_of("main", "random", "rf"), mask))),
                               "block": float(np.mean(point(seeds_of("main", "block", "rf"), mask))),
                               "gap": g["macro_f1_gap"], "gap_ci95": g["ci95"]})

    # Under the tile split only four models exist, one per held-out tile: show each.
    per_tile = []
    for t in sorted(fields.tile.unique()):
        mask = (fields.tile == t).to_numpy()
        entry = {"tile": t, "n_fields": int(mask.sum())}
        for split in ("random", "tile"):
            for model in ("rf", "tcn"):
                pl = seeds_of("main", split, model)
                if pl:
                    entry[f"{split}_{model}"] = float(np.mean(point(pl, mask)))
        per_tile.append(entry)

    per_class = []
    for model in ("rf", "tcn"):
        for split in ("random", "block", "tile"):
            pl = seeds_of("main", split, model)
            for c in range(N_CLASSES):
                m = y == c
                if pl and m.any():
                    per_class.append({"model": model, "split": split, "class": CLASSES[c + 1], "n_fields": int(m.sum()),
                                      "recall": float(np.mean([(p[m] == c).mean() for p in pl]))})

    ece = []
    for model in ("rf", "tcn"):
        for split in ("random", "block", "tile"):
            q = runs[(runs.group == "main") & (runs.split == split) & (runs.model == model)]
            if len(q):
                ece.append({"model": model, "split": split, "ece": float(q.ece.mean()), "ece_seed_sd": float(q.ece.std(ddof=1))})

    tiles = sorted(fields.tile.unique())
    summary = {
        "main": main, "gaps": gaps, "h3": h3, "buffer": buffer, "buffer_control": control,
        "block_size": block_size, "field_size": field_size, "per_tile": per_tile, "per_class": per_class, "ece": ece,
        "n_labelled_fields": int(n), "n_blocks_2km": int(len(uniq)), "n_bootstrap": n_boot,
        "class_counts": {CLASSES[c + 1]: int((y == c).sum()) for c in range(N_CLASSES)},
        "class_counts_by_tile": {t: {CLASSES[c + 1]: int(((y == c) & (fields.tile == t).to_numpy()).sum()) for c in range(N_CLASSES)} for t in tiles},
        "fields_per_block": {"median": float(np.median([len(m) for m in members])), "max": int(max(len(m) for m in members)),
                             "single_field_blocks": int(sum(len(m) == 1 for m in members))},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
