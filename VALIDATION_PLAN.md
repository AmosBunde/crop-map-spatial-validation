# Validation plan

Written on 20 November 2026, before any experiment on the full dataset. Changes after
results are seen will be recorded at the bottom with a date and a reason.

## Question

By how much does a random hold-out overstate the accuracy of a field-level crop-type
classifier, compared with holding out whole areas, for smallholder fields in western Kenya?

## Hypotheses

- **H1.** Macro-F1 under the `random` split is higher than under the `block` split,
  and higher again than under the `tile` split.
- **H2.** The gap grows with the buffer between training and test fields, then levels
  off at the distance where spatial autocorrelation in the features dies out.
- **H3.** The neural model loses more than the random forest when moving from
  `random` to `tile`, because it has more capacity to fit place-specific signal.
- **H4.** Confidence is less trustworthy off-site: expected calibration error is
  higher under `block` and `tile` than under `random`.

Any of these may turn out false. A null result on H1 would itself be worth reporting.

## Unit of analysis

One labelled field. A field's label is the majority label of its pixels; features are
cloud-masked band means per date. Pixels are never split across train and test.

## Splits

| Name | Construction | Folds |
|------|--------------|-------|
| `random` | Stratified by class over fields | 5 |
| `block` | 2 km square blocks on the map grid; whole blocks held out, stratified by class | 5 |
| `block` + buffer | As above, dropping training fields within 250, 500, 1000, 2000 m of any test field | 5 |
| `tile` | Leave one of the four tiles out | 4 |

Block size sensitivity: 1, 2 and 5 km. The block size for the headline comparison is
2 km, fixed here, and will not be chosen after seeing results.

## Models

| Name | Input | Why |
|------|-------|-----|
| Majority class | none | Floor |
| Random forest | flattened series, 16 features x 13 dates | Strong, standard baseline for this task |
| Temporal CNN | series as 16 channels over 13 dates | A deep model small enough to train per fold on CPU |

Hyperparameters are fixed in `models.py` and are not tuned on test folds. If tuning is
added, it will be nested inside the training folds using the same split scheme.

## Metrics

- **Primary: macro-F1** over the classes present, pooled across out-of-fold predictions.
  Classes are heavily imbalanced, so accuracy alone would mostly measure maize.
- Secondary: balanced accuracy, overall accuracy, per-class recall.
- Calibration: expected calibration error with 10 bins.

## Uncertainty

- 95% intervals from 1,000 bootstrap resamples of **spatial blocks**, not fields.
  Resampling fields would treat neighbouring errors as independent and give intervals
  that are too narrow.
- Each configuration is repeated over 5 seeds; seed-to-seed spread is reported separately
  from the bootstrap interval.
- The headline number is the difference in macro-F1 between `random` and `block`, with
  an interval from a paired block bootstrap.

## Checks on the data before modelling

1. Reproduce the coordinate recovery (`cropval georeference`) on the full download.
2. Plot fields on the mosaic and confirm polygons sit on visible field patterns.
3. Class counts per tile and per block; note any class absent from a tile.
4. Cloud fraction per date and tile; decide whether any date is unusable.
5. Field-size distribution; repeat the headline comparison for fields of at least 4 and
   at least 9 pixels to see whether tiny fields drive the result.
6. Semivariogram of NDVI at peak season, to give H2 an independent estimate of the
   autocorrelation range.

## Threats to validity

- **Label collection was clustered.** Enumerators walked routes, so labelled fields are
  bunched. This is the mechanism being studied, and it also means blocks differ in size.
- **Class mix differs by area.** A drop under `tile` may reflect a change in class
  balance as well as a change in appearance. Per-class recall and a class-balance table
  per fold will be reported to separate the two.
- **One season, one region.** The estimate is for Busia in 2019.
- **Mixed pixels.** Many fields are smaller than a few pixels.
- **Georeferencing is recovered, not original**, with an uncertainty of a few metres.
  This is far below the 2 km block size and cannot move a field between blocks in
  any meaningful number.

## Deliverables

1. A results table: split x model, macro-F1 with interval.
2. One figure: macro-F1 against buffer distance, both models.
3. A map of out-of-fold errors under `tile`.
4. A short write-up stating what a practitioner should report instead of a random-split score.

## Amendments

All made on 20 November 2026, after an independent code review and before the results
in the README were produced. None was made in response to a result, except where stated.

1. **Macro-F1 in resamples.** A bootstrap draw that contains no field of a class leaves
   that class out of the average instead of scoring it zero, matching "the classes
   present" above. On the full data all seven classes are present.
2. **One shared bootstrap.** All statistics use the same 1,000 block resamples, and each
   draw scores the mean over five seeds. The first implementation pooled separate
   per-seed bootstraps, which gave intervals for a single-seed run, not for the
   seed mean that is reported.
3. **Point estimates** of gaps are the observed difference of seed means, not a bootstrap mean.
4. **Imputation inside folds.** Values a field never observed are filled from training
   rows in each fold. The first implementation used a median over all fields. Cloud gaps
   are interpolated over real acquisition dates, within each field.
5. **Field-size sensitivity** re-scores the same predictions on larger fields with a fixed
   class set. The first design retrained on fewer fields and let the class set shrink,
   which confounded three things.
6. **Size-matched control for the buffer experiment.** Added because a buffer also shrinks
   the training set. Each buffered run has a control with the same number of training
   fields and no gap. The control for the temporal CNN was added after the forest's
   control had been seen, to make the same comparison for both models.
7. **Tile design.** Per-tile scores are reported, and the README states that the block
   bootstrap interval is conditional on the four fitted models.
8. **Headline model.** The plan did not say which model the headline would quote. It is
   the one with the higher random-split score, the temporal CNN, with the forest shown
   beside it everywhere.
9. **Not done.** The per-fold class-balance table is replaced by class counts per tile.
   Hyperparameters were not tuned. Check 2 (overlaying polygons on imagery) was done by
   eye on the exported map only.
