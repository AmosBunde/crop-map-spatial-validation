# Follow-up plan: does a phenology-structured model degrade less with distance?

Written on 20 November 2026, before the model described here was run on any labelled
field. It was not lodged with an outside registry. Changes after results are seen will be
listed at the bottom with a reason.

## Where this comes from

A re-analysis of this study and two others placed each test item at its distance from the
training data. Predictors fitted directly to their target got worse with distance. Two
outputs that are constrained by a model of the problem (depth in a radiance field, a
runtime simulator built from operator costs) did not. That was noticed after the fact.
This plan tests it in a domain where it has not been looked at: crop-type mapping.

## The prediction

On the same test fields, a classifier that sees each field only through the parameters of
a crop growth curve loses less accuracy with distance from the training fields than a
classifier that sees the raw time series.

## The models compared

- `rf`: the existing random forest on the raw series (13 dates x 16 features). Already run.
- `tcn`: the existing temporal convolutional network. Already run.
- `pheno`: new. For each field, and for each of NDVI, NDRE and NDWI, fit
  `index(t) = base + amplitude * bell(t; peak, rise, fall)`, where `bell` is a Gaussian
  with separate widths before and after the peak and `amplitude >= 0`. The fit is a grid
  search over `(peak, rise, fall)` with `base` and `amplitude` solved in closed form. The
  field is then described by 18 numbers (base, amplitude, peak day, rise width, fall
  width and fit error, for three indices). The same random forest as `rf`, with the same
  settings, is trained on those 18 numbers.

`pheno` and `rf` differ only in what the forest is given, so the comparison isolates the
representation. The fit uses a field's own series and nothing else, so no information
crosses between training and test fields.

## Runs

`pheno` on the `random` and `block` (2 km) designs, seeds 0 to 4, five folds each: ten
runs. These are the two designs that train on the same number of fields. No setting of
`pheno` will be tuned on labelled data; the grid is fixed in the code before the first run.

## Analysis

Each test field's distance to its nearest training field is computed from the rebuilt
folds. Fields are grouped at 0, 50, 100, 200, 400 and 800 metres (six groups), the groups
already used for `tcn`. Error is `1 - macro-F1` in each group. The quantity of interest is

    R(model) = error in the farthest group / error in the nearest group.

Intervals come from a bootstrap over 2 km blocks, with the models kept paired inside each
resample (2,000 resamples).

## What counts as support

- **P1 (the test).** `R(pheno) < R(rf)`, and the 95% interval for `R(pheno) - R(rf)`
  excludes zero.
- **P2 (the same against the network).** `R(pheno) < R(tcn)`, interval excluding zero.
- **G (guard against a trivial pass).** A model that has learned nothing has a flat curve.
  The majority-class baseline scores macro-F1 0.088. P1 and P2 only count if `pheno`'s
  macro-F1 in the nearest group is at least halfway between 0.088 and `rf`'s macro-F1 in
  the nearest group.
- **P3 (strong form).** In the farthest group, `pheno`'s macro-F1 is not below `rf`'s:
  the 95% interval for the difference includes zero or lies above it.

If G fails, the result is "untestable with this model", not support. If P1 fails, the
prediction failed in this domain and will be reported that way.

## What I expect, honestly

I think P1 is more likely to fail than pass. Thirteen dates from June to November cover
only part of the growing calendar, smallholder fields are intercropped, and a bell curve
may throw away the signal the forest uses. I am running it because the claim is on my
profile as a prediction that can fail, and this is the cheapest place to let it.

## Changes after results

None yet.
