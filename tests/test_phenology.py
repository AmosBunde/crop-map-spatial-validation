import numpy as np

from cropval.phenology import INDICES, PARAMETERS, bells, candidates, fit_curve, phenology_parameters

DAYS = np.array([0, 25, 30, 35, 45, 60, 70, 80, 95, 105, 110, 120, 150], dtype=float)


def curve(base, amp, peak, rise, fall):
    return base + amp * bells(DAYS, np.array([[peak, rise, fall]]))[0]


def test_recovers_a_curve_that_lies_on_the_grid():
    y = curve(0.2, 0.5, 60.0, 25.0, 40.0)
    base, amp, peak, rise, fall, rmse = fit_curve(y[None, :], DAYS)[0]
    assert (peak, rise, fall) == (60.0, 25.0, 40.0)
    assert abs(base - 0.2) < 1e-9 and abs(amp - 0.5) < 1e-9 and rmse < 1e-6


def test_fits_many_fields_at_once_and_tolerates_noise():
    rng = np.random.default_rng(0)
    peaks = rng.choice([40.0, 70.0, 100.0], 200)
    y = np.stack([curve(0.3, 0.4, p, 25.0, 25.0) for p in peaks]) + rng.normal(0, 0.01, (200, len(DAYS)))
    out = fit_curve(y, DAYS)
    assert np.mean(np.abs(out[:, 2] - peaks) <= 10.0) > 0.95
    assert np.all(out[:, 1] > 0.3) and np.all(out[:, 5] < 0.03)


def test_flat_series_has_zero_amplitude():
    out = fit_curve(np.full((1, len(DAYS)), 0.37), DAYS)[0]
    assert out[1] == 0.0 and abs(out[0] - 0.37) < 1e-12 and out[5] < 1e-9


def test_amplitude_is_never_negative():
    out = fit_curve(-curve(0.0, 0.5, 60.0, 25.0, 25.0)[None, :], DAYS)[0]
    assert out[1] >= 0.0


def test_offset_moves_the_base_and_nothing_else():
    y = curve(0.2, 0.5, 80.0, 40.0, 15.0)
    a, b = fit_curve(y[None, :], DAYS)[0], fit_curve(y[None, :] + 0.1, DAYS)[0]
    assert abs((b[0] - a[0]) - 0.1) < 1e-9 and np.allclose(a[1:], b[1:], atol=1e-9)


def test_missing_values_give_missing_parameters():
    y = np.stack([curve(0.2, 0.5, 60.0, 25.0, 25.0)] * 2)
    y[1, 3] = np.nan
    out = fit_curve(y, DAYS)
    assert np.isfinite(out[0]).all() and np.isnan(out[1]).all()


def test_each_field_is_fitted_independently():
    rng = np.random.default_rng(1)
    y = rng.uniform(0, 1, (30, len(DAYS)))
    assert np.array_equal(fit_curve(y, DAYS)[7], fit_curve(y[7:8], DAYS)[0])


def test_parameter_cube_has_one_date_and_eighteen_features():
    rng = np.random.default_rng(2)
    columns = ["B02", "NDVI", "NDRE", "NDWI", "cloud_frac"]
    x = rng.uniform(0, 1, (5, len(DAYS), len(columns))).astype(np.float32)
    out = phenology_parameters(x, DAYS, columns)
    assert out.shape == (5, 1, len(INDICES) * len(PARAMETERS))
    assert np.allclose(out[:, 0, :6], fit_curve(x[:, :, 1], DAYS), atol=1e-5)


def test_grid_is_fixed():
    assert candidates().shape == (22 * 5 * 5, 3)
