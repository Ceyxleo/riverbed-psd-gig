"""Unit checks for the fitting library and the derived metrics.

These run without the data deposit except where marked; anything needing
`data/` is skipped if the inputs have not been fetched.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import geninvgauss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from psd_gig import function_library as funcs  # noqa: E402
from psd_gig.fit_specs import FUNCTION_SPECS  # noqa: E402

DATA = ROOT / "data"
GRID = np.logspace(np.log2(1 / 64), np.log2(256), 200, base=2)


# --------------------------------------------------------------- the 25 CDFs

@pytest.mark.parametrize("spec", FUNCTION_SPECS, ids=lambda s: s.id)
def test_cdf_is_finite_over_the_grain_size_range(spec):
    """Every candidate must evaluate without blowing up across the sieve range.

    Monotonicity is a property of the *fitted* parameters, not of arbitrary ones:
    several of these forms are only valid cumulative distributions on part of
    their parameter space. That is checked separately in
    ``test_headline_fits_are_valid_distributions``.
    """
    guess = spec.base_initial_guess or tuple([1.0] * spec.n_params)
    x = np.log2(GRID) if spec.base_x_transform == "log2" else GRID
    with np.errstate(all="ignore"):
        y = np.asarray(funcs.__dict__[spec.func.__name__](x, *guess), dtype=float)
    assert np.isfinite(y).sum() > 20, f"{spec.id} produced almost no finite values"


@pytest.fixture(scope="module")
def measured_ranges():
    """Per sample, the smallest and largest sieve size that actually carries data."""
    path = DATA / "usgs_psd_samples_qc.csv"
    if not path.exists():
        pytest.skip("run scripts/00_fetch_inputs.py first")
    samples = pd.read_csv(path, low_memory=False)
    sizes = pd.read_csv(DATA / "reference" / "sieve_sizes_all43.csv")
    lookup = {column: float(sizes[column].iloc[0]) for column in sizes.columns}
    columns = [c for c in samples.columns if c in lookup]
    values = samples[columns].to_numpy(float)
    grid = np.array([lookup[c] for c in columns])
    order = np.argsort(grid)
    values, grid = values[:, order], grid[order]
    present = np.isfinite(values)
    low = np.where(present.any(axis=1), grid[present.argmax(axis=1)], np.nan)
    high = np.where(present.any(axis=1),
                    grid[present.shape[1] - 1 - present[:, ::-1].argmax(axis=1)], np.nan)
    return pd.DataFrame({"sample_ID": samples["sample_ID"].astype(int),
                         "d_min": low, "d_max": high}).set_index("sample_ID")


# Functions whose fits are valid cumulative distributions everywhere in the
# measured sieve range. Every claim in the paper rests on one of these. Fitting
# is unconstrained least squares on the cumulative curve, which does not enforce
# monotonicity or the 0-100% bounds, so not every candidate qualifies.
WELL_BEHAVED = ["GLH_p05", "GLH_p1", "Tanh", "Lognormal", "GIG_2p", "Weibull",
                "GLH", "GIG_3p"]


@pytest.mark.skipif(not (DATA / "fitted_functions").exists(),
                    reason="run scripts/00_fetch_inputs.py first")
@pytest.mark.parametrize("label", WELL_BEHAVED)
def test_headline_fits_are_valid_distributions(label, measured_ranges):
    """Every function the paper's conclusions depend on is a proper CDF when fitted."""
    spec = next(s for s in FUNCTION_SPECS if s.label == label)
    path = DATA / "fitted_functions" / f"fits_F{spec.number:02d}_{spec.label}.csv"
    fit = pd.read_csv(path).dropna(subset=["fitted_A"])
    fit["sample_ID"] = fit["sample_ID"].astype(int)
    sample = fit.sample(n=min(50, len(fit)), random_state=0)
    columns = ["fitted_A", "fitted_B", "fitted_C"][: spec.n_params]

    checked = 0
    for _, row in sample.iterrows():
        if row["sample_ID"] not in measured_ranges.index:
            continue
        bounds = measured_ranges.loc[row["sample_ID"]]
        parameters = [row[c] for c in columns]
        if not np.isfinite([bounds.d_min, bounds.d_max]).all() or not all(
                np.isfinite(parameters)):
            continue
        grid = np.logspace(np.log2(bounds.d_min), np.log2(bounds.d_max), 80, base=2)
        x = np.log2(grid) if spec.base_x_transform == "log2" else grid
        with np.errstate(all="ignore"):
            y = np.asarray(spec.func(x, *parameters), dtype=float)
        y = y[np.isfinite(y)]
        if len(y) < 20:
            continue
        checked += 1
        assert y.min() >= -0.5, f"{spec.id} sample {row['sample_ID']} drops below 0%"
        assert y.max() <= 100.5, f"{spec.id} sample {row['sample_ID']} exceeds 100%"
        assert np.all(np.diff(y) >= -0.5), \
            f"{spec.id} sample {row['sample_ID']} is not monotone in its measured range"
    assert checked >= 20, f"{spec.id}: only {checked} usable parameter sets"


@pytest.mark.skipif(not (DATA / "fitted_functions").exists(),
                    reason="run scripts/00_fetch_inputs.py first")
def test_lognormal_d84_extrapolates_far_beyond_the_coarsest_sieve():
    """The extrapolation caveat behind the D84 comparison in Fig. 5.

    The coarsest sieve grade in the dataset is 256 mm and the coarsest principal
    grade is 128 mm, yet the Lognormal fits imply D84 values far above both for a
    handful of samples. Those samples dominate Lognormal's D84 RMSRE.
    """
    percentiles = pd.read_csv(DATA / "dvalues.csv", dtype={"site_no": str})
    largest_grade = 256.0
    beyond = percentiles["D84_Lognormal"] > largest_grade
    assert beyond.any(), "expected at least one Lognormal D84 beyond the coarsest sieve"
    assert percentiles["D84_Lognormal"].max() > 1000, \
        "the known worst case is about 1,969 mm"
    assert beyond.sum() < 0.001 * len(percentiles), \
        "extrapolation beyond the sieve range should stay rare"


def test_function_count_and_families():
    assert len(FUNCTION_SPECS) == 25
    assert sum(spec.n_params == 2 for spec in FUNCTION_SPECS) == 15
    assert sum(spec.n_params == 3 for spec in FUNCTION_SPECS) == 10


def test_grid_searched_functions_are_the_documented_nine():
    grid_searched = {spec.label for spec in FUNCTION_SPECS if spec.grid}
    assert grid_searched == {"GLH_p05", "GLH_p1", "Lognormal", "NIG", "GIG_2p",
                             "Weibull", "GLH", "Logn_PL", "GIG_3p"}


# ------------------------------------------------------------------- metrics

def test_bic_matches_the_published_definition():
    """BIC = N ln(MSE) + p ln(N); this is what back-solving the archive assumes."""
    from psd_gig.fitting import _metric_row
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    row = _metric_row(lambda v, a: v * a, x, y, np.array([10.0]), 1)
    assert row["RMSE"] == pytest.approx(0.0, abs=1e-12)
    assert row["R^2"] == pytest.approx(1.0)

    predicted = y + np.array([1.0, -1.0, 1.0, -1.0, 1.0])
    row = _metric_row(lambda v, a: predicted, x, y, np.array([1.0]), 2)
    mse = 1.0
    assert row["BIC"] == pytest.approx(5 * np.log(mse) + 2 * np.log(5))


def test_log2_density_jacobian_integrates_to_one():
    """Fig. 1 plots f(D) * D * ln2 against log2 D; that must still be a density."""
    eta, beta = -0.4, 1.6
    log2_grid = np.linspace(-12, 12, 200_001)
    x = 2.0 ** log2_grid
    density = geninvgauss.pdf(x, eta, beta) * x * np.log(2.0)
    assert np.trapezoid(density, log2_grid) == pytest.approx(1.0, abs=1e-4)


def test_fredle_index_uses_four_percentiles():
    """FI = sqrt(D16 D84) / sqrt(D75/D25) -- four percentiles, not D84 alone."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("wm", ROOT / "scripts" / "02_water_metrics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    frame = pd.DataFrame({
        "site_no": ["1"], "SLOPE": [0.001], "BANKFULL_XSEC_AREA": [50.0],
        "BANKFULL_WIDTH": [25.0], "BANKFULL_DEPTH": [2.0], "Q100": [500.0],
    })
    for variant in module.VARIANTS:
        frame[f"D16_{variant}"] = 1.0
        frame[f"D25_{variant}"] = 2.0
        frame[f"D50_{variant}"] = 4.0
        frame[f"D75_{variant}"] = 8.0
        frame[f"D84_{variant}"] = 16.0
    out = module.compute(frame)
    expected = np.sqrt(1.0 * 16.0) / np.sqrt(8.0 / 2.0)
    assert out["fredle_index_GIG_2p"].iloc[0] == pytest.approx(expected)

    # Critical shear stress is linear in D84, so its relative error equals D84's.
    tau = out["critical_shear_stress_GIG_2p_Pa"].iloc[0]
    assert tau == pytest.approx(0.045 * (2650 - 1000) * 9.81 * 0.016)


def test_channel_reynolds_number_depends_on_grain_size():
    """Re is not independent of grain size: U scales as D84^(-1/6)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("wm", ROOT / "scripts" / "02_water_metrics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def frame_with(d84):
        frame = pd.DataFrame({
            "site_no": ["1"], "SLOPE": [0.001], "BANKFULL_XSEC_AREA": [50.0],
            "BANKFULL_WIDTH": [25.0], "BANKFULL_DEPTH": [2.0], "Q100": [500.0],
        })
        for variant in module.VARIANTS:
            for percentile, value in (("D16", 1.0), ("D25", 2.0), ("D50", 4.0),
                                      ("D75", 8.0), ("D84", d84)):
                frame[f"{percentile}_{variant}"] = value
        return module.compute(frame)

    coarse = frame_with(64.0)["Re_GIG_2p"].iloc[0]
    fine = frame_with(1.0)["Re_GIG_2p"].iloc[0]
    assert fine / coarse == pytest.approx(64.0 ** (1 / 6), rel=1e-9)


# ------------------------------------------------------- end-to-end smoke run

@pytest.mark.skipif(not (DATA / "usgs_psd_samples_qc.csv").exists(),
                    reason="run scripts/00_fetch_inputs.py first")
def test_smoke_pipeline_runs():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "01_fit_functions.py"),
         "--config", "configs/fit_smoke.yaml"],
        cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr[-2000:]
    written = sorted((ROOT / "outputs" / "fit_smoke" / "base_fits").glob("*.csv"))
    assert len(written) == 25


@pytest.mark.skipif(not (DATA / "usgs_psd_samples_qc.csv").exists(),
                    reason="run scripts/00_fetch_inputs.py first")
def test_fitting_uses_every_sieve_grade():
    """The archived fits used all 43 grades, not the 12 principal codes."""
    from psd_gig.config import load_config
    from psd_gig.fit_data import load_input_data

    config = load_config(ROOT / "configs" / "fit_usgs.yaml")
    _, size_columns = load_input_data(config)
    assert len(size_columns) == 43
