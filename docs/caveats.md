# Caveats

Things that affect how the numbers should be read. Each is reproducible from this repository.

## The channel Reynolds number is not independent of grain size

`scripts/03_water_metrics.py` estimates Manning's n from D84 by the Strickler relation
(`n = 0.0342 D84^(1/6)`), then bankfull velocity from Manning's equation, then
`Re = U H / ν`. So `U ∝ D84^(-1/6)` and `Re ∝ D84^(-1/6)`. The shear Reynolds number
`Re* = u* D84 / ν` contains D84 directly. Both axes of Fig. 3 therefore carry grain size,
and Re should not be described as an independent measure of flow intensity.
`tests/test_pipeline.py::test_channel_reynolds_number_depends_on_grain_size` pins the
exponent.

Related: partial Spearman correlations between the fitted parameters and Re*, controlling
for D84, collapse for every function — GIG η from +0.85 to +0.18, β from −0.77 to −0.20,
and Lognormal and Weibull to near zero or the opposite sign.

## Which percentiles enter each variable

Bedload transport, critical shear stress and the 100-year flood stage are driven by D84
alone. The Fredle Index uses four percentiles:
`FI = sqrt(D16 · D84) / sqrt(D75 / D25)`. It is not a D84-only quantity, though it is a
percentile-based approximation of the Lotspeich & Everest form, which computes the geometric
mean from the full sieve distribution rather than from `sqrt(D16 · D84)`.

Critical shear stress is linear in D84 (`τc = θc (ρs − ρ) g D84`), so its relative error is
*identically* D84's. Its row in Fig. 5 is a consistency check, not independent evidence.

## Bedload sample selection

Bedload is defined only where the Shields stress exceeds the critical value. Of the 19,763
samples with complete percentile estimates:

| Step | Rule | n | Sites |
|---|---|---|---|
| start | complete D values | 19,763 | 2,114 |
| threshold of motion | τ\* > τ\*c = 0.047 for all four D84 estimates | 14,979 | 1,530 |
| zero reference | 412 samples whose reference q_b rounds to zero at 1e-6, making the relative error undefined | 14,567 | 1,466 |

So 4,784 samples are excluded because the bed is below the threshold of motion, and 412 more
because dividing by the reference is undefined.

## Percentile estimates can extrapolate past the coarsest sieve

The coarsest grade in the dataset is 256 mm (128 mm among the principal grades), but a fitted
distribution can place D84 well beyond it. The archived Lognormal fits reach 1,969 mm for one
sample. Such samples are rare but dominate Lognormal's D84 RMSRE, so the D84 comparison is
partly a statement about extrapolation behaviour rather than about fit quality inside the
measured range. `tests/test_pipeline.py::test_lognormal_d84_extrapolates_far_beyond_the_coarsest_sieve`
records this.

## Not every candidate is a valid distribution at its fitted parameters

Fitting is unconstrained least squares on the cumulative curve, which does not enforce
monotonicity or the 0–100% bounds. Run `python scripts/07_cdf_validity.py`: inside each
sample's own measured sieve range, eight of the 25 candidates are valid cumulative
distributions throughout — GLH_p05, GLH_p1, Tanh, Lognormal, GIG_2p, Weibull, GLH, GIG_3p —
and the rest are not. pPL_Exp is non-monotone for 66% of samples and reaches 204%; PL_Weib is
non-monotone for 38%. Every claim in the paper rests on one of the eight well-behaved
functions, but "25 candidate distributions" is doing some work in that sentence.

## Two fitting layers exist, and the paper quotes both

Nine functions are fitted twice: once from a single default initial guess ("base"), and once
over a grid of initial guesses, keeping the lowest-BIC result ("grid-selected"). Six of those
nine have both layers preserved in the legacy tree, and the deposit ships both --
`02_fits/fits_F*.csv` are the grid-selected finals, `02_fits/base_fits/base_F*.csv` the base
fits.

This matters because **the manuscript's main text quotes the base fits while Table S2 quotes
the grid-selected ones**:

| Function | Main text | Table S2 (grid-selected) | Base fit |
|---|---|---|---|
| GIG_3p | 16.7 | 16.676 | 16.746 |
| GIG_2p | 19.5 | 19.472 | 19.472 |
| Logn_PL | 19.8 | 19.570 | **19.810** |
| Lognormal | 21.3 | 20.767 | **21.334** |
| GLH | 21.9 | 21.583 | **21.853** |
| Weibull | 24.5 | 24.135 | **24.499** |

GIG_2p is the one function whose two layers have the same median, so the main text's
comparison puts GIG's grid-selected number against its competitors' base-fit numbers. The gap
to Logn_PL reads as 0.34 BIC units that way; on a consistent set it is 0.10, which the paper's
own Kass & Raftery criterion calls negligible.

The headline claims survive either consistent choice — GIG_2p is first among two-parameter
functions and second overall under both — but the main text should be brought onto the
grid-selected values, which is what Table S2, Table S3, Fig. 4, the percentiles and every
downstream variable use.

Everything in this repository uses the grid-selected layer.

## Grid search is applied to nine of the 25 functions

Nine functions are optimised over a grid of initial parameter values; the other sixteen get a
single default guess with one retry. This is defensible — those nine are the initialisation-
sensitive ones — and can be demonstrated:

* the nine two-parameter functions outside the grid set are insensitive. Under a 50-point
  grid their median BIC is unchanged to three decimals and fewer than 6% of samples improve
  at all.
* the seven three-parameter functions outside it *are* sensitive. Under a 133-point grid,
  LSLaplace improves from a median BIC of 37.5 to 25.5, Logn_Weib from 32.9 to 27.8.

Rerunning those seven with a grid does not change the paper's conclusions: on a 120-sample
probe, GIG_3p's count of outright wins was unchanged (33 of 120) and GIG_2p's moved by one.
The mid-table ranking of Table S2 would change.

## Percentile solver: analytic inverse versus the archived grid scan

The archived percentiles (`03_percentiles/dvalues_archived.csv` in the deposit) were produced
by a grid scan of the fitted CDF; `scripts/02_compute_dvalues.py` uses an analytic inverse
with a bracketed-root fallback. They agree to about 1e-4 relative, except where the original
scan hit its upper bound. `scripts/03_water_metrics.py` defaults to the archived table so
that the published Fig. 5 values reproduce exactly; pass
`--dvalues outputs/dvalues/dvalues_recomputed.csv` to use the analytic inverse instead, which
shifts the Fredle Index reduction against Lognormal from 29.1% to 30.3%.
`outputs/dvalues/dvalues_archived_vs_recomputed.csv` quantifies the difference per percentile.

## Sample identifiers

`sample_ID` is meaningful only within a given release of the data deposit. Different
retrieval runs of the source records have assigned different identifiers historically, so do
not join across versions on it.
