# Figure and table map

Every display item in the paper, the script that draws it, and what it reads.

| Item | Script | Reads |
|---|---|---|
| Fig. 1 | `scripts/fig_01.py` (+ `_figure1_common.py`) | `outputs/tables/analysis_frame.csv`, `data/fitted_functions/`, `data/reference/sieve_sizes_all43.csv` |
| Fig. 2 | `scripts/fig_02.py` | `outputs/tables/analysis_frame.csv`, `data/reference/wbd_hu2.gpkg` |
| Fig. 3 | `scripts/fig_03.py` | `outputs/water/sample_water_metrics.csv`, `data/best_fit_by_sample.csv`, `data/fitted_functions/` |
| Fig. 4 | `scripts/fig_04_S04.py` | `outputs/tables/bic_matrix.tsv` |
| Fig. 5 | `scripts/fig_05.py` | `outputs/water/sample_water_metrics.csv` |
| Fig. S1 | `scripts/fig_S01.py` | `data/reference/google_scholar_decade_counts.csv` |
| Fig. S2 | `scripts/fig_S02.py` | `outputs/tables/analysis_frame.csv`, `data/reference/wbd_hu2.gpkg` |
| Fig. S3 | `scripts/fig_S03.py` | `outputs/tables/bic_matrix.tsv` |
| Fig. S4 | `scripts/fig_04_S04.py --input outputs/tables/rmse_matrix.tsv --metric RMSE` | `outputs/tables/rmse_matrix.tsv` |
| Fig. S5 | `scripts/fig_S05.py` | `outputs/rhine/`, `outputs/fit_rhine/fitted_functions/` |
| Table S1 | — | `02_fits/function_catalog.csv` in the deposit; forms in `src/psd_gig/function_library.py` |
| Table S2 | `scripts/04_bic_tables.py` | `data/fitted_functions/` |
| Table S3 | `scripts/04_bic_tables.py` | `data/fitted_functions/` |
| Table S4 | `scripts/04_bic_tables.py` | `data/fitted_functions/` |
| Table S5 | `scripts/05_significance.py` | `outputs/water/sample_water_metrics.csv` |

## Main-text numbers

| Claim | Source |
|---|---|
| best-fit shares, skewness and D50 classes | `data/usgs_sample_statistics.csv` via `audit_numbers.py --section best-fit` |
| median BIC per function | `outputs/tables/table_S2_median_metrics.csv` |
| lowest-BIC shares (Fig. 4c) | `outputs/tables/figure4_delta_bic_summary.csv` |
| RMSRE matrix and error reductions | `outputs/tables/figure5_rmsre_matrix.csv` |
| η and β correlations with Re\*, Re | `audit_numbers.py --section fig3` |
| flood-stage MAE and RMSE | `audit_numbers.py --section flood` |
| Lower Rhine median BIC and best-fit shares | `outputs/rhine/rhine_function_summary.csv` |

`python scripts/audit_numbers.py` checks all of these at once and reports any that disagree
with the manuscript.
