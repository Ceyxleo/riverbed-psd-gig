# What a full grid search would cost

All figures measured on this machine, single core, `maxfev = 20000`, scaled to the full
19,765 USGS samples. The nine already-grid-searched functions are timed either from the
completed `outputs/fit_workflow_fast` run or by re-timing their own grid here; the sixteen
others by actually running the grid on 200 random samples.

## Already grid-searched — no need to re-run

| Function | Initial guesses | Time | Source |
|---|---|---|---|
| Lognormal | 50 | 0.1 h | fast-run log |
| Logn_PL | 8 | 0.03 h | fast-run log |
| Weibull | 50 | 0.3 h | fast-run log |
| GLH_p1 | 20 | 0.5 h | measured |
| GIG_2p | 25 | 1.9 h | fast-run log |
| GIG_3p | 8 | 14.1 h | fast-run log |
| GLH | 8 | 26.9 h | fast-run log |
| GLH_p05 | 20 | 75.5 h | measured |
| NIG | 50 | 220 h | measured |
| **total** | | **~340 h** | |

Three functions account for 96% of that: NIG, GLH_p05 and GLH all evaluate their CDF through
SciPy's `norminvgauss` / `genhyperbolic`, which integrates numerically on every residual
evaluation. At the archival `maxfev = 1e6` they are far worse again.

## Not yet grid-searched — cheap

Measured by running the grid on 200 random samples and scaling.

| Family | Grid | Time |
|---|---|---|
| 9 two-parameter (Algeb, Erf_PL, Exp_PL, Gamma, Tanh, Log_exp, Ln, LLaplace, PL) | 50 guesses | 0.9 h |
| 7 three-parameter (Logn_Weib, Logn_Tanh, LSLaplace, PL_Tanh, pPL_Exp, PL_omExp, PL_Weib) | 8 guesses | 0.2 h |
| the same seven | 133 guesses | 4.3 h |

**Filling the gap costs 1.1 h at the existing grid densities, or 5.1 h with the denser
three-parameter grid** — single core. Split by function across 16 cores: about 15 minutes, or
1.3 hours.

## Reading

Cost and merit point in opposite directions here. The expensive functions are already done;
the sixteen that lack a grid search are the cheapest in the set. The three-parameter ones
among them are also the ones that measurably improve under a grid — LSLaplace's median BIC
goes from 37.5 to 25.5 on a 120-sample probe — so the denser grid is where the 5 hours should
go.

Doing it changes Tables S2, S3 and S4 and Fig. 4, and needs a co-author decision. It does not
change the paper's conclusions: on that probe GIG_3p's count of outright wins was unchanged
and GIG_2p's moved by one. See `docs/caveats.md`.

Re-running all 25 uniformly is a different proposition: about **340 h single core, roughly
22 h across 16 cores**, and a uniform grid does not help because it would raise GLH and GIG_3p
from 8 guesses to 27.

Raw measurements: `outputs/fit_cost_per_call.csv`.
