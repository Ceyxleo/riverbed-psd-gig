# Lower Rhine input — not distributed

The independent validation samples used in the paper come from:

> Chowdhury, K., Blom, A., Ylla Arbós, C. & Schielen, R. M. J.
> *Schematized model of the Lower Rhine River and its branches in SOBEK RE*, version 2.
> 4TU.ResearchData (2025). https://doi.org/10.4121/eb78267a-137b-4f61-bb7e-6549915a24c7

That dataset is licensed **CC BY-NC-ND 4.0** — no derivatives, non-commercial — so neither
the samples nor the per-sample fitted parameters derived from them can be redistributed here
or in the data deposit. Only aggregate results, as published in the paper, are included.

## Building the input yourself

1. Download the dataset from the DOI above and accept its licence.
2. Extract the bed-sediment gradation table for the Lower Rhine and its branches into a CSV
   with one row per sample. Name each sieve column for its size in millimetres; values are
   cumulative percent finer.
3. Save it here as `rhine_data_to_fit.csv`. The expected shape is:

   ```
   river_km,river_name,site_no,0.5 mm,2 mm,8 mm,31.5 mm,125 mm,sample_ID
   849,Bovenrijn-Waal,Bovenrijn-Waal_849,8.61,19.4,41.77,92.81,100.0,rhine_1
   850,Bovenrijn-Waal,Bovenrijn-Waal_850,7.8,15.95,36.61,89.84,100.0,rhine_2
   ```

   `site_no` is `river_name` joined to `river_km`; `sample_ID` is `rhine_<n>`.
   The published analysis uses 67 samples with 5 sieve grades each.

4. Then run:

   ```bash
   make rhine
   ```

   which fits all 25 functions (`configs/fit_rhine.yaml`, `maxfev = 20000`), writes the
   summary tables to `outputs/rhine/`, and draws Supplementary Fig. S5.

`configs/fit_rhine.yaml` reads the grain size from the column name, so any column named
`"<number> mm"` is picked up automatically and the exact set of sieve sizes does not have to
match the published one.
