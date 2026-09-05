# riverbed-psd-gig
#
#   make setup     create the conda environment
#   make inputs    copy the data deposit's tables into data/
#   make smoke     3 samples through all 25 functions (seconds)
#   make fits      grid-search the 16 cheap functions (~27 min across JOBS cores)
#   make all       percentiles -> water metrics -> tables -> figures
#   make rhine     Lower Rhine fit, summary and Fig. S5 (needs the restricted input)
#   make deposit   rebuild the data deposit
#
PY ?= python
DEPOSIT ?= ../psd-gig-data-v1.0.0
FORMAT ?= png
JOBS ?= 10

.PHONY: setup inputs smoke fits base rhine dvalues water tables figures all \
        deposit test clean

setup:
	conda env create -f environment.yml

inputs:
	$(PY) scripts/00_fetch_inputs.py --deposit $(DEPOSIT)

smoke:
	$(PY) scripts/01_fit_functions.py --config configs/fit_smoke.yaml

fits:
	$(PY) scripts/01_fit_functions.py --config configs/fit_usgs.yaml \
	 --jobs $(JOBS) --overwrite

# Single-guess fits, for comparison only. Writes to outputs/fit_usgs/base_fits/ and
# never touches the grid results.
base:
	$(PY) scripts/01_fit_functions.py --config configs/fit_usgs.yaml \
		--mode base --jobs $(JOBS) --overwrite

dvalues:
	$(PY) scripts/02_compute_dvalues.py

water: dvalues
	$(PY) scripts/03_water_metrics.py

tables: water
	$(PY) scripts/04_bic_tables.py
	$(PY) scripts/05_significance.py

figures: tables
	$(PY) scripts/fig_01.py --format $(FORMAT)
	$(PY) scripts/fig_02.py --format $(FORMAT)
	$(PY) scripts/fig_03.py --format $(FORMAT)
	$(PY) scripts/fig_04_S04.py --formats $(FORMAT)
	$(PY) scripts/fig_04_S04.py --formats $(FORMAT) --metric RMSE \
		--input outputs/tables/rmse_matrix.tsv --output-prefix figures/Figure_S4
	$(PY) scripts/fig_05.py --format $(FORMAT)
	$(PY) scripts/fig_S01.py --format $(FORMAT)
	$(PY) scripts/fig_S02.py --format $(FORMAT)
	$(PY) scripts/fig_S03.py --format $(FORMAT)

all: tables figures

rhine:
	$(PY) scripts/01_fit_functions.py --config configs/fit_rhine.yaml \
		--jobs $(JOBS) --overwrite
	$(PY) scripts/06_rhine_summary.py
	$(PY) scripts/fig_S05.py --format $(FORMAT)

deposit:
	$(PY) scripts/build_data_deposit.py --phase all --deposit $(DEPOSIT)

test:
	$(PY) -m pytest -q

clean:
	rm -rf outputs figures/*.png figures/*.pdf
