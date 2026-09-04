# riverbed-psd-gig
#
#   make setup     create the conda environment
#   make inputs    copy the data deposit's tables into data/
#   make smoke     3 samples through all 25 functions (seconds)
#   make all       water metrics -> tables -> figures -> audit
#   make rhine     Lower Rhine fit, summary and Fig. S5 (needs the restricted input)
#   make deposit   rebuild the data deposit
#
PY ?= python
DEPOSIT ?= ../psd-gig-data-v1.0.0
FORMAT ?= png

.PHONY: setup inputs smoke water tables figures audit all rhine verify deposit test clean

setup:
	conda env create -f environment.yml

inputs:
	$(PY) scripts/00_fetch_inputs.py --deposit $(DEPOSIT)

smoke:
	$(PY) scripts/01_fit_functions.py --config configs/fit_smoke.yaml

water:
	$(PY) scripts/02_water_metrics.py

tables: water
	$(PY) scripts/03_bic_tables.py
	$(PY) scripts/04_significance.py

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

audit:
	$(PY) scripts/audit_numbers.py

all: tables figures audit

rhine:
	$(PY) scripts/01_fit_functions.py --config configs/fit_rhine.yaml
	$(PY) scripts/05_rhine_summary.py
	$(PY) scripts/fig_S05.py --format $(FORMAT)

verify:
	$(PY) scripts/verify_fits.py --samples 60 --functions 10 14 15

deposit:
	$(PY) scripts/build_data_deposit.py --phase all --deposit $(DEPOSIT)

test:
	$(PY) -m pytest -q

clean:
	rm -rf outputs/tables outputs/water outputs/verify outputs/audit_manuscript_numbers.csv
