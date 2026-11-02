VENV := .venv
BIN := $(VENV)/bin
DEMOS := toy_mass toy_mass_nosys toy_nn toy_nn_nosys

.PHONY: install demo plots test data

# jaxopt (used for the likelihood fit) pins the stack to jax 0.3.14 / Python 3.9
install:
	python3.9 -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True $(BIN)/pip install --editable .
	$(BIN)/pip install jaxlib==0.3.14 -f https://storage.googleapis.com/jax-releases/jax_releases.html
	$(BIN)/pip install pytest

# toy ntuples in tests/toy_files
data:
	$(BIN)/python tests/generate_toy_files.py

# train both demos with and without systematics (one at a time, memory)
demo: data
	for d in $(DEMOS); do $(BIN)/tomatos --config configs/$$d.yaml --prep --train --plot || exit 1; done
	$(MAKE) plots

# README figures in assets/
plots:
	$(BIN)/python scripts/readme_plots.py \
		--mass results/tomatos_cls_var_steps_300_bins_5_mass results/tomatos_cls_var_steps_300_bins_5_mass_nosys \
		--nn results/tomatos_cls_nn_steps_300_bins_5_nn results/tomatos_cls_nn_steps_300_bins_5_nn_nosys

test:
	$(BIN)/python -m pytest tests -q
