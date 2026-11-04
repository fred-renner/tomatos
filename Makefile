VENV := .venv
BIN := $(VENV)/bin
DEMOS := toy_mass toy_nn

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

# train both demos (one at a time, memory)
demo: data
	for d in $(DEMOS); do $(BIN)/tomatos --config configs/$$d.yaml --prep --train --plot || exit 1; done
	$(MAKE) plots

# README gifs in assets/: the manual, stepwise optimization, then both
# pipelines manual vs automated
plots:
	$(BIN)/python scripts/stepwise.py configs/toy_mass.yaml
	$(BIN)/python scripts/stepwise.py configs/toy_nn.yaml
	$(BIN)/python scripts/pipeline_gif.py --run results/tomatos_cls_var_steps_300_bins_5_mass \
		--stepwise results/stepwise/mass.npz --out assets/pipeline_mass.gif
	$(BIN)/python scripts/pipeline_gif.py --run results/tomatos_cls_nn_steps_300_bins_5_nn \
		--stepwise results/stepwise/nn.npz --out assets/pipeline_nn.gif

test:
	$(BIN)/python -m pytest tests -q
