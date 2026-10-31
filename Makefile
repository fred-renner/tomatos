VENV := .venv
BIN := $(VENV)/bin
DEMOS := demo_cls_var demo_cls_nn

.PHONY: install demo gif test data

# jaxopt (used for the likelihood fit) pins the stack to jax 0.3.14 / Python 3.9
install:
	python3.9 -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True $(BIN)/pip install --editable .
	$(BIN)/pip install jaxlib==0.3.14 -f https://storage.googleapis.com/jax-releases/jax_releases.html
	$(BIN)/pip install pytest

# preprocess, train and plot both demos, then render the GIFs
demo:
	for d in $(DEMOS); do $(BIN)/tomatos --config configs/$$d.yaml --prep --train --plot || exit 1; done
	$(MAKE) gif

gif:
	mkdir -p assets
	$(BIN)/python scripts/make_gif.py --config configs/demo_cls_var.yaml --out assets/cls_var.gif
	$(BIN)/python scripts/make_gif.py --config configs/demo_cls_nn.yaml --out assets/cls_nn.gif

test:
	$(BIN)/python -m pytest tests -q

# regenerate tests/files from CERN Open Data (several GB download)
data:
	$(BIN)/python tests/generate_test_files.py --download
