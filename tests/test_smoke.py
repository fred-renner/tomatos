import argparse
import os

import h5py
import numpy as np
import pytest
import yaml

import tomatos.config
import tomatos.preprocess
import tomatos.training

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.parametrize("demo", ["demo_cls_var", "demo_cls_nn"])
def test_few_training_steps(demo, tmp_path, monkeypatch):
    # configs use paths relative to the repo root
    monkeypatch.chdir(REPO)
    with open(os.path.join(REPO, "configs", f"{demo}.yaml")) as f:
        yml = yaml.safe_load(f)
    # the one-cycle lr schedule needs >= 5 steps
    yml["num_steps"] = 8
    yml["plot_inputs"] = False
    yml["results_path"] = str(tmp_path) + "/"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(yml))

    args = argparse.Namespace(config=str(config_path), debug=False, train=True)
    config = tomatos.config.Setup(args)
    tomatos.preprocess.run(config)
    tomatos.training.run(config)

    with h5py.File(config.metrics_file_path, "r") as m:
        assert np.all(np.isfinite(m["test_loss"][:]))
        # step 0 holds the solver's initial value (inf), single nan steps
        # are tolerated by the optimizer (zero_nans)
        assert np.any(np.isfinite(m["train_loss"][1:]))
