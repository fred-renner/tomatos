#!/usr/bin/env python3
import argparse
import logging
import equinox as eqx
import jax
import tomatos.config
import tomatos.training
import tomatos.plotting
import tomatos.preprocess
import tomatos.utils
import tomatos.giffer
import tomatos.nn
import json
import numpy as np
from tomatos.config import get_config

jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_platforms", "cpu")
# some debugging options
jax.numpy.set_printoptions(precision=5, suppress=True, floatmode="fixed")
# jax.numpy.set_printoptions(suppress=True)
# jax.config.update("jax_disable_jit", True)

# jax.config.update("jax_check_tracer_leaks", True)
# useful to find the cause of nan's
# jax.config.update("jax_debug_nans", True)

parser = argparse.ArgumentParser()
parser.add_argument("--bins", type=int, default=5)
parser.add_argument("--suffix", type=str, default="")
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--debug", action="store_true", default=False)
parser.add_argument("--k-fold", type=int, default=0)
parser.add_argument("--loss", type=str, default="cls")
parser.add_argument("--aux", type=float, default=1)
parser.add_argument("--aux-list", type=lambda s: [float(item) for item in s.split("_")])

args = parser.parse_args()

config = get_config(args)


def run():

    tomatos.utils.setup_logger(config)
    if config.plot_inputs:
        tomatos.plotting.plot_inputs(config)

    # need to write scaler to json then load into config
    tomatos.preprocess.run(config)

    nn_model = tomatos.nn.NeuralNetwork(n_features=config.nn_inputs_idx_end)
    # split model into parameters to optimize and the nn architecture
    nn_pars, nn_arch = eqx.partition(nn_model, eqx.is_array)

    # save config
    with open(config.config_file_path, "w") as file:
        json.dump(tomatos.utils.to_python_lists(config.__dict__), file)

    config.nn_arch = nn_arch
    config.init_pars = tomatos.utils.init_opt_pars(config, nn_pars)

    tomatos.training.run(config)

    # plot()
    # tomatos.giffer.run(config.model)


def plot():
    config = tomatos.configuration.Setup(args)
    results = {}
    with open(config.metadata_file_path, "r") as file:
        results = json.load(file)

    with open(config.metrics_file_path, "r") as file:
        metrics = json.load(file)
    tomatos.plotting.main(
        results["config"],
        results["bins"],
        results["yields"],
        metrics,
    )
