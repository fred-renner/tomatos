#!/usr/bin/env python3
import argparse
import logging
import equinox as eqx
import jax
import tomatos.configuration
import tomatos.nn_builder
import tomatos.training
import tomatos.plotting
import tomatos.preprocess
import tomatos.utils
import tomatos.giffer
import json
import numpy as np

JAX_CHECK_TRACER_LEAKS = True
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_platforms", "cpu")
# some debugging options
jax.numpy.set_printoptions(precision=5, suppress=True, floatmode="fixed")
# jax.numpy.set_printoptions(suppress=True)
jax.config.update("jax_disable_jit", True)
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


def run():
    config = tomatos.configuration.Setup(args)
    tomatos.utils.setup_logger(config)
    if config.plot_inputs:
        tomatos.plotting.plot_inputs(config)

    tomatos.preprocess.run(config)
    
    nn_pars, nn_arch = tomatos.nn_builder.init(config)

    config.nn_arch = nn_arch
    # add vars to optimization
    opt_pars = {}
    opt_pars["nn"] = nn_pars

    opt_pars["bw"] = config.bw_init
    if config.include_bins:
        # exclude boundaries
        opt_pars["bins"] = config.bins[1:-1]

    for key in config.opt_cuts:
        var_idx = config.vars.index(key)
        config.opt_cuts[key]["idx"] = var_idx
        init = config.opt_cuts[key]["init"]
        init *= config.scaler.scale_[var_idx]
        init += config.scaler.min_[var_idx]
        opt_pars[key + "_cut"] = init

    logging.info(opt_pars)

    best_params, last_params, metrics, infer_metrics = tomatos.training.run(
        config=config,
        opt_pars=opt_pars,
        args=args,
    )

    bins, yields = tomatos.utils.get_hist(config, nn, best_params, data=test)

    # save model to file
    model = eqx.combine(best_params["nn"], nn_arch)
    eqx.tree_serialise_leaves(config.model_path + "epoch_best.eqx", model)

    model = eqx.combine(last_params["nn"], nn_arch)
    eqx.tree_serialise_leaves(config.model_path + "epoch_last.eqx", model)

    # save metrics
    with open(config.metrics_file_path, "w") as file:
        json.dump(tomatos.utils.to_python_lists(metrics), file)
        logging.info(config.metrics_file_path)

    # save infer_metrics
    with open(config.model_path + "infer_metrics.json", "w") as file:
        json.dump(tomatos.utils.to_python_lists(infer_metrics), file)
        logging.info(config.model_path + "infer_metrics.json")

    md = {
        "config": tomatos.utils.to_python_lists(config.__dict__),
        "bins": tomatos.utils.to_python_lists(bins),
        "yields": tomatos.utils.to_python_lists(yields),
    }

    # save metadata
    with open(config.metadata_file_path, "w") as file:
        json.dump(md, file)
        logging.info(config.metadata_file_path)

    plot()
    tomatos.giffer.run(config.model)


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
