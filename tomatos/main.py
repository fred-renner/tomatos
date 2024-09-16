#!/usr/bin/env python3
import argparse
import logging
import equinox as eqx
import jax
import tomatos.batching
import tomatos.configuration
import tomatos.nn_setup
import tomatos.optimization
import tomatos.plotting
import tomatos.preprocess
import tomatos.utils
import json
import numpy as np
import pprint

JAX_CHECK_TRACER_LEAKS = True
jax.config.update("jax_enable_x64", True)

# some debugging options
jax.numpy.set_printoptions(precision=5, suppress=True, floatmode="fixed")
# jax.numpy.set_printoptions(suppress=True)
# jax.config.update("jax_disable_jit", True)
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

    logging.basicConfig(
        filename=config.results_path + "log.txt",
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.getLogger("pyhf").setLevel(logging.WARNING)
    logging.getLogger("relaxed").setLevel(logging.WARNING)
    pprint.pprint(tomatos.utils.to_python_lists(config.__dict__))

    train, valid, test = tomatos.preprocess.prepare_data(config)
    logging.info(f"datasets: {len(train)}")
    init_pars, nn, nn_setup = tomatos.nn_setup.init(config)

    logging.info(f"train size: {train[0].shape[0]}")
    logging.info(f"valid size: {valid[0].shape[0]}")
    logging.info(f"test size: {test[0].shape[0]}")

    batch_iterator = tomatos.batching.make_iterator(train, batch_size=config.batch_size)

    best_params, last_params, metrics = tomatos.optimization.run(
        config=config,
        valid=valid,
        test=test,
        batch_iterator=batch_iterator,
        init_pars=init_pars,
        nn=nn,
        args=args,
    )

    # save best epoch
    with open(config.best_epoch_results_path, "w") as file:
        json.dump(tomatos.utils.to_python_lists(metrics["best_results"]), file)
        logging.info(config.best_epoch_results_path)

    bins, yields = tomatos.utils.get_hist(config, nn, best_params, data=train)

    results = {
        "config": tomatos.utils.to_python_lists(config.__dict__),
        "metrics": tomatos.utils.to_python_lists(metrics),
        "bins": tomatos.utils.to_python_lists(bins),
        "yields": tomatos.utils.to_python_lists(yields),
    }

    # save model to file
    model = eqx.combine(best_params["nn_pars"], nn_setup)
    eqx.tree_serialise_leaves(config.results_path + "neos_model.eqx", model)

    model = eqx.combine(best_params["nn_pars"], nn_setup)
    eqx.tree_serialise_leaves(config.results_path + "last_epoch_neos_model.eqx", model)

    # save metadata
    with open(config.metadata_file_path, "w") as file:
        json.dump(results, file)
        logging.info(config.metadata_file_path)

    plot()


def plot():
    config = tomatos.configuration.Setup(args)
    results = {}
    with open(config.metadata_file_path, "r") as file:
        results = json.load(file)

    tomatos.plotting.main(
        results["config"],
        results["bins"],
        results["yields"],
        results["metrics"],
    )
