#!/usr/bin/env python3
import argparse
import json
import logging

import equinox as eqx
import jax
import pyhf

import hh_neos.batching
import hh_neos.configuration
import hh_neos.nn_architecture
import hh_neos.optimization
import hh_neos.plotting
import hh_neos.preprocess
import hh_neos.utils

JAX_CHECK_TRACER_LEAKS = True
import jax.numpy as jnp
import numpy as np
import relaxed
import json


jax.config.update("jax_enable_x64", True)

parser = argparse.ArgumentParser()
parser.add_argument("--bins", type=int)
args = parser.parse_args()


def run():
    config = hh_neos.configuration.Setup(args)

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
    logging.info(json.dumps(hh_neos.utils.to_python_lists(config.__dict__), indent=4))

    data = hh_neos.preprocess.prepare_data(config)
    logging.info(f"datasets: {len(data)}")
    init_pars, nn, nn_setup = hh_neos.nn_architecture.init(config)

    train, valid_test = hh_neos.batching.split_data(data, ratio=config.train_data_ratio)
    valid, test = hh_neos.batching.split_data(valid_test, ratio=0.8)
    logging.info(f"train size: {train[0].shape[0]}")
    logging.info(f"valid size: {valid[0].shape[0]}")
    logging.info(f"test size: {test[0].shape[0]}")

    batch_iterator = hh_neos.batching.make_iterator(train, batch_size=config.batch_size)

    best_params, metrics = hh_neos.optimization.run(
        config=config,
        valid=valid,
        test=test,
        batch_iterator=batch_iterator,
        init_pars=init_pars,
        nn=nn,
    )

    bins, yields = hh_neos.utils.get_hist(config, nn, best_params, data)

    results = {
        "config": hh_neos.utils.to_python_lists(config.__dict__),
        "metrics": hh_neos.utils.to_python_lists(metrics),
        "bins": hh_neos.utils.to_python_lists(bins),
        "yields": hh_neos.utils.to_python_lists(yields),
    }

    # save model to file
    model = eqx.combine(best_params["nn_pars"], nn_setup)
    eqx.tree_serialise_leaves(config.results_path + "neos_model.eqx", model)

    # save metadata
    with open(config.metadata_file_path, "w") as file:
        json.dump(results, file)
        logging.info(config.metadata_file_path)

    plot()


def plot():
    config = hh_neos.configuration.Setup(args)
    results = {}
    with open(config.metadata_file_path, "r") as file:
        results = json.load(file)

    hh_neos.plotting.plot_metrics(
        results["metrics"],
        results["config"],
    )
    hh_neos.plotting.hist(
        results["config"],
        results["bins"],
        results["yields"],
    )


if __name__ == "__main__":
    # run()
    plot()
