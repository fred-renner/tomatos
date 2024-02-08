#!/usr/bin/env python3
import argparse
import json
import logging

import equinox as eqx
import jax
import pyhf

import atos.batching
import atos.configuration
import atos.nn_setup
import atos.optimization
import atos.plotting
import atos.preprocess
import atos.utils

JAX_CHECK_TRACER_LEAKS = True
import jax.numpy as jnp
import numpy as np
import relaxed
import json


jax.config.update("jax_enable_x64", True)

parser = argparse.ArgumentParser()
parser.add_argument("--bins", type=int, required=True)
args = parser.parse_args()


def run():
    config = atos.configuration.Setup(args)

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
    logging.info(json.dumps(atos.utils.to_python_lists(config.__dict__), indent=4))

    data = atos.preprocess.prepare_data(config)
    logging.info(f"datasets: {len(data)}")
    init_pars, nn, nn_setup = atos.nn_setup.init(config)

    train, valid_test = atos.batching.split_data(data, ratio=config.train_data_ratio)
    valid, test = atos.batching.split_data(valid_test, ratio=0.8)
    logging.info(f"train size: {train[0].shape[0]}")
    logging.info(f"valid size: {valid[0].shape[0]}")
    logging.info(f"test size: {test[0].shape[0]}")

    batch_iterator = atos.batching.make_iterator(train, batch_size=config.batch_size)

    best_params, metrics = atos.optimization.run(
        config=config,
        valid=valid,
        test=test,
        batch_iterator=batch_iterator,
        init_pars=init_pars,
        nn=nn,
    )

    bins, yields = atos.utils.get_hist(config, nn, best_params, data)

    results = {
        "config": atos.utils.to_python_lists(config.__dict__),
        "metrics": atos.utils.to_python_lists(metrics),
        "bins": atos.utils.to_python_lists(bins),
        "yields": atos.utils.to_python_lists(yields),
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
    config = atos.configuration.Setup(args)
    results = {}
    with open(config.metadata_file_path, "r") as file:
        results = json.load(file)

    atos.plotting.plot_metrics(
        results["metrics"],
        results["config"],
    )
    atos.plotting.hist(
        results["config"],
        results["bins"],
        results["yields"],
    )


if __name__ == "__main__":
    # run()
    plot()
