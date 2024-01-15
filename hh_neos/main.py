#!/usr/bin/env python3
import pickle
import json
import jax
import pyhf

import hh_neos.batching
import hh_neos.configuration
import hh_neos.nn_architecture
import hh_neos.optimization
import hh_neos.preprocess
import hh_neos.plotting
import hh_neos.utils
from pprint import pprint
import sys
import argparse
import equinox as eqx

JAX_CHECK_TRACER_LEAKS = True
import numpy as np
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

parser = argparse.ArgumentParser()
parser.add_argument("--bins", type=int)
args = parser.parse_args()


def run():
    config = hh_neos.configuration.Setup(args)
    sys.stdout = hh_neos.utils.Logger(config)
    pprint(vars(config))
    data = hh_neos.preprocess.prepare_data(config)
    print([x.shape for x in data])
    init_pars, nn, nn_setup = hh_neos.nn_architecture.init(config)
    print(init_pars)
    train, test = hh_neos.batching.split_data(data, train_size=config.train_data_ratio)
    batch_iterator = hh_neos.batching.make_iterator(train)

    best_params, metrics = hh_neos.optimization.run(
        config=config,
        data=data,
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
        print(config.metadata_file_path)

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

