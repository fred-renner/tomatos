#!/usr/bin/env python3
import pickle

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

JAX_CHECK_TRACER_LEAKS = True

jax.config.update("jax_enable_x64", True)
pyhf.set_backend("jax")


def run():
    config = hh_neos.configuration.Setup()
    sys.stdout = hh_neos.utils.Logger(config)
    pprint(vars(config))
    data = hh_neos.preprocess.prepare_data(config)
    print([x.shape for x in data])
    init_pars, nn = hh_neos.nn_architecture.init(config)
    train, test = hh_neos.batching.split_data(data, train_size=0.8)
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
        "config": config,
        "metrics": metrics,
        "bins": bins,
        "yields": yields,
    }
    with open(config.results_file_path, "wb") as file:
        pickle.dump(results, file)

    plot()


def plot():
    config = hh_neos.configuration.Setup()
    results = {}
    with open(config.results_file_path, "rb") as file:
        results = pickle.load(file)
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
