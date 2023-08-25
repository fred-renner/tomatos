#!/usr/bin/env python3
from functools import partial

import jax
import pyhf

import hh_neos.batching
import hh_neos.configuration
import hh_neos.nn_architecture
import hh_neos.optimization
import hh_neos.preprocess
import hh_neos.plotting

JAX_CHECK_TRACER_LEAKS = True

jax.config.update("jax_enable_x64", True)
pyhf.set_backend("jax")


def run():
    config = hh_neos.configuration.Setup()
    data, data_min, data_max = hh_neos.preprocess.prepare_data(config)
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

    hh_neos.plotting.metrics(metrics, config)
    hh_neos.plotting.hist(config, nn, best_params, data, test, data_min, data_max)


# run()
