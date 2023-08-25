#!/usr/bin/env python3

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax.random import PRNGKey
import pyhf
from functools import partial
import matplotlib.pyplot as plt
import neos
import relaxed
import preprocess
import workspace

JAX_CHECK_TRACER_LEAKS = True
import configuration
import optimization
import nn_architecture
import batching

jax.config.update("jax_enable_x64", True)
pyhf.set_backend("jax")
import numpy as np


def run():
    config = configuration.Setup()
    data = preprocess.prepare_data(config)
    print([x.shape for x in data])
    init_pars, nn = nn_architecture.init(config)
    train, test = batching.split_data(data, train_size=0.8)
    batch_iterator = batching.make_iterator(train)

    best_params, metrics = optimization.run(
        config=config,
        data=data,
        test=test,
        batch_iterator=batch_iterator,
        init_pars=init_pars,
        nn=nn,
    )


run()
