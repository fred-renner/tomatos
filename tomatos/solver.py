from functools import partial

import matplotlib.pyplot as plt
import optax
from jaxopt import OptaxSolver

import tomatos.pipeline
import tomatos.utils


def setup(config, pars):
    # there are many schedules you can play with
    # https://optax.readthedocs.io/en/latest/api/optimizer_schedules.html#

    # this has worked particularly well and has a strong regularization effect
    # due to in between large lr
    if "linear_cycle" in config.lr_schedule:
        lr_schedule = optax.linear_onecycle_schedule(
            transition_steps=config.num_steps,
            peak_value=config.lr,
            pct_start=0.3,
            div_factor=5,
            final_div_factor=10,
            pct_final=0.85,
        )
    elif "constant" in config.lr_schedule:
        lr_schedule = optax.constant_schedule(config.lr)

    learning_rates = [lr_schedule(i) for i in range(config.num_steps)]

    # nice to plot this right away
    plt.figure(figsize=(6, 5))
    plt.plot(learning_rates)
    plt.yscale("log")
    plt.xlabel("Batch")
    plt.ylabel("Learning Rate")
    plt.tight_layout()
    plt.savefig(config.plot_path + "lr_schedule.pdf")
    plt.close()

    # successively apply gradient updates for gradient transformations with
    # optax.chain
    # https://optax.readthedocs.io/en/latest/api/combining_optimizers.html

    # mask gradient updates only for passed vars
    def mask(pars: dict, vars: list):
        return {key: key in vars for key in pars}

    # limiting bandwidth and cut updates is important to avoid gradient
    # explosion for these
    optimizer = optax.chain(
        optax.zero_nans(),  # if nans, zero out, otherwise opt breaks entirely
        optax.adam(lr_schedule),
        optax.masked(
            optax.clip(max_delta=config.update_limit_bw),
            mask(pars, ["bw"]),
        ),
        optax.masked(
            optax.clip(max_delta=config.update_limit_cuts),
            mask(pars, [key for key in pars.keys() if "cut_" in key]),
        ),
    )

    # has_aux allows, to return additional values from loss_fn than just the
    # loss value
    # dont jit the tomatos.pipeline.loss_fn, only literally jits this function
    # and will fail in the current setup
    return OptaxSolver(tomatos.pipeline.loss_fn, opt=optimizer, has_aux=True, jit=False)
