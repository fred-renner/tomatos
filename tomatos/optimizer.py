import matplotlib.pyplot as plt
import optax
from jaxopt import OptaxSolver


def get(config, loss_fn, params):
    # there are many schedules you can play with
    # https://optax.readthedocs.io/en/latest/api/optimizer_schedules.html#

    # this has worked particularly well and has a strong regularization effect
    # due to in between large lr
    lr_schedule = optax.linear_onecycle_schedule(
        transition_steps=config.num_steps,
        peak_value=0.001,
        pct_start=0.3,
        div_factor=25,
        final_div_factor=10000,
        pct_final=0.85,
    )

    if config.debug:
        lr_schedule = optax.constant_schedule(config.lr)

    learning_rates = [lr_schedule(i) for i in range(config.num_steps)]
    config.lr_schedule = learning_rates

    # nice to plot this right away
    plt.figure(figsize=(6, 5))
    plt.plot(learning_rates)
    # plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.tight_layout()
    plt.savefig(config.results_path + "lr_schedule.pdf")
    plt.close()

    # successively apply gradient updates for gradient transformations with
    # optax.chain
    # https://optax.readthedocs.io/en/latest/api/combining_optimizers.html

    # apply updates only to vars
    def mask(params: dict, vars: list):
        return {key: key in vars for key in params}

    optimizer = optax.chain(
        optax.zero_nans(),  # if nans, zero out, otherwise opt breaks entirely
        optax.adam(lr_schedule),
        optax.masked(
            optax.clip(max_delta=0.001), mask(params, ["bw"])
        ),  # limit bandwidth updates
        optax.masked(
            optax.clip(max_delta=0.0001),
            mask(params, config.opt_cuts.keys()),
        ),  # limit cut updates
    )

    # has_aux allows, to return additional values from loss_fn than just the
    # loss value
    return OptaxSolver(loss_fn, opt=optimizer, has_aux=True, jit=True)
