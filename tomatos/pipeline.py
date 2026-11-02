import jax
import jax.numpy as jnp
import neos

import tomatos.histograms
import tomatos.select
import tomatos.utils
import tomatos.workspace


def make_hists(
    pars, data, config, scale, validate_only=False, filter_return_hists=False
):
    # event manipulations are done via weights to the base weights
    base_weights = data[:, :, config.weight_idx]
    cut_weights = tomatos.select.cuts(pars, data, config, validate_only)
    # apply cuts
    base_weights = jnp.multiply(base_weights, cut_weights)
    # get event selections
    sel_weights = tomatos.select.events(data, config, base_weights)
    # fill
    hists = tomatos.histograms.fill_hists(
        pars, data, config, sel_weights, scale, validate_only
    )
    hists = tomatos.workspace.zero_protect(hists)
    # flatten and filter if desired
    hists = tomatos.utils.filter_hists(config, hists) if filter_return_hists else hists
    return hists


def loss_fn(
    pars,  # OptaxSolver expects opt_pars as first arg
    data,
    config,
    scale,
    validate_only=False,
    filter_return_hists=True,
    stat_only=None,
):
    # config is not a jax compatible type (pytree), it is closed over when
    # jitting instead, see jit_loss_fn and solver.setup
    hists = make_hists(pars, data, config, scale, validate_only)
    # training ignores the systematics only in the A/B check, validation
    # always uses all of them unless asked
    if stat_only is None:
        stat_only = not validate_only and not config.train_systematics
    model = tomatos.workspace.pyhf_model(hists, config, stat_only)
    loss_value = neos.loss_from_model(model, loss="cls")

    # flatten and reduces to the configured filter
    hists = tomatos.utils.filter_hists(config, hists) if filter_return_hists else hists
    return loss_value, hists


# jitted versions with config closed over, compile once per data shape
def jit_loss_fn(config, validate_only, stat_only=None):
    return jax.jit(
        lambda pars, data, scale: loss_fn(
            pars, data, config, scale, validate_only, stat_only=stat_only
        )
    )


def jit_make_hists(config, validate_only):
    return jax.jit(
        lambda pars, data, scale: make_hists(
            pars, data, config, scale, validate_only, filter_return_hists=True
        )
    )
