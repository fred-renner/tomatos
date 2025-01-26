import jax.numpy as jnp
import jax
import neos
import pyhf
import tomatos.histograms
import tomatos.select
import tomatos.utils
import tomatos.train_utils
import tomatos.workspace
import tomatos.constraints
import pprint

from functools import partial


def make_hists(pars, data, config, scale, validate_only=False, filter_hists=False):
    # event manipulations are done via weights to the base weights

    base_weights = data[:, :, config.weight_idx]
    cut_weights = tomatos.select.cuts(pars, data, config, validate_only)
    # apply cuts
    base_weights = jnp.multiply(base_weights, cut_weights)
    # get event selections
    sel_weights = tomatos.select.events(data, config, base_weights)

    hists = tomatos.histograms.fill_hists(
        pars, data, config, sel_weights, scale, validate_only
    )
    # calculate additional hists based on existing hists
    hists = tomatos.workspace.hist_transforms(hists)
    # flatten and reduces to the configured filter
    hists = tomatos.utils.filter_hists(config, hists) if filter_hists else hists
    return hists


def loss_fn(
    pars,  # OptaxSolver expects opt_pars as first arg
    data,
    config,
    scale,
    validate_only=False,
    filter_hists=True,
):

    hists = make_hists(pars, data, config, scale, validate_only)
    model, hists = tomatos.workspace.pyhf_model(hists, config)

    if "bce" in config.objective:
        # adjist to data you want to use, lets see if anyone wants to use
        # this
        loss_value = tomatos.train_utils.bce(ones=data[1, :, 0], zeros=data[0, :, 0])
    if "cls" in config.objective:
        loss_value = neos.loss_from_model(model, loss="cls")

        if not validate_only:
            loss_value = tomatos.constraints.penalize_loss(loss_value, hists)

    # flatten and reduces to the configured filter
    hists = tomatos.utils.filter_hists(config, hists) if filter_hists else hists
    return loss_value, hists
