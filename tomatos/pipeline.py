import jax.numpy as jnp
import jax
import neos
import pyhf
import tomatos.histograms
import tomatos.select
import tomatos.utils
import tomatos.workspace
import tomatos.constraints
import pprint

from functools import partial


# this fixes the compilation of static args at compile time
# basically the more you hints you give what is fixed, more code can be
# optimized for hardware accelaration, both for speed and memory, if a
# statically marked variable changes results in recompilation
# https://jax.readthedocs.io/en/latest/jit-compilation.html#marking-arguments-as-static
@partial(jax.jit, static_argnames=["config", "validate_only"])
def make_hists(pars, data, config, scale, validate_only):
    # event manipulations are done via weights to the base weights
    base_weights = data[:, :, config.weight_idx]
    cut_weights = tomatos.select.cuts(pars, data, config, validate_only)
    # apply cuts
    base_weights *= cut_weights
    # get event selections
    sel_weights = tomatos.select.events(data, config, base_weights)

    hists = tomatos.histograms.fill_hists(
        pars, data, config, sel_weights, scale, validate_only
    )
    # calculate additional hists based on existing hists
    hists = tomatos.workspace.hist_transforms(hists)
    return hists


@partial(jax.jit, static_argnames=["config", "validate_only"])
def loss_fn(
    pars,  # OptaxSolver expects opt_pars as first arg
    data,
    config,
    scale,
    validate_only=False,
):

    hists = make_hists(pars, data, config, scale, validate_only)
    model, hists = tomatos.workspace.pyhf_model(hists, config)

    # do we want to keep this
    if "bce" in config.objective:
        return tomatos.utils.bce(pars, data, config), hists

    if "cls" in config.objective:
        loss_value = neos.loss_from_model(model, loss="cls")

        if not validate_only:
            loss_value = tomatos.constraints.penalize_loss(loss_value, hists)

        return loss_value, hists
