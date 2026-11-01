import pprint
from functools import partial

import jax
import jax.numpy as jnp
import neos
import pyhf
import relaxed

import tomatos.constraints
import tomatos.histograms
import tomatos.select
import tomatos.train_utils
import tomatos.utils
import tomatos.workspace


def discovery_z(model, fit_lr=1e-3):
    # expected discovery significance Z = sqrt(q0) on the Asimov s+b
    # dataset. neos "discovery" returns p0 instead, which spans orders of
    # magnitude and its gradient vanishes just where it gets interesting
    # (5 sigma ~ 3e-7), Z is linear in that regime
    init = jnp.asarray(model.config.suggested_init())
    hypothesis_pars = init.at[model.config.poi_index].set(1.0)
    asimov = jnp.asarray(model.expected_data(hypothesis_pars))
    # profile all nuisance parameters for the bkg only hypothesis mu = 0,
    # the fit is differentiated implicitly (relaxed, jaxopt)
    bkg_only_pars = relaxed.mle.fixed_poi_fit(
        asimov, model, poi_condition=0.0, init_pars=init, lr=fit_lr
    )
    q0 = -2 * (
        model.logpdf(bkg_only_pars, asimov)[0]
        - model.logpdf(hypothesis_pars, asimov)[0]
    )
    # tiny offset keeps the sqrt gradient finite at q0 = 0
    return jnp.sqrt(jnp.maximum(q0, 0.0) + 1e-12)


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
    # calculate additional hists based on existing hists
    hists = tomatos.workspace.hist_transforms(hists, validate_only)
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
    model, hists = tomatos.workspace.pyhf_model(hists, config, validate_only, stat_only)

    if "bce" in config.objective:
        # adjist to data you want to use, lets see if anyone wants to use
        # this
        loss_value = tomatos.train_utils.bce(ones=data[1, :, 0], zeros=data[0, :, 0])
    if "cls" in config.objective:
        # cls_* also names what is optimized (nn or var), the hypothesis
        # test itself is configurable
        if getattr(config, "test", "cls") == "discovery":
            loss_value = -discovery_z(model)
        else:
            loss_value = neos.loss_from_model(model, loss="cls")

        if not validate_only:
            loss_value = tomatos.constraints.penalize_loss(loss_value, hists, config)

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
