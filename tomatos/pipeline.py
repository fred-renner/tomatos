from typing import Callable, Iterable

import jax.numpy as jnp
import neos
import pyhf
import jax
import tomatos.histograms
import tomatos.utils
import tomatos.workspace
import logging

pyhf.set_backend("jax")


Array = jnp.ndarray


def pipeline(
    pars: dict[str, Array],
    data: tuple[Array, ...],
    nn: Callable,
    loss_type: str,
    bandwidth: float,
    slope: float,
    sample_names: Iterable[str],  # we're using a list of dict keys for bookkeeping!
    config: object,
    aux_info: dict,
    include_bins=True,
    do_m_hh=False,
    do_systematics=False,
    do_stat_error=False,
    validate_only=False,
) -> float:

    # zip up our data arrays with the corresponding sample names
    data_dct = {k: v for k, v in zip(sample_names, data)}

    # use a neural network + differentiable histograms [bKDEs] to get the
    # yields
    bins = pars["bins"] if "bins" in pars else config.bins
    if include_bins:
        bins = bin_correction(bins)
        pars["bins"] = bins

    if do_m_hh:
        hists = tomatos.histograms.hists_from_mhh(
            data=data_dct,
            bins=bins,
            bandwidth=bandwidth,
            include_bins=include_bins,
        )
    else:
        hists = tomatos.histograms.hists_from_nn(
            nn_pars=pars["nn_pars"],
            nn=nn,
            config=config,
            vbf_cut=pars["vbf_cut"],
            eta_cut=pars["eta_cut"],
            data=data_dct,
            bandwidth=bandwidth,
            slope=slope,
            bins=bins,
        )

    # build our statistical model, and calculate the loss!
    model, hists = tomatos.workspace.model_from_hists(
        do_m_hh,
        hists,
        config,
        do_systematics,
        do_stat_error,
        validate_only,
    )

    # if you want s/b discrimination, no need to do anything complex!
    if loss_type.lower() in ["bce", "binary cross-entropy"]:
        return tomatos.utils.bce(data=data_dct, pars=pars["nn_pars"], nn=nn), hists

    if validate_only:
        loss_value = neos.loss_from_model(model, loss=loss_type)
    else:
        # cant get sharp evaluation of hists here as hists with e.g. bw=1e-6
        # loses gradient, --> send in
        # start at 2
        kde_error_penality = jnp.minimum(aux_info["kde_error"], 2)
        # scale to sth comparable as cls
        kde_error_penality *= 0.01
        # this means we start when drops below 0.01
        loss_value = neos.loss_from_model(model, loss=loss_type)  # + kde_error_penality
    return loss_value, hists


# would be better to tell optimization gradient to vanish if bins gets out of bound
def bin_correction(bins):
    # make sure bins don't overlap and are unique, need to avoid loops and
    # whatnot since this is a jitted function --> jnp.where
    # find duplicates
    is_not_duplicate = bins[1:] != bins[:-1]
    # comparison does not include last value for condition
    is_not_duplicate = jnp.concatenate((is_not_duplicate, jnp.array([True])))
    # pad duplicates
    unique_increment = jnp.arange(bins.size) * 0.001
    # now return former values or pad if duplicate
    bins = jnp.where(is_not_duplicate, bins, bins + unique_increment)
    # take care of out of bound
    # bins = jnp.where(bins > 0.001, bins, 0.1)
    # bins = jnp.where(bins < 0.999, bins, 0.9)
    # monotonically increase
    bins = jnp.sort(bins)
    return bins
