from typing import Callable, Iterable

import jax.numpy as jnp
import neos
import pyhf

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
    sample_names: Iterable[str],  # we're using a list of dict keys for bookkeeping!
    config: object,
    include_bins=True,
    do_m_hh=False,
    do_systematics=False,
    do_stat_error=False,
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
        # logging.info("Cuts are fixed in pipeline.py")
        # pars["vbf_cut"] = 0.0768125547167431
        # pars["eta_cut"] = 0.16166851655920972
        hists = tomatos.histograms.hists_from_nn(
            nn_pars=pars["nn_pars"],
            nn=nn,
            config=config,
            vbf_cut=pars["vbf_cut"],
            eta_cut=pars["eta_cut"],
            data=data_dct,
            bandwidth=bandwidth,  # for the bKDEs
            bins=bins,
        )

    # if you want s/b discrimination, no need to do anything complex!
    if loss_type.lower() in ["bce", "binary cross-entropy"]:
        return tomatos.utils.bce(data=data_dct, pars=pars["nn_pars"], nn=nn), hists

    # build our statistical model, and calculate the loss!
    model = tomatos.workspace.model_from_hists(
        do_m_hh, hists, config, do_systematics, do_stat_error
    )

    # # this particular fit_lr quite influences the minimization
    return (
        neos.loss_from_model(model, loss=loss_type, fit_lr=1e-3),
        hists,
    )  # , fit_lr=1e-5)


# could actually think of something with sin or cos...
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
