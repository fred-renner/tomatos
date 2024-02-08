from functools import partial
from typing import Callable, Iterable

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import neos
import numpy as np
import pyhf

import hh_neos.histograms
import hh_neos.utils
import hh_neos.workspace

# jax.config.update("jax_enable_x64", True)
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
        hists = hh_neos.histograms.hists_from_mhh(
            data=data_dct,
            bins=bins,
            bandwidth=bandwidth,
            include_bins=include_bins,
        )
    else:
        hists = hh_neos.histograms.hists_from_nn(
            nn_pars=pars["nn_pars"],
            nn=nn,
            data=data_dct,
            bandwidth=bandwidth,  # for the bKDEs
            bins=bins,
        )

    # if you want s/b discrimination, no need to do anything complex!
    if loss_type.lower() in ["bce", "binary cross-entropy"]:
        return hh_neos.utils.bce(data=data_dct, pars=pars["nn_pars"], nn=nn), hists

    # build our statistical model, and calculate the loss!
    model = hh_neos.workspace.model_from_hists(
        do_m_hh, hists, config, do_systematics, do_stat_error
    )

    # this particular fit_lr quite influences the minimization
    return neos.loss_from_model(model, loss=loss_type), hists  # , fit_lr=1e-5)


# could actually think of something with sin or cos...
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
