from functools import partial
from typing import Callable, Iterable

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import neos
import pyhf
import numpy as np
import hh_neos.histograms
import hh_neos.workspace

jax.config.update("jax_enable_x64", True)
pyhf.set_backend("jax")


Array = jnp.ndarray


def pipeline(
    pars: dict[str, Array],
    data: tuple[Array, ...],
    nn: Callable,
    loss: str,
    bandwidth: float,
    sample_names: Iterable[str],  # we're using a list of dict keys for bookkeeping!
    bins: Array = None,  # in case you don't want to optimise binning
    include_bins=True,
    do_m_hh=False,
) -> float:
    # zip up our data arrays with the corresponding sample names
    data_dct = {k: v for k, v in zip(sample_names, data)}

    # if you want s/b discrimination, no need to do anything complex!
    if loss.lower() in ["bce", "binary cross-entropy"]:
        return neos.losses.bce(data=data_dct, pars=pars["nn_pars"], nn=nn)

    # make sure bins don't overlap and are unique, cannot find another way as
    # with all this jnp.where
    if "bins" in pars:
        # find duplicates
        is_not_duplicate = pars["bins"][:-1] != pars["bins"][1:]
        # comparison does not include first value for condition
        is_not_duplicate = jnp.concatenate((jnp.array([False]), is_not_duplicate))
        # pad duplicates uniquely, smaller padding breaks the whole thing...
        unique_increment = jnp.arange(pars["bins"].size) * 0.001
        # now return former values or pad if duplicate
        pars["bins"] = jnp.where(
            is_not_duplicate, pars["bins"], pars["bins"] + unique_increment
        )
        # take care of 0 and 1 
        pars["bins"] = jnp.where(
            pars["bins"]>0, pars["bins"], pars["bins"] + 0.1
        )
        pars["bins"] = jnp.where(
            pars["bins"]<1, pars["bins"], pars["bins"] - 0.1
        )
        # monotonically increase
        pars["bins"] = jnp.sort(pars["bins"])
        
    # use a neural network + differentiable histograms [bKDEs] to get the
    # yields
    if do_m_hh:
        hists = hh_neos.histograms.hists_from_mhh(
            data=data_dct,
            bins=jnp.array([0, *pars["bins"], 1]) if "bins" in pars else bins,
            bandwidth=bandwidth,
            include_bins=include_bins,
        )
    else:
        hists = hh_neos.histograms.hists_from_nn(
            pars=pars["nn_pars"],
            nn=nn,
            data=data_dct,
            bandwidth=bandwidth,  # for the bKDEs
            bins=jnp.array([0, *pars["bins"], 1]) if "bins" in pars else bins,
        )

    # build our statistical model, and calculate the loss!
    model = hh_neos.workspace.model_from_hists(do_m_hh, hists)

    return neos.loss_from_model(model, loss=loss)
