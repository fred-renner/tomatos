import jax
import jax.numpy as jnp
import pyhf
from typing import Callable, Iterable
from functools import partial
import matplotlib.pyplot as plt
import neos
import workspace
import histograms

jax.config.update("jax_enable_x64", True)
pyhf.set_backend("jax")
import numpy as np

# matplotlib settings
plt.rc("figure", figsize=(5, 3), dpi=150, facecolor="w")
plt.rc("legend", fontsize=6)

Array = jnp.ndarray


def pipeline(
    pars: dict[str, Array],
    data: tuple[Array, ...],
    nn: Callable,
    loss: str,
    bandwidth: float,
    sample_names: Iterable[str],  # we're using a list of dict keys for bookkeeping!
    bins: Array = None,  # in case you don't want to optimise binning
    lumi: float = 10.0,  # overall scale factor
    include_bins=True,
    do_m_hh=False,
) -> float:
    # zip up our data arrays with the corresponding sample names
    data_dct = {k: v for k, v in zip(sample_names, data)}

    # if you want s/b discrimination, no need to do anything complex!
    if loss.lower() in ["bce", "binary cross-entropy"]:
        return neos.losses.bce(data=data_dct, pars=pars["nn_pars"], nn=nn)

    # use a neural network + differentiable histograms [bKDEs] to get the
    # yields
    if do_m_hh:
        hists = histograms.hists_from_mhh(
            data=data_dct,
            bins=jnp.array([0, *pars["bins"], 1]) if "bins" in pars else bins,
            bandwidth=bandwidth,
            include_bins=include_bins,
        )
    else:
        hists = histograms.hists_from_nn(
            pars=pars["nn_pars"],
            nn=nn,
            data=data_dct,
            bandwidth=bandwidth,  # for the bKDEs
            bins=jnp.array([0, *pars["bins"], 1]) if "bins" in pars else bins,
            overall_scale=lumi,
        )

    # build our statistical model, and calculate the loss!
    model = workspace.model_from_hists(hists)

    return neos.loss_from_model(model, loss=loss)
