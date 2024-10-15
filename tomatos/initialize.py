from typing import Callable, Iterable

import jax.numpy as jnp
import neos
import pyhf
import tomatos.histograms
import tomatos.utils
import tomatos.workspace

pyhf.set_backend("jax")


Array = jnp.ndarray


def pipeline(
    pars: dict[str, Array],
    data: tuple[Array, ...],
    nn: Callable,
    bandwidth: float,
    sample_names: Iterable[str],  # we're using a list of dict keys for bookkeeping!
    config: object,
    bins: Array,
) -> float:
    data_dct = {k: v for k, v in zip(sample_names, data)}
    data_dct = {"bkg": data_dct["bkg"]}
    hists = tomatos.histograms.get_hists(
        nn_pars=pars["nn_pars"],
        nn=nn,
        config=config,
        vbf_cut=0.00001,
        eta_cut=0.00001,
        data=data_dct,
        bandwidth=bandwidth,
        slope=1e6,
        bins=bins,
    )

    def variance(h):
        # working with numbers around 1 faster
        h /= jnp.mean(h)
        # Calculate the mean of the h
        mean = jnp.mean(h)
        # Compute the variance from the mean (squared difference)
        variance = jnp.mean((h - mean) ** 2)
        return variance

    return variance(hists["bkg"]), hists
