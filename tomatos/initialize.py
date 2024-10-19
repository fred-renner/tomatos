from typing import Callable, Iterable

import jax.numpy as jnp
import neos
import pyhf
import tomatos.histograms
import tomatos.utils
import tomatos.workspace
import numpy as np
from jax._src.debugging import debug_print

pyhf.set_backend("jax", default=True, precision="64b")

Array = jnp.ndarray
w_CR = 0.0038


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
    data_dct = {"bkg": data_dct["bkg"], "NOSYS": data_dct["NOSYS"]}
    hists = tomatos.histograms.get_hists(
        nn_pars=pars["nn_pars"],
        nn=nn,
        config=config,
        vbf_cut=0.00001,
        eta_cut=0.00001,
        data=data_dct,
        bandwidth=pars["bw"],
        slope=1e6,
        bins=bins,
    )

    # def variance(h):
    #     # working with numbers around 1 faster
    #     h /= jnp.mean(h)
    #     # Calculate the mean of the h
    #     mean = jnp.mean(h)
    #     # Compute the variance from the mean (squared difference)
    #     variance = jnp.mean((h - mean) ** 2)
    #     return variance

    # return variance(hists["bkg"]), hists
    model, hists = get_model(hists)

    return neos.loss_from_model(model, loss="cls"), hists


def threshold_uncertainty(h, threshold, a, find="below"):
    # even though a divergent behavior at low h is desirable, power law, exp
    # etc. were too aggressive for a stable training
    if find == "below":
        penalty = jnp.where(h < threshold, -a * (h - threshold) / h, 0)
    elif find == "above":
        penalty = jnp.where(h > threshold, a * (h - threshold) / h, 0)
    up = 1 + penalty
    down = 1 - penalty

    down = jnp.where(down < 0, 0, down)

    return up, down


def get_model(hists):
    # Ensure values are above a threshold to avoid zeros or negative values
    hists["bkg"] *= w_CR
    hists = {k: jnp.where(v < 0.01, 0.01, v) for k, v in hists.items()}

    # # Calculate mean and standard deviation for bkg
    # mean_bkg = jnp.mean(hists["bkg"])
    # std_bkg = jnp.std(hists["bkg"])

    # # Calculate the z-score for the background histogram
    # z_bkg = jnp.abs((hists["bkg"] - mean_bkg) / std_bkg)

    # # Apply the z-score to adjust the background shape systematically
    # hists["bkg_shape_sys_up"] = hists["bkg"] * (1 + z_bkg)
    # hists["bkg_shape_sys_down"] = hists["bkg"] * (1 - z_bkg)

    # # Calculate mean and standard deviation for NOSYS
    # mean_nosys = jnp.mean(hists["NOSYS"])
    # std_nosys = jnp.std(hists["NOSYS"])

    # # Calculate the z-score for the NOSYS histogram
    # z_nosys = (hists["NOSYS"] - mean_nosys) / std_nosys

    # # Apply the z-score to adjust the NOSYS shape systematically
    # hists["NOSYS_shape_sys_up"] = hists["NOSYS"] * (1 + z_nosys)
    # hists["NOSYS_shape_sys_down"] = hists["NOSYS"] * (1 - z_nosys)

    # minimum counts via penalization

    bkg_protect_up, bkg_protect_down = threshold_uncertainty(
        hists["bkg"],
        threshold=1,#jnp.sum(hists["bkg"]) / len(hists["bkg"]) * 2 / 3,
        a=20,
        find="below",
    )
    hists["bkg_shape_sys_up"] = hists["bkg"] * bkg_protect_up
    hists["bkg_shape_sys_down"] = hists["bkg"] * bkg_protect_down

    # Ensure values remain above the threshold
    hists = {k: jnp.where(v < 0.01, 0.01, v) for k, v in hists.items()}

    # Build the model specification
    spec = {
        "channels": [
            {
                "name": "singlechannel",  # only one "channel" (data region)
                "samples": [
                    {
                        "name": "signal",
                        "data": hists["NOSYS"],  # signal
                        "modifiers": [
                            {
                                "name": "mu",
                                "type": "normfactor",
                                "data": None,
                            },
                            # {
                            #     "name": "NOSYS_estimate_shape",
                            #     "type": "histosys",
                            #     "data": {
                            #         "hi_data": hists["NOSYS_shape_sys_up"],
                            #         "lo_data": hists["NOSYS_shape_sys_down"],
                            #     },
                            # },
                        ],
                    },
                    {
                        "name": "background",
                        "data": hists["bkg"],  # background
                        "modifiers": [
                            {
                                "name": "bkg_estimate_shape",
                                "type": "histosys",
                                "data": {
                                    "hi_data": hists["bkg_shape_sys_up"],
                                    "lo_data": hists["bkg_shape_sys_down"],
                                },
                            },
                        ],
                    },
                ],
            },
        ],
    }

    return pyhf.Model(spec, validate=False), hists
