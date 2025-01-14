from typing import Callable, Iterable

import jax.numpy as jnp
import jax
import neos
import pyhf
import tomatos.histograms
import tomatos.utils
import tomatos.workspace

from functools import partial


# this fixes the compilation of static args at compile time
# bottom line, better performance
@partial(jax.jit, static_argnames=["config", "nn"])
def loss_fun(
    pars: dict[str, jnp.ndarray],
    data: tuple[jnp.ndarray, ...],
    config: object,
    nn: Callable,
    bandwidth: float,
    slope: float,
    validate_only=False,
    scale=1,
):

    # zip up our data arrays with the corresponding sample names
    data_dct = {k: v for k, v in zip(config.sample_names, data)}

    # use a neural network + differentiable histograms [bKDEs] to get the
    # yields
    bins = config.bins

    if include_bins:
        bins = jnp.array([0, *pars["bins"], 1])
    hists = tomatos.histograms.get_hists(
        nn_pars=pars["nn_pars"],
        nn=nn,
        config=config,
        vbf_cut=pars["vbf_cut"],
        eta_cut=pars["eta_cut"],
        data=data_dct,
        bandwidth=pars["bw"],
        slope=slope,
        bins=bins,
        scale=scale,
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
    if loss_type == "bce":
        return tomatos.utils.bce(data=data_dct, pars=pars["nn_pars"], nn=nn), hists

    if loss_type == "cls":
        loss_value = neos.loss_from_model(model, loss=loss_type)

        if not validate_only:
            # minimum counts via penalization
            bkg_protect_up, bkg_protect_down = tomatos.workspace.threshold_uncertainty(
                hists["bkg"],
                threshold=1,
                a=1,
                find="below",
            )
            bkg_penalty = jnp.sum(bkg_protect_up - 1) * 0.01
            loss_value += bkg_penalty

            # bkg_vr_protect_up, bkg_vr_protect_down = tomatos.workspace.threshold_uncertainty(
            #     hists["bkg_VR_xbb_2_NW"],
            #     threshold=1,
            #     a=1,
            #     find="below",
            # )
            # vr_penalty = jnp.sum(bkg_vr_protect_up - 1) * 0.01
            # loss_value += vr_penalty

            # bkg_shape_sys_protect_up, bkg_shape_sys_protect_down = threshold_uncertainty(
            #     bkg_shapesys_up,
            #     threshold=2,
            #     a=config.aux,
            #     find="above",
            # )

        return loss_value, hists
