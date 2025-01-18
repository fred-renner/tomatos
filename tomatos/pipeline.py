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
# basically the more you hints you give what is fixed, more code can be
# optimized for hardware accelaration
@partial(jax.jit, static_argnames=["config"])
def loss_fn(
    pars,
    data,
    config,
    bandwidth,
    slope,
    scale,
    validate_only=False,
):
    ##### analysis computation could be here, need on the fly input for this scaling

    data = tomatos.histograms.apply_cuts(pars, data, config)

    hists = tomatos.histograms.get_hists(pars, data, config, scale)
    print(hists)
    # build our statistical model, and calculate the loss!
    model, hists = tomatos.workspace.model_from_hists(
        hists,
        config,
    )

    # if you want s/b discrimination, no need to do anything complex!
    if loss_type == "bce":
        return tomatos.utils.bce(data=data_dct, pars=pars["nn"], nn=nn), hists

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
