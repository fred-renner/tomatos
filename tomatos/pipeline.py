import jax.numpy as jnp
import jax
import neos
import pyhf
import tomatos.histograms
import tomatos.utils
import tomatos.workspace
import tomatos.constraints

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
    # get selections?
    hists = tomatos.histograms.get_hists(pars, data, config, scale)
    # calculate additional hists based on hists
    hists = tomatos.workspace.hist_transforms(hists)
    # build our statistical model, and calculate the loss!
    model, hists = tomatos.workspace.get_pyhf_model(hists, config)

    if "bce" in config.objective:
        return tomatos.utils.bce(data=data_dct, pars=pars["nn"], nn=nn), hists

    if "cls" in config.objective:
        loss_value = neos.loss_from_model(model, loss="cls")

        if not validate_only:
            loss_value = tomatos.constraints.penalize_loss(loss_value, hists)

        return loss_value, hists
