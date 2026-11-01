from functools import partial

import jax
import jax.numpy as jnp
import relaxed

import tomatos.utils


# @partial(jax.jit, static_argnames=["config", "validate_only"])
# @jax.jit
def cuts(pars, data, config, validate_only):
    # remove approximation with large slope for validation --> sharp cuts
    slope = 1e20 if validate_only else config.slope
    # init with shape (samples, events)
    cut_weights = jnp.ones(data.shape[:2])
    # collect them over all cuts and apply to weights once
    for var, var_dict in config.opt_cuts.items():
        # get the sigmoid weights for var_cut
        cut_weights *= relaxed.cut(
            data=data[:, :, var_dict["idx"]],
            cut_val=pars["cut_" + var],
            slope=slope,
            keep=var_dict["keep"],
        )

    # NB
    # if you wonder why not doing cuts like this:
    # var = jnp.where(var > cut_param, var, 0)
    # discontinuous --> no gradient

    return cut_weights


# @partial(jax.jit, static_argnames=["config"])
# @jax.jit
def events(data, config, base_weights):
    # apply selections via weights
    # you can also put the computation of selections at the preselection stage
    btag_1 = data[:, :, config.vars.index("bool_btag_1")]
    btag_2 = data[:, :, config.vars.index("bool_btag_2")]
    h_m_idx = config.vars.index("h_m")
    if config.objective == "cls_var":
        h_m = tomatos.utils.inverse_min_max_scale(
            config,
            data[:, :, h_m_idx],
            h_m_idx,
        )
    elif config.objective == "cls_nn":
        h_m = data[:, :, h_m_idx]

    def in_windows(windows):
        mask = jnp.zeros_like(h_m, dtype=bool)
        for lo, hi in windows:
            mask = mask | ((lo < h_m) & (h_m < hi))
        return mask

    SR = in_windows(config.regions["SR"])
    VR = in_windows(config.regions["VR"])
    CR = in_windows(config.regions["CR"])

    weights = {
        # "base_weights": base_weights,
        "SR_btag_1": base_weights * SR * btag_1,
        "SR_btag_2": base_weights * SR * btag_2,
        "VR_btag_1": base_weights * VR * btag_1,
        "VR_btag_2": base_weights * VR * btag_2,
        "CR_btag_1": base_weights * CR * btag_1,
        "CR_btag_2": base_weights * CR * btag_2,
        "SR_btag_2_my_sf_unc_up": base_weights
        * SR
        * btag_2
        * data[:, :, config.vars.index("my_sf_unc_up")],
        "SR_btag_2_my_sf_unc_down": base_weights
        * SR
        * btag_2
        * data[:, :, config.vars.index("my_sf_unc_down")],
    }

    # bkg modelling weight unc (toy), enters the estimate via SR_btag_1
    if "bkg_mod_up" in config.vars:
        for direction in ["up", "down"]:
            weights[f"SR_btag_1_bkg_mod_{direction}"] = (
                base_weights
                * SR
                * btag_1
                * data[:, :, config.vars.index(f"bkg_mod_{direction}")]
            )

    return weights
