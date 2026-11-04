from functools import partial

import jax
import jax.numpy as jnp
import relaxed

# weight variations: the same events with other weights, from branches in
# the nominal ntuples. example: the toy's background uncertainty
WEIGHT_VARIATIONS = {
    "BKG_UNC": {"sample": "bkg", "up": "bkg_unc_up", "down": "bkg_unc_down"},
}


def cuts(pars, data, config, validate_only):
    # remove approximation with large slope for validation --> sharp cuts
    # in training the sigmoid width is the kde bandwidth (both act on min max
    # scaled vars), one smoothness knob for cuts and histograms, see
    # constraints.bw_controller
    slope = 1e20 if validate_only else 1 / pars["bw"]
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


def events(data, config, base_weights):
    # apply selections via weights, one fit region: everything that passes
    # the cuts. weight based systematics are just other weights of the same
    # events
    weights = {"nominal": base_weights}
    for sys, sys_dict in WEIGHT_VARIATIONS.items():
        for direction in ["up", "down"]:
            var_idx = config.vars.index(sys_dict[direction])
            weights[f"{sys}_{direction}"] = base_weights * data[:, :, var_idx]

    return weights
