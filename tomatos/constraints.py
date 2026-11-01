import jax
import jax.numpy as jnp
import numpy as np


def min_events_per_bin(h, thresh, intensity=0.001):
    # check where the hist is smaller than thresh and use the relative
    # deviation per bin to penalize
    bin_penalty = jnp.where(h < thresh, (thresh - h) / h, 0)
    penalty = jnp.sum(bin_penalty) * intensity
    return penalty


def smooth_min_events_per_bin(h, thresh, intensity, sharpness=10):
    # the above has a kink at thresh and goes like 1/h for emptying bins,
    # exploding gradients just where the optimizer is pushing. this is a
    # softplus version of a linear hinge in units of thresh: ~0 above,
    # ~(thresh - h) / thresh below, smooth in between. keeping the bins
    # populated is shared with the bandwidth controller below, which widens
    # the kde of a starving bin so that this gradient reaches events outside
    # of it
    deficit = (thresh - h) / thresh
    penalty = jax.nn.softplus(sharpness * deficit) / sharpness
    return jnp.sum(penalty) * intensity


def penalize_loss(loss_value, hists, config):

    # reasonable for fitting + critical to distribute events with bandwidth
    # reduction mechansim
    if getattr(config, "bw_controller", False):
        loss_value += smooth_min_events_per_bin(
            hists["SR_btag_2"]["bkg_estimate"]["NOSYS"],
            thresh=config.min_bkg,
            intensity=config.penalty_intensity,
        )
    else:
        loss_value += min_events_per_bin(
            hists["SR_btag_2"]["bkg_estimate"]["NOSYS"],
            thresh=10,
            intensity=0.001,
        )

    return loss_value


def bw_controller(config, bw, kde_hists, sharp_hists):
    # the kde bandwidth has no useful gradient: the loss would always prefer
    # a wide kernel, which smears background into every bin and looks
    # optimistic. so it is set from outside, from what the sharp analysis
    # measures. start wide, smooth landscape and every bin reachable (a
    # strong regularization), then narrow while the binned kde misrepresents
    # the counted histogram, and widen again when a bin starves.
    # both hists on the same train batch and parameters
    kde = kde_hists["bkg_estimate_NOSYS"]
    sharp = sharp_hists["bkg_estimate_NOSYS"]
    # relative per bin, floored to not chase noise in tiny bins
    mismatch = jnp.max(jnp.abs(kde - sharp) / jnp.maximum(sharp, config.min_bkg))
    shrink, grow = config.bw_rates
    # bias first: a wide kernel hides starvation from the penalty
    if mismatch > config.max_mismatch:
        bw, state = bw * shrink, "shrink"
    elif jnp.min(sharp) < config.min_bkg:
        bw, state = bw * grow, "grow"
    else:
        state = "hold"
    bw = jnp.clip(bw, config.bw_min, config.bw_max)
    return bw, state, float(mismatch)


def opt_pars(config, opt_pars):
    # a large step can gow below 0 which breaks opt since this flips
    # the cdf (not the pdf) used for histogram calculation, also not go
    # below a minimum to maintain gradients
    # stay in jax types, numpy ones would retrigger compilation of the update
    opt_pars["bw"] = jnp.maximum(config.bw_min, jnp.abs(opt_pars["bw"]))

    if config.include_bins:
        # maintain order and avoid going out of 0, 1 range
        opt_pars["bins"] = jnp.clip(jnp.sort(jnp.abs(opt_pars["bins"])), 1e-6, 1 - 1e-6)
    return opt_pars
