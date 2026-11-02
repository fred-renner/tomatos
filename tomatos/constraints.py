import jax.numpy as jnp
import numpy as np

# The kde bandwidth is the strongest regularization in here. A wide kernel
# smooths the loss landscape, every bin sees events and gets gradients from
# far away, a narrow one approaches the sharp histogram that we actually
# evaluate. It has no useful gradient itself (the loss would always prefer
# the optimistic smearing of a wide kernel), so it is set from outside:
# start wide and go as low as the optimization allows.

# The same bandwidth also sets the width of the sigmoid cuts (select.cuts).

# per step factors, fixed on purpose, small enough to let the optimizer
# follow the slowly changing landscape
BW_SHRINK = 0.97
BW_GROW = 1.1
# safety only: relative change of the train loss in one step that counts as
# unstable
MAX_LOSS_JUMP = 0.5
# init: share of each sample's yield every bin should see
MIN_BIN_SHARE = 0.01
# numerical floor of the kde, not a tuning parameter
BW_FLOOR = 1e-4


def bkg_hist(config, hists):
    return hists[f"{config.bkg_sample}_{config.nominal}"]


def populated(config, hists):
    # every bin has enough background, signal can be anywhere
    return bool(jnp.min(bkg_hist(config, hists)) >= config.min_bkg)


def covered(config, hists):
    # every bin sees a share of every sample, so the gradients reach all of
    # them, and has enough background
    for sample in config.samples:
        h = hists[f"{sample}_{config.nominal}"]
        if jnp.min(h) < MIN_BIN_SHARE * jnp.sum(h):
            return False
    return populated(config, hists)


def find_bw_init(config, pars, data, scale, kde_hists_fn):
    # start value only: scan down from very wide and take the narrowest
    # kernel that still covers every bin at the initial parameters. where it
    # ends up is up to the controller
    candidates = np.geomspace(1.0, 0.002, 40)
    bw_init = candidates[0]
    for bw in candidates:
        hists = kde_hists_fn({**pars, "bw": jnp.array(bw)}, data, scale)
        if not covered(config, hists):
            break
        bw_init = bw
    return bw_init


def bw_controller(config, bw, sharp_hists, loss, prev_loss, bw_start):
    # sharp_hists: counted hists at the parameters the step started from,
    # what we evaluate in the end
    # shrink while all bins are populated and the loss moves smoothly, widen
    # when a bin starves or the loss jumps. widening a starving bin smears
    # background back into it, so emptying a bin stops paying off
    jump = abs(loss - prev_loss) / max(abs(prev_loss), 1e-12)
    if not populated(config, sharp_hists):
        bw, state = bw * BW_GROW, "grow (starving bin)"
    elif not np.isfinite(loss) or jump > MAX_LOSS_JUMP:
        bw, state = bw * BW_GROW, "grow (loss jump)"
    else:
        bw, state = bw * BW_SHRINK, "shrink"
    # never wider than the start, the scan already found the widest useful
    # kernel. e.g. an untrained nn puts everything in one bin, bins starve
    # until it learned to spread its output
    return jnp.clip(bw, BW_FLOOR, bw_start), state


def opt_pars(config, opt_pars):
    # stay in jax types, numpy ones would retrigger compilation of the update
    if config.include_bins:
        # maintain order and avoid going out of 0, 1 range
        opt_pars["bins"] = jnp.clip(jnp.sort(jnp.abs(opt_pars["bins"])), 1e-6, 1 - 1e-6)
    return opt_pars
