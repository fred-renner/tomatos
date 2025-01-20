import jax.numpy as jnp


def min_events_per_bin(h, thresh, intensity=0.01):
    # check where the hist is smaller than thresh and use the relative
    # deviation per bin to penalize
    bin_penalty = jnp.where(h < thresh, (thresh - h) / h, 0)
    penalty = jnp.sum(bin_penalty) * intensity
    return penalty


def penalize_loss(loss_value, hists):

    # reasonable for fitting, also critical to distribute events with bandwidth
    # reduction mechansim
    loss_value += min_events_per_bin(hists["bkg_estimate"], thresh=5, intensity=0.01)

    return loss_value
