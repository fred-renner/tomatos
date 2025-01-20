from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import tomatos.utils

Array = jnp.ndarray
import relaxed
import equinox as eqx


# modified from relaxed and added weights, its nice to have it here to see
# whats going on
@partial(jax.jit, static_argnames=["density", "reflect_infinities"])
def hist(
    data: Array,
    weights: Array,
    bins: Array,
    bandwidth: float,  # | None = None,
    density: bool = False,
    reflect_infinities: bool = False,
) -> Array:
    """Differentiable histogram, defined via a binned kernel density estimate (bKDE).

    Parameters
    ----------
    data : Array
        1D array of data to histogram.
    weights : Array
        weights to data
    bins : Array
        1D array of bin edges.
    bandwidth : float
        The bandwidth of the kernel. Bigger == lower gradient variance, but more bias.
    density : bool
        Normalise the histogram to unit area.
    reflect_infinities : bool
        If True, define bins at +/- infinity, and reflect their mass into the edge bins.

    Returns
    -------
    Array
        1D array of bKDE counts.
    """

    # 7.2.3 nathan thesis
    # get cumulative counts (area under kde) for each set of bin edges

    # bins=np.array([0,1,2,3])
    # bins.reshape(-1, 1)
    # array([[0],
    #        [1],
    #        [2],
    #        [3]])
    cdf = jsp.stats.norm.cdf(bins.reshape(-1, 1), loc=data, scale=bandwidth)
    # multiply with weight
    cdf = cdf * weights

    # sum kde contributions in each bin
    counts = (cdf[1:, :] - cdf[:-1, :]).sum(axis=1)

    if density:  # normalize by bin width and counts for total area = 1
        db = jnp.array(jnp.diff(bins), float)  # bin spacing
        counts = counts / db / counts.sum(axis=0)

    if reflect_infinities:
        counts = (
            counts[1:-1]
            + jnp.array([counts[0]] + [0] * (len(counts) - 3))
            + jnp.array([0] * (len(counts) - 3) + [counts[-1]])
        )

    return counts


@partial(jax.jit, static_argnames=["config"])
def apply_cuts(pars, data, config):
    # if you wonder why not doing cuts like this:
    # var = jnp.where(var > cut_param, var, 0)
    # it's discontinuous --> no gradient

    # apply cuts to weights
    cut_weights = jnp.ones(data.shape[:2])
    # collect them over all cuts and apply to weights once
    for var, var_dict in config.opt_cuts.items():
        # get the sigmoid weights for var_cut
        cut_weights *= relaxed.cut(
            data=data[:, :, var_dict["idx"]],
            cut_val=pars[var + "_cut"],
            slope=config.slope,
            keep=var_dict["keep"],
        )
    # apply to weights
    # inside a jitted function this is not a copy
    # https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ndarray.at.html
    data = data.at[:, :, config.weight_idx].multiply(cut_weights)
    return data


def select_events(data, config):
    # apply selections via weights
    # on the fly here increases the effective events that can be
    # processed per batch, if selections are written at the preselection it
    # would double the input vars for each selection, e.g. j1_pt_btag_1,
    # j1_pt_btag_2
    base_weights = data[:, :, config.weight_idx]
    btag_1 = data[:, :, config.vars.index("bool_btag_1")]
    btag_2 = data[:, :, config.vars.index("bool_btag_2")]
    h_m = tomatos.utils.min_max_unscale(
        config,
        data[:, :, config.vars.index("h_m")],
        config.vars.index("h_m"),
    )

    SR = (110e3 < h_m) & (h_m < 130e3)
    VR = (100e3 < h_m) & (h_m < 110e3) | (140e3 < h_m) & (h_m < 150e3)
    CR = (80e3 < h_m) & (h_m < 100e3) | (150e3 < h_m) & (h_m < 170e3)

    weights = {
        "SR_btag_1": base_weights * SR * btag_1,
        "SR_btag_2": base_weights * SR * btag_2,
        "VR_btag_1": base_weights * VR * btag_1,
        "VR_btag_2": base_weights * VR * btag_2,
        "CR_btag_1": base_weights * CR * btag_1,
        "CR_btag_2": base_weights * CR * btag_2,
        "SR_btag_2_my_sf_unc_up": data[:, :, config.vars.index("weight_my_sf_unc_up")]
        * SR
        * btag_2,
        "SR_btag_2_my_sf_unc_down": data[
            :, :, config.vars.index("weight_my_sf_unc_down")
        ]
        * SR
        * btag_2,
    }
    return weights


@partial(jax.jit, static_argnames=["config"])
def get_hists(
    pars,
    data,
    config,
    scale,
):
    # any magic in here will at the end just call the upper hist() function

    selections = [
        "SR_btag_1",
        "SR_btag_2",
        "VR_btag_1",
        "VR_btag_2",
        "CR_btag_1",
        "CR_btag_2",
    ]
    # this will hold: hists[sel][sample][sys]
    hists = {sel: {sample: {} for sample in config.samples} for sel in selections}

    bins = jnp.array([0, *pars["bins"], 1]) if config.include_bins else config.bins
    # event selections
    sel_weights = select_events(data, config)

    # get nn output
    if config.objective == "cls_nn":

        nn = eqx.combine(pars["nn"], config.nn_arch)

        # this also illustrates nicely how jax.vmap works
        def predict_sample(i):
            # Forward pass for one sample
            sample_data = data[i, :, : config.nn_inputs_idx_end]
            # vmap all events in batch for this sample
            sample_nn_output = jax.vmap(nn)(sample_data)
            # flatten output of [[out_1], [out_2],...]
            return sample_nn_output.ravel()

        nn_output = jax.vmap(predict_sample)(jnp.arange(len(config.sample_sys)))

    # bandwidth and bins dont change in get_hists(), this is essentially the
    # same as setting static_argnames of these everywhere here
    hist_ = partial(hist, bandwidth=pars["bw"], bins=bins)

    def compute_hist(i, weights, w2=False):
        # calc hist per sample i helper for vmap
        if config.objective == "cls_var":
            sample_data = data[i, :, config.cls_var_idx]
        elif config.objective == "cls_nn":
            sample_data = nn_output[i, :]
        sample_weights = weights[i, :]
        if w2:
            sample_weights = jnp.power(sample_weights, 2)
        # scale works also as estimate for w2
        return hist_(sample_data, sample_weights) * scale[i]

    for sel in selections:
        # let's see if we can afford this memory-wise, otherwise
        # fall back to sequential processing of i
        hists_vector = jax.vmap(lambda i: compute_hist(i, weights=sel_weights[sel]))(
            jnp.arange(len(config.sample_sys))
        )
        for sample_sys, h in zip(config.sample_sys, hists_vector):
            sample, sys = config.sample_sys_dict[sample_sys]
            hists[sel][sample][sys] = h

    hists = extra_hists(config, compute_hist, sel_weights, hists)

    return hists


def extra_hists(config, compute_hist, sel_weights, hists):

    # Compute w2 histograms for the final selection btag_2 only
    # going over len(config.samples) works because NOSYS are the first ones per
    # sample, see config
    hists_nominal_w2_vector = jax.vmap(
        lambda i: compute_hist(i, weights=sel_weights["SR_btag_2"], w2=True)
    )(jnp.arange(len(config.samples)))

    # workaround stat up and down hists
    for sample, h_w2 in zip(config.samples, hists_nominal_w2_vector):
        sigma = jnp.sqrt(h_w2)
        hists["SR_btag_2"][sample][config.nominal + "_stat_up"] = (
            hists["SR_btag_2"][sample][config.nominal] + sigma
        )
        hists["SR_btag_2"][sample][config.nominal + "_stat_down"] = (
            hists["SR_btag_2"][sample][config.nominal] - sigma
        )

    # dedicated stat error calc for bkg estimate
    h_w2_SR_btag_1 = compute_hist(
        config.sample_sys.index("bkg_NOSYS"),
        weights=sel_weights["SR_btag_1"],
    )
    sigma = jnp.sqrt(h_w2_SR_btag_1)
    hists["SR_btag_1"]["bkg"]["NOSYS_stat_up"] = h_w2_SR_btag_1 + sigma
    hists["SR_btag_1"]["bkg"]["NOSYS_stat_down"] = h_w2_SR_btag_1 - sigma

    # some special signal weight unc, e.g. btag sf
    signal_idx = config.sample_sys.index("ggZH125_vvbb_NOSYS")
    hists["SR_btag_2"]["ggZH125_vvbb"]["my_sf_unc_up"] = compute_hist(
        signal_idx, weights=sel_weights["SR_btag_2_my_sf_unc_up"]
    )
    hists["SR_btag_2"]["ggZH125_vvbb"]["my_sf_unc_down"] = compute_hist(
        signal_idx, weights=sel_weights["SR_btag_2_my_sf_unc_down"]
    )

    return hists
