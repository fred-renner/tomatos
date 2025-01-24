from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import tomatos.utils

import relaxed
import equinox as eqx


# modified from relaxed and added weights, its nice to have it here to see
# whats going on
@partial(jax.jit, static_argnames=["density", "reflect_infinities"])
def hist(
    data: jnp.array,
    weights: jnp.array,
    bins: jnp.array,
    bandwidth: float,  # | None = None,
    density: bool = False,
    reflect_infinities: bool = False,
) -> jnp.array:
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


# @partial(jax.jit, static_argnames=["config"])
def get_nn_output(pars, data, config):
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
    return nn_output


# @partial(jax.jit, static_argnames=["config", "validate_only"])
def fill_hists(
    pars,
    data,
    config,
    sel_weights,
    scale,
    validate_only,
):
    # any magic in here will at the end just call the upper hist() function

    bins = jnp.array([0, *pars["bins"], 1]) if config.include_bins else config.bins
    # make hists sharp if validation
    bw = 1e-20 if validate_only else pars["bw"]

    # bandwidth and bins dont change in get_hists(), this is essentially the
    # same as setting static_argnames of these everywhere here
    hist_ = partial(hist, bandwidth=bw, bins=bins)

    # this will hold: hists[sel][sample][sys]
    hists = {
        sel: {sample: {} for sample in config.samples} for sel in config.regions_to_sel
    }

    # get nn output
    if config.objective == "cls_nn":
        nn_output = get_nn_output(pars, data, config)

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

    for sel in config.regions_to_sel:
        hists_vector = jax.vmap(lambda i: compute_hist(i, weights=sel_weights[sel]))(
            jnp.arange(len(config.sample_sys))
        )
        for sample_sys, h in zip(config.sample_sys, hists_vector):
            sample, sys = config.sample_sys_dict[sample_sys]
            hists[sel][sample][sys] = h

    def extra_hists(hists):
        # Compute w2 histograms only for the ones we need
        # going over len(config.samples) works because NOSYS are the first ones per
        # sample, see config
        hists_nominal_w2_vector = jax.vmap(
            lambda i: compute_hist(i, weights=sel_weights["SR_btag_2"], w2=True)
        )(jnp.arange(len(config.samples)))

        # workaround stat up and down hists
        for sample, h_w2 in zip(config.samples, hists_nominal_w2_vector):
            sigma = jnp.sqrt(h_w2)
            hists["SR_btag_2"][sample][config.nominal + "_STAT_1UP"] = (
                hists["SR_btag_2"][sample][config.nominal] + sigma
            )
            hists["SR_btag_2"][sample][config.nominal + "_STAT_1DOWN"] = (
                hists["SR_btag_2"][sample][config.nominal] - sigma
            )

        # dedicated stat error calc for bkg estimate
        h_w2_SR_btag_1 = compute_hist(
            config.sample_sys.index("bkg_NOSYS"),
            weights=sel_weights["SR_btag_1"],
        )
        sigma = jnp.sqrt(h_w2_SR_btag_1)
        hists["SR_btag_1"]["bkg"]["NOSYS_STAT_1UP"] = h_w2_SR_btag_1 + sigma
        hists["SR_btag_1"]["bkg"]["NOSYS_STAT_1DOWN"] = h_w2_SR_btag_1 - sigma

        # some special signal weight unc, e.g. btag sf
        signal_idx = config.sample_sys.index("ggZH125_vvbb_NOSYS")
        hists["SR_btag_2"]["ggZH125_vvbb"]["MY_SF_UNC_1UP"] = compute_hist(
            signal_idx, weights=sel_weights["SR_btag_2_my_sf_unc_up"]
        )
        hists["SR_btag_2"]["ggZH125_vvbb"]["MY_SF_UNC_1DOWN"] = compute_hist(
            signal_idx, weights=sel_weights["SR_btag_2_my_sf_unc_down"]
        )
        return hists

    hists = extra_hists(hists)

    return hists
