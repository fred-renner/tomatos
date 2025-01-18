from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jsp


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


@partial(jax.jit, static_argnames=["config"])
def get_hists(
    pars,
    data,
    config,
    scale,
):
    # any magic in here is always just a call of the upper hist() function

    # this will hold the hists at the end
    hists = {}

    if config.include_bins:
        bins = jnp.array([0, *pars["bins"], 1])
    else:
        bins = config.bins

    # get nn output
    if config.objective == "cls_nn":

        nn = eqx.combine(pars["nn"], config.nn_arch)

        def predict_sample(i):
            # Forward pass for one sample
            sample_data = data[i, :, : config.nn_inputs_idx_end]
            sample_nn_output = jax.vmap(nn)(sample_data)
            return sample_nn_output.ravel()

        nn_output = jax.vmap(predict_sample)(jnp.arange(len(config.sample_sys)))

    def select_weights(i, select):
        # here you can add event selections, by multiplying to the event weights
        base_weights = data[i, :, config.weight_idx]
        if select == "nominal":
            return base_weights
        elif select == "btag_1":
            return base_weights * data[i, :, config.vars.index("bool_btag_1")]
        elif select == "btag_2":
            return base_weights * data[i, :, config.vars.index("bool_btag_2")]
        elif select == "my_sf_unc_up":
            return data[i, :, config.vars.index("weight_my_sf_unc_up")]
        elif select == "my_sf_unc_down":
            return data[i, :, config.vars.index("weight_my_sf_unc_down")]

    def sample_hist(i, select, w2=False):
        # this is a helper vectorization over the first ith dimension
        if config.objective == "cls_var":
            data = data[i, :, config.cls_var_idx]
        elif config.objective == "cls_nn":
            data = nn_output[i, :]
        weights = select_weights(i, select=select)
        if w2:
            weights = jnp.power(weights, 2)
        h = hist(
            data=data,
            weights=weights,
            bandwidth=pars["bw"],
            bins=bins,
        )
        # scale works also as estimate for w2
        return h * scale[i]

    # Compute histograms for all samples_sys per selection sel
    for sel in ["btag_1", "btag_2"]:
        # this holds len(hists)=n_samples
        hists_vector = jax.vmap(lambda i: sample_hist(i, select=sel))(
            jnp.arange(len(config.sample_sys))
        )
        for i, sample_sys in enumerate(config.sample_sys):
            hists[sample_sys + "_" + sel] = hists_vector[i]

    # Compute w2 histograms for the final selection btag_2 only
    hists_nominal_w2_vector = jax.vmap(
        lambda i: sample_hist(i, select="btag_2", w2=True)
    )(jnp.arange(len(config.samples)))

    # workaround stat up and down hists
    for i, sample in enumerate(config.samples):
        sample_NOSYS = sample + "_" + config.nominal + "_btag_2"
        hists[sample_NOSYS + "_w2"] = hists_nominal_w2_vector[i]
        sigma = jnp.sqrt(hists_nominal_w2_vector[i])
        hists[sample_NOSYS + "_stat_up"] = hists[sample_NOSYS] + sigma
        hists[sample_NOSYS + "_stat_down"] = hists[sample_NOSYS] - sigma

    # some special signal weight unc
    signal_idx = config.sample_sys.index("ggZH125_vvbb_NOSYS")
    hists["signal_my_sf_unc_up"] = sample_hist(signal_idx, select="my_sf_unc_up")
    hists["signal_my_sf_unc_down"] = sample_hist(signal_idx, select="my_sf_unc_down")

    return hists
