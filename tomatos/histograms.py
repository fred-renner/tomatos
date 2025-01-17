from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jsp


Array = jnp.ndarray
import relaxed
import equinox as eqx


# from relaxed and added weights
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
    # this will hold the hists at the end
    hists = {}

    if config.include_bins:
        bins = jnp.array([0, *pars["bins"], 1])
    else:
        bins = config.bins

    hist_ = partial(hist, bandwidth=pars["bw"], bins=bins)

    if config.mode == "cls_var":

        # looks a bit funny but it allows vectorization over sample_sys
        def sample_hist(i):
            return hist_(
                data=data[i, :, config.cls_var_idx],
                weights=data[i, :, config.weight_idx],
            )

        def sample_hist_w2(i):
            return hist_(
                data=data[i, :, config.cls_var_idx],
                weights=jnp.power(data[i, :, config.weight_idx], 2),
            )

    else:
        # combine parameters and nn architecture
        nn = eqx.combine(pars["nn"], config.nn_arch)

        # vectorization of the nn foward pass over all samples
        def predict_sample(i):
            # this is for one sample
            sample_data = data[i, :, : config.nn_inputs_idx_end]
            sample_nn_output = jax.vmap(nn)(sample_data)
            return sample_nn_output.ravel()

        nn_output = jax.vmap(predict_sample)(jnp.arange(len(config.sample_sys)))

        # looks a bit funny but it allows vectorization over sample_sys with
        # jax.vmap
        def sample_hist(i):
            return hist_(
                data=nn_output[i, :],
                weights=data[i, :, config.weight_idx],
            )

        def sample_hist_w2(i):
            return hist_(
                data=nn_output[i, :],
                weights=jnp.power(data[i, :, config.weight_idx], 2),
            )

    # vectorize hist computation
    hists_vector = jax.vmap(sample_hist)(jnp.arange(len(config.sample_sys)))
    # hists dict for each sample
    for i, sample_sys in enumerate(config.sample_sys):
        hists[sample_sys] = hists_vector[i]

    # w^2 will also be estimated and scaled up just like the counts
    # note that this is over samples only, not sys
    hists_nominal_w2_vector = jax.vmap(sample_hist_w2)(jnp.arange(len(config.samples)))

    for i, sample in enumerate(config.samples):
        hists[sample + "_" + config.nominal + "_w2"] = hists_nominal_w2_vector[i]

    # its just an illustration that you can do this but should rather be
    # treated with a designated SAMPLE/SYSTEMATIC.root
    signal_idx = config.sample_sys.index("ggZH125_vvbb_NOSYS")
    w_sf_idx_up = config.vars.index("weight_my_sf_unc_up")
    w_sf_idx_down = config.vars.index("weight_my_sf_unc_down")

    if config.mode == "cls_var":
        hists["weight_my_sf_unc_up"] = hist_(
            data=data[signal_idx, :, config.cls_var_idx],
            weights=data[signal_idx, :, w_sf_idx_up],
        )
        hists["weight_my_sf_unc_u"] = hist_(
            data=data[signal_idx, :, config.cls_var_idx],
            weights=data[signal_idx, :, w_sf_idx_down],
        )
    else:
        hists["weight_my_sf_unc_up"] = hist_(
            data=nn_output[signal_idx, :],
            weights=data[signal_idx, :, w_sf_idx_up],
        )
        hists["weight_my_sf_unc_up"] = hist_(
            data=nn_output[signal_idx, :],
            weights=data[signal_idx, :, w_sf_idx_down],
        )

    for i, sample_sys in enumerate(config.sample_sys):
        # scale, including w2
        scaled_hist = hists_vector[i] * scale[i]
        hists[sample_sys] = jnp.where(scaled_hist > 0, scaled_hist, 0.001)

    # stat error
    for sample in config.samples:
        sample_NOSYS = sample + "_" + config.nominal
        sigma = jnp.sqrt(hists[sample_NOSYS + "_w2"])
        hists[sample_NOSYS + "_stat_up"] = hists[sample_NOSYS] + sigma
        hists[sample_NOSYS + "_stat_down"] = hists[sample_NOSYS] - sigma

    return hists
