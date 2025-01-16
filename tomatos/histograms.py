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
    # bandwidth = bandwidth or events.shape[-1] ** -0.25  # Scott's rule
    # proof is for gaussian... and also wrong above, must be 3.49*sigma*n^(-1/3)
    # https://www.stat.cmu.edu/~rnugent/PCMI2016/papers/ScottBandwidth.pdf

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


# much easier to get w2sum from a dedicated jitted function
@jax.jit
def get_w2sum(
    data: Array,
    weights: Array,
    bins: Array,
    bandwidth: float,
) -> Array:
    """get w2 sum from Differentiable histogram, defined via a binned kernel density estimate (bKDE).

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
    cdf = cdf * jnp.power(weights, 2)

    # sum kde contributions in each bin
    counts = (cdf[1:, :] - cdf[:-1, :]).sum(axis=1)

    return counts


def get_hists(
    pars,
    data,
    config,
):

    if config.include_bins:
        bins = jnp.array([0, *pars["bins"], 1])
    else:
        bins = config.bins

    # values = {k: data[k][:, 0, :] for k in data}
    # weights = {k: data[k][:, 1, 0] * scale for k in data}

    # apply cuts to weights
    cut_weights = jnp.ones(data.shape[:2])
    # collect them over all cuts and apply to weights once
    for var, var_dict in config.opt_cuts.items():
        # get the sigmoid weights for var_cut
        cut_weights *= relaxed.cut(
            data[:, :, var_dict["idx"]],
            pars[var + "_cut"],
            slope=config.slope,
            keep=var_dict["keep"],
        )
    # apply to weights
    # inside a jitted function this is not a copy
    # https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ndarray.at.html
    data = data.at[:, :, config.weight_idx].multiply(cut_weights)

    hists = {}

    # make dedicated function, reduce compile
    if config.mode == "cls_var":
        for i, sample_sys in enumerate(config.sample_sys):
            hists[sample_sys] = hist(
                data=data[i, :, config.cls_var_idx],
                weights=data[i, :, config.weight_idx],
                bandwidth=pars["bw"],
                bins=bins,
            )

    else:

        # combine parameters and nn architecture
        nn = eqx.combine(pars["nn"], config.nn_arch)

        # the following is the same as, but faster
        # for i, sample_sys in enumerate(config.sample_sys):
        #     nn_prediction = jax.vmap(nn)(data[i, :, : config.nn_inputs_idx]).ravel()
        def process_sample(i):
            sample_data = data[i, :, : config.nn_inputs_idx]
            nn_outputs = jax.vmap(nn)(sample_data)
            return nn_outputs.ravel()

        nn_prediction = jax.vmap(process_sample)(jnp.arange(len(config.sample_sys)))

        hists = {
            k: make_hist(data=nn_output[k], weights=weights[k])
            for k, v in nn_output.items()
        }

    if config.do_stat_error:
        # calculate stat error
        NOSYS_stat_err = jnp.sqrt(
            get_w2sum(
                data=nn_output["NOSYS"],
                weights=weights["NOSYS"],
                bandwidth=bandwidth,
                bins=bins,
            )
        )
        bkg_stat_err = jnp.sqrt(
            get_w2sum(
                data=nn_output["bkg"],
                weights=weights["bkg"],
                bandwidth=bandwidth,
                bins=bins,
            )
        )

        hists["NOSYS_stat_up"] = hists["NOSYS"] + NOSYS_stat_err
        hists["NOSYS_stat_down"] = hists["NOSYS"] - NOSYS_stat_err
        hists["bkg_stat_up"] = hists["bkg"] + bkg_stat_err
        hists["bkg_stat_down"] = hists["bkg"] - bkg_stat_err

    return hists
