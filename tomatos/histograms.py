from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import relaxed

import tomatos.select



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


@partial(jax.jit, static_argnames=["nn_arch", "nn_inputs_idx_end"])
def get_nn_output(
    pars,
    data,
    nn_arch,
    nn_inputs_idx_end,
):
    nn = eqx.combine(pars["nn"], nn_arch)

    # this also illustrates nicely how jax.vmap works
    def predict_sample(i):
        # Forward pass for one sample
        sample_data = data[i, :, :nn_inputs_idx_end]
        # vmap all events in batch for this sample
        sample_nn_output = jax.vmap(nn)(sample_data)
        # flatten output of [[out_1], [out_2],...]
        return sample_nn_output.ravel()

    nn_output = jax.vmap(predict_sample)(jnp.arange(data.shape[0]))
    return nn_output


# jitting however not of much help here
@partial(jax.jit, static_argnames=["objective", "cls_var_idx", "w2"])
def compute_hist_wrapper(
    i,
    objective,
    data,
    cls_var_idx,
    nn_output,
    weights,
    scale,
    bw,
    bins,
    w2=False,
):
    # lots of args due to the pure function paradigm
    # these ifs only work because of static_argnames
    if objective == "cls_var":
        sample_data = data[i, :, cls_var_idx]
    elif objective == "cls_nn":
        sample_data = nn_output[i, :]

    sample_weights = weights[i, :]

    if w2:
        sample_weights = jnp.power(sample_weights, 2)

    # Scale works also as an estimate for w2
    return (
        hist(
            data=sample_data,
            weights=sample_weights,
            bandwidth=bw,
            bins=bins,
        )
        * scale[i]
    )


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

    # this will hold: hists[sample][sys]
    hists = {sample: {} for sample in config.samples}

    # get nn output
    if config.objective == "cls_nn":
        nn_output = get_nn_output(
            pars,
            data,
            config.nn_arch,
            config.nn_inputs_idx_end,
        )
    else:
        nn_output = None

    compute_hist = partial(
        compute_hist_wrapper,
        objective=config.objective,
        data=data,
        cls_var_idx=config.cls_var_idx,
        nn_output=nn_output,
        scale=scale,
        bw=bw,
        bins=bins,
    )
    # calc all hists for all samples
    hists_vector = jax.vmap(lambda i: compute_hist(i, weights=sel_weights["nominal"]))(
        jnp.arange(len(config.sample_sys))
    )

    # this is the sequential version
    # hists_vector = []
    # for i in range(len(config.sample_sys)):
    #     hist = compute_hist(i, weights=sel_weights["nominal"])
    #     hists_vector.append(hist)

    for sample_sys, h in zip(config.sample_sys, hists_vector):
        sample, sys = config.sample_sys_dict[sample_sys]
        hists[sample][sys] = h

    # Compute w2 histograms only for the ones we need
    # going over len(config.samples) works because NOSYS are the first ones per
    # sample, see config
    hists_nominal_w2_vector = jax.vmap(
        lambda i: compute_hist(i, weights=sel_weights["nominal"], w2=True)
    )(jnp.arange(len(config.samples)))

    # workaround stat up and down hists, see workspace
    for sample, h_w2 in zip(config.samples, hists_nominal_w2_vector):
        sigma = jnp.sqrt(h_w2)
        hists[sample]["STAT_1UP"] = hists[sample][config.nominal] + sigma
        hists[sample]["STAT_1DOWN"] = hists[sample][config.nominal] - sigma

    # weight based systematics, e.g. scale factors or modelling
    for sys, sys_dict in tomatos.select.WEIGHT_VARIATIONS.items():
        i = config.sample_sys.index(f"{sys_dict['sample']}_{config.nominal}")
        for direction in ["up", "down"]:
            hists[sys_dict["sample"]][f"{sys}_1{direction.upper()}"] = compute_hist(
                i, weights=sel_weights[f"{sys}_{direction}"]
            )

    return hists
