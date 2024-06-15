# copied here from the neos and relaxed package to keep track of changes
# from __future__ import annotations

from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import jax.scipy as jsp

import logging

JAX_CHECK_TRACER_LEAKS = True


Array = jnp.ndarray
import relaxed

w_CR = 0.0036312547281962607


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

    bins = jnp.array([-jnp.inf, *bins, jnp.inf]) if reflect_infinities else bins

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
    cdf = cdf * (weights**2)

    # sum kde contributions in each bin
    counts = (cdf[1:, :] - cdf[:-1, :]).sum(axis=1)

    return counts


def get_vbf_cut_weights(config, values, vbf_cut, eta_cut):
    # get signal
    og_values = jnp.copy(values)

    # taken from sklearn inverse_transform
    og_values -= config.scaler_min
    og_values /= config.scaler_scale

    # # this indexing is horrible, but for now
    # Calculate the components for both jets
    # the issue is, if we give it at the beginning it becomes an input
    # variable... would this be bad?
    # Convert pt, eta, phi, energy to px, py, pz, E for vector operations
    # https://root.cern.ch/doc/master/GenVector_2PtEtaPhiE4D_8h_source.html
    def pt_eta_phi_E_to_px_py_pz(pt, eta, phi, energy):
        px = pt * jnp.cos(phi)
        py = pt * jnp.sin(phi)
        pz = pt * jnp.sinh(eta)
        return px, py, pz, energy

    px1, py1, pz1, E1 = pt_eta_phi_E_to_px_py_pz(
        og_values[:, 0],
        og_values[:, 1],
        og_values[:, 2],
        og_values[:, 3],
    )
    px2, py2, pz2, E2 = pt_eta_phi_E_to_px_py_pz(
        og_values[:, 4],
        og_values[:, 5],
        og_values[:, 6],
        og_values[:, 7],
    )

    # Compute invariant mass and eta difference for the jet pairs
    m_jj_squared = (
        (E1 + E2) ** 2 - (px1 + px2) ** 2 - (py1 + py2) ** 2 - (pz1 + pz2) ** 2
    )
    m_jj = jnp.sqrt(m_jj_squared)

    eta_difference = jnp.abs(og_values[:, 1] - og_values[:, 5])

    # Apply cuts
    # need to scale otherwise optimization not working
    def min_max_scale(x):
        return (x - jnp.min(x)) / (jnp.max(x) - jnp.min(x))

    m_jj_cut_w = relaxed.cut(min_max_scale(m_jj), vbf_cut, slope=1000, keep="above")
    eta_cut_w = relaxed.cut(
        min_max_scale(eta_difference), eta_cut, slope=100, keep="above"
    )

    def invert_min_max_scale(y, min_x, max_x):
        return y * (max_x - min_x) + min_x

    optimized_m_jj = invert_min_max_scale(vbf_cut, jnp.min(m_jj), jnp.max(m_jj))
    optimized_delta_eta_jj = invert_min_max_scale(
        eta_cut, jnp.min(eta_difference), jnp.max(eta_difference)
    )
    logging.info(f"optimized m_jj: { optimized_m_jj}")
    logging.info(f"optimized delta_eta_jj: { optimized_delta_eta_jj}")
    weight = m_jj_cut_w * eta_cut_w

    return weight, optimized_m_jj, optimized_delta_eta_jj


def get_vbf_cut_weights_unscaled(m_jj, delta_eta_jj, vbf_cut, eta_cut):
    m_jj_cut_w = relaxed.cut(m_jj, vbf_cut, slope=1e-4, keep="above")
    eta_cut_w = relaxed.cut(delta_eta_jj, eta_cut, slope=1, keep="above")
    weight = m_jj_cut_w * eta_cut_w
    return weight


def hists_from_nn(
    config,
    nn_pars: Array,
    data: dict[str, Array],
    nn: Callable,
    bandwidth: float,
    bins: Array,
    vbf_cut: Array,
    eta_cut: Array,
) -> dict[str, Array]:
    """Function that takes in data + analysis config parameters, and constructs yields."""

    # k index is sample index
    values = {k: data[k][:, 0, :] for k in data}
    # it prints 0. but they are not
    weights = {k: data[k][:, 1, 0] for k in data}

    cutted_weight, optimized_m_jj, optimized_delta_eta_jj = get_vbf_cut_weights(
        config, values["NOSYS"], vbf_cut, eta_cut
    )
    if config.objective == "cls":
        weights = {k: w * cutted_weight for k, w in weights.items()}

    # define our histogram-maker with some hyperparameters (bandwidth, binning)
    make_hist = partial(hist, bandwidth=bandwidth, bins=bins)

    # apply the neural network to each data sample, and keep track of the
    # sample names in a dict
    nn_apply = partial(nn, nn_pars)
    nn_output = {k: jax.vmap(nn_apply)(values[k]).ravel() for k in values}

    hists = {
        k: make_hist(data=nn_output[k], weights=weights[k])
        for k, v in nn_output.items()
    }

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

    # add VR and CR from data für bkg estimate
    estimate_regions = [
        "CR_xbb_1",
        "CR_xbb_2",
        "VR_xbb_1",
        "VR_xbb_2",
    ]

    # need another bandwidth hist since we want sharp hists
    sharp_hist = partial(hist, bandwidth=1e-8, bins=bins)

    # apply optimized m_jj and eta_jj cut on the estimate_regions
    for reg in estimate_regions:
        cutted_weight_estimate = get_vbf_cut_weights_unscaled(
            config.bkg_estimate[f"m_jj_{reg}"],
            config.bkg_estimate[f"eta_jj_{reg}"],
            vbf_cut=optimized_m_jj,
            eta_cut=optimized_delta_eta_jj,
        )
        bkg_estimate_data = config.bkg_estimate[reg]
        hists[f"bkg_{reg}"] = sharp_hist(
            data=jax.vmap(nn_apply)(bkg_estimate_data[:, 0, :]).ravel(),
            weights=cutted_weight_estimate,  # nominal weights are 1.0 since data
        )
        # print(hists[f"bkg_{reg}"])

    hists["NOSYS_stat_up"] = hists["NOSYS"] + NOSYS_stat_err
    hists["NOSYS_stat_down"] = hists["NOSYS"] - NOSYS_stat_err
    hists["bkg_stat_up"] = hists["bkg"] + bkg_stat_err
    hists["bkg_stat_down"] = hists["bkg"] - bkg_stat_err

    if config.objective == "bce":
        hists["bkg"] *= w_CR

    return hists


def hists_from_mhh(
    data: dict[str, Array],
    bins: Array,
    bandwidth: float,
    include_bins=False,
):
    values = {k: data[k][:, 0, :] for k in data}

    weights = {k: data[k][:, 1, 0] for k in data}

    if include_bins:
        bins = jnp.concatenate(
            (
                jnp.array([bins[0]]),
                jnp.where(bins[1:] > bins[:-1], bins[1:], bins[:-1] + 1e-4),
            ),
            axis=0,
        )

    make_hist = partial(hist, bandwidth=bandwidth, bins=bins)
    hists = {
        k: make_hist(data=values[k].ravel(), weights=weights[k])
        for k, v in data.items()
    }

    return hists
