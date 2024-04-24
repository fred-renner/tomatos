# copied here from the neos and relaxed package to keep track of changes
# from __future__ import annotations

from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import vector
from sklearn.preprocessing import MinMaxScaler
import numpy as np

JAX_CHECK_TRACER_LEAKS = True
jnp.set_printoptions(precision=5)

Array = jnp.ndarray
import relaxed


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
    #
    weights = {k: data[k][:, 1, 0] for k in data}
    # it prints 0. but they are not

    # get signal
    og_values = jnp.copy(values["NOSYS"])

    # taken from sklearn inverse_transform
    og_values -= config.scaler_min
    og_values /= config.scaler_scale

    # # this indexing is horrible, but for now
    # Calculate the components for both jets
    # Convert pt, eta, phi, energy to px, py, pz, E for vector operations
    # https://root.cern.ch/doc/master/GenVector_2PtEtaPhiE4D_8h_source.html
    def pt_eta_phi_E_to_px_py_pz(pt, eta, phi, energy):
        px = pt * jnp.cos(phi)
        py = pt * jnp.sin(phi)
        pz = pt * jnp.sinh(eta)
        return px, py, pz, energy

    px1, py1, pz1, E1 = pt_eta_phi_E_to_px_py_pz(
        og_values[:, 0], og_values[:, 1], og_values[:, 2], og_values[:, 3]
    )
    px2, py2, pz2, E2 = pt_eta_phi_E_to_px_py_pz(
        og_values[:, 4], og_values[:, 5], og_values[:, 6], og_values[:, 7]
    )

    # Compute invariant mass and eta difference for the jet pairs
    m_jj_squared = (
        (E1 + E2) ** 2 - (px1 + px2) ** 2 - (py1 + py2) ** 2 - (pz1 + pz2) ** 2
    )
    m_jj = jnp.sqrt(
        jnp.maximum(0.0, m_jj_squared)
    )  # Ensure the argument of sqrt is non-negative, adjust units if necessary
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

    print("unscaled vbf: ", invert_min_max_scale(vbf_cut, jnp.min(m_jj), jnp.max(m_jj)))
    # weights = {k: w * m_jj_cut_w * eta_cut_w for k, w in weights.items()}

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
    hists["NOSYS_stat_up"] = hists["NOSYS"] + NOSYS_stat_err
    hists["NOSYS_stat_down"] = hists["NOSYS"] - NOSYS_stat_err
    hists["bkg_stat_up"] = hists["bkg"] + bkg_stat_err
    hists["bkg_stat_down"] = hists["bkg"] - bkg_stat_err

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
