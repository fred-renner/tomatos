import copy
import json
import logging
import os
import sys
from functools import partial
from time import perf_counter

import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from alive_progress import alive_it
from jaxopt import OptaxSolver

import tomatos.batcher
import tomatos.constraints
import tomatos.histograms
import tomatos.nn
import tomatos.pipeline
import tomatos.solver
import tomatos.utils
import tomatos.workspace


def log_bw(metrics, opt_pars):
    metrics["bw"] = opt_pars["bw"]
    logging.info(f"bw: {opt_pars['bw']}")


def log_cuts(config, opt_pars, metrics, infer_metrics_i):
    for var, cut_dict in config.opt_cuts.items():
        cut_var = f"cut_{var}"
        opt_cut = tomatos.utils.inverse_min_max_scale(
            config, opt_pars[cut_var], cut_dict["idx"]
        )
        logging.info(f"{cut_var}: {opt_cut}")
        if cut_var not in metrics:
            metrics[cut_var] = []
        metrics[cut_var] = opt_cut
        infer_metrics_i[cut_var] = opt_cut


def log_kde(config, metrics, opt_pars, train_data, train_sf, hists, bins, kde_hists_fn):
    kde = sample_kde_distribution(
        config=config,
        opt_pars=opt_pars,
        data=train_data,
        scale=train_sf,
        hists=hists,
        bins=bins,
        kde_hists_fn=kde_hists_fn,
    )
    for key, h in kde.items():
        metrics["kde_" + key] = h


def log_hists(config, metrics, test_hists, hists):
    for h_key, h in hists.items():
        metrics["h_" + h_key] = h
    for h_key, h in test_hists.items():
        metrics["h_" + h_key + "_test"] = h

        # nominal hists
    logging.info("--- Nominal (binned KDE) ---")
    for key, h in hists.items():
        if config.nominal in key and not "STAT" in key:
            logging.info(f"{key.ljust(25)}: {h}")


def log_bins(config, metrics, bins, infer_metrics_i):
    if config.objective == "cls_var":
        scaled_bins = tomatos.utils.inverse_min_max_scale(
            config, np.copy(bins), config.cls_var_idx
        )
    else:
        scaled_bins = bins
    metrics["bins"] = scaled_bins
    infer_metrics_i["bins"] = scaled_bins

    if config.include_bins:
        logging.info(f"{'bins'.ljust(25)}: {scaled_bins}")

    return bins


def rescale_kde(config, hist, kde, bins):

    # need to upscale sampled kde hist as it is a very fine binned version of
    # the histogram, use the largest bin for it,
    # NB: this is an approximation, only works properly for the largest bin

    # use the largest bin of a binned kde hist
    max_bin_idx = np.argmax(hist)
    max_bin_edges = np.array([bins[max_bin_idx], bins[max_bin_idx + 1]])
    # integrate histogram for this bin
    hist_x_width = np.diff(max_bin_edges)
    hist_height = hist[max_bin_idx]
    area_hist = hist_x_width * hist_height

    # integrate kde for this bin
    kde_indices = (max_bin_edges * config.kde_sampling).astype(int)
    kde_heights = kde[kde_indices[0] : kde_indices[1]]
    kde_dx = 1 / config.kde_sampling
    area_kde = np.sum(kde_dx * kde_heights)

    scale_factor = area_hist / area_kde
    kde_scaled = kde * scale_factor

    return kde_scaled


def sample_kde_distribution(
    config,
    opt_pars,
    data,
    scale,
    hists,
    bins,
    kde_hists_fn,
):
    # get kde distribution by sampling with a many bin histogram
    # enough to get kde only from the nominal ones
    sample_indices = np.arange(len(config.samples))
    nominal_data = data[sample_indices, :, :]

    # to also collect the background estimate
    kde_dist = kde_hists_fn(opt_pars, nominal_data, scale)

    kde_dist = {
        h_key: rescale_kde(config, hists[h_key], kde_dist[h_key], bins)
        for h_key in kde_dist
    }

    return kde_dist


def jit_kde_hists(config):
    # make a custom config, many bin hists sample the kde
    kde_config = copy.deepcopy(config)
    kde_config.bins = config.kde_bins
    kde_config.include_bins = False
    return jax.jit(
        lambda pars, data, scale: tomatos.pipeline.make_hists(
            pars, data, kde_config, scale, filter_return_hists=True
        )
    )


def log_sharp_hists(
    opt_pars,
    train_data,
    config,
    train_sf,
    hists,
    metrics,
    sharp_hists_fn,
):
    # actually might be enough to compare to test hists, depends a bit on the
    # uncertainty behavior...

    # sharp evaluation train data hists
    sharp_hists = sharp_hists_fn(opt_pars, train_data, train_sf)
    logging.info("--- Nominal (binned KDE) / (Sharp hist) ---")
    for (h_key, h), (_, sharp_h) in zip(hists.items(), sharp_hists.items()):
        if config.nominal in h_key and not "STAT" in h_key:
            # hist approx ratio
            metrics["h_" + h_key + "_sharp"] = sharp_h
            logging.info(f"{h_key.ljust(25)}: {h/sharp_h}")


def do_metrics_exist(config):
    if os.path.exists(config.metrics_file_path) and not config.debug:
        user_input = input(
            f"{config.metrics_file_path} exists. \n"
            "Seems like you trained this already \n"
            "Overwrite and Proceed? (y/n):"
        )
        if user_input.lower() != "y":
            logging.info("OK, Bye!")
            sys.exit(1)


def init_metrics(config, metrics):
    with h5py.File(config.metrics_file_path, "w") as h5f:
        for key, value in metrics.items():
            if isinstance(value, float):
                shape = (config.num_steps,)
                dtype = "f4"
            elif isinstance(value, int):
                shape = (config.num_steps,)
                dtype = "i"
            elif isinstance(value, list):
                shape = (config.num_steps, len(value))
                dtype = "f4"
            h5f.create_dataset(key, shape=shape, dtype=dtype, compression="gzip")
            h5f[key][0] = value


def write_metrics(config, metrics, i):
    metrics = tomatos.utils.to_python_lists(metrics)
    if i == 0:
        init_metrics(config, metrics)
    else:
        with h5py.File(config.metrics_file_path, "r+") as h5f:
            for key, value in metrics.items():
                dataset = h5f[key]
                if isinstance(value, (float, int)):
                    dataset[i] = value
                else:
                    dataset[i, :] = value


def save_model(
    i,
    test_loss,
    best_test_loss,
    config,
    opt_pars,
    infer_metrics,
    infer_metrics_i,
):
    # pick best training and save
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        infer_metrics["epoch_best"] = infer_metrics_i
        infer_metrics["epoch_best"]["epoch"] = i
        model = eqx.combine(opt_pars["nn"], config.nn_arch)
        eqx.tree_serialise_leaves(config.model_path + "epoch_best.eqx", model)
    # save every 10th model to file
    if i % 10 == 0 and i != 0:
        epoch_name = f"epoch_{i:005d}"
        infer_metrics[epoch_name] = infer_metrics_i
        model = eqx.combine(opt_pars["nn"], config.nn_arch)
        eqx.tree_serialise_leaves(config.model_path + epoch_name + ".eqx", model)

    if i == (config.num_steps - 1):
        # save infer metrics
        with open(config.infer_metrics_file_path, "w") as file:
            json.dump(tomatos.utils.to_python_lists(infer_metrics), file)
