import logging
import sys
from functools import partial
from time import perf_counter
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import optax
import jax
from jaxopt import OptaxSolver

import tomatos.histograms
import tomatos.utils
import tomatos.workspace
import tomatos.pipeline
import tomatos.solver
import tomatos.initialize
import equinox as eqx
import tomatos.batcher
import tomatos.constraints
import copy
import json
import tomatos.nn
from alive_progress import alive_it
import h5py
import os


def binary_cross_entropy(preds, labels):
    epsilon = 1e-15  # To avoid log(0)
    preds = jnp.clip(preds, epsilon, 1 - epsilon)
    return -jnp.mean(labels * jnp.log(preds) + (1 - labels) * jnp.log(1 - preds))


def bce(ones, zeros):
    labels = jnp.concatenate([jnp.ones_like(ones), jnp.zeros_like(zeros)])
    preds = jnp.concatenate((ones, zeros))
    return binary_cross_entropy(preds, labels)


def log_cuts(config, opt_pars, metrics, infer_metrics_i):
    for var, cut_dict in config.opt_cuts.items():
        var_cut = f"{var}_cut"
        opt_cut = tomatos.utils.inverse_min_max_scale(
            config, opt_pars[var_cut], cut_dict["idx"]
        )
        logging.info(f"{var_cut}: {opt_cut}")
        if var_cut not in metrics:
            metrics[var_cut] = []
        metrics[var_cut] = opt_cut
        infer_metrics_i[var_cut] = opt_cut


def log_kde(config, metrics, opt_pars, train_data, train_sf, hists, bins):
    kde = sample_kde_distribution(
        config=config,
        opt_pars=opt_pars,
        data=train_data,
        scale=train_sf,
        hists=hists,
        bins=bins,
    )
    for key, h in kde.items():
        metrics["kde_" + key] = h


def log_hists(config, metrics, test_hists, hists):
    for h_key, h in hists.items():
        metrics["h_" + h_key] = h
        metrics["h_" + h_key + "_test"] = h

        # nominal hists
    logging.info("--- Nominal Hists ---")
    for key, h in hists.items():
        if config.nominal in key and not "STAT" in key:
            logging.info(f"{key.ljust(25)}: {h}")


def log_bins(config, metrics, bins, infer_metrics_i):
    scaled_bins = (
        tomatos.utils.inverse_min_max_scale(config, np.copy(bins), config.cls_var_idx)
        if config.objective == "cls_var"
        else bins
    )
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
):
    # get kde distribution by sampling with a many bin histogram
    # enough to get kde only from the nominal ones
    sample_indices = np.arange(len(config.samples))
    nominal_data = data[sample_indices, :, :]
    kde_config = copy.deepcopy(config)
    kde_bins = np.linspace(0, 1, config.kde_sampling)
    kde_config.bins = kde_bins
    kde_config.include_bins = False

    # to also collect the background estimate
    kde_dist = tomatos.pipeline.make_hists(
        opt_pars,
        nominal_data,
        kde_config,
        scale,
        filter_hists=True,
    )

    kde_dist = {
        h_key: rescale_kde(config, hists[h_key], kde_dist[h_key], bins)
        for h_key in kde_dist
    }

    return kde_dist


def log_sharp_hists(
    opt_pars,
    train_data,
    config,
    train_sf,
    hists,
    metrics,
):
    # sharp evaluation train data hists
    sharp_hists = tomatos.pipeline.make_hists(
        opt_pars,
        train_data,
        config,
        train_sf,
        validate_only=True,  # sharp hists
        filter_hists=True,
    )
    logging.info("--- (Nominal hist) / (Sharp hist) Ratios ---")
    for (h_key, h), (_, sharp_h) in zip(hists.items(), sharp_hists.items()):
        if config.nominal in h_key and not "STAT" in h_key:
            # hist approx ratio
            metrics["h_" + h_key + "_sharp"] = sharp_h
            logging.info(f"{h_key.ljust(25)}:  {h/sharp_h}")


def do_metrics_exist(config):
    if os.path.exists(config.metrics_file_path) and not config.debug:
        user_input = input(
            f"{config.metrics_file_path} exists. \n"
            f"Did you train this already? \n"
            f"Overwrite and Proceed? (y/n):"
        )
        if user_input.lower() != "y":
            logging.info("OK, Bye!")
            sys.exit(1)


def init_metrics(config, metrics):
    with h5py.File(config.metrics_file_path, "w") as h5f:
        for key, value in metrics.items():
            if isinstance(value, float):
                shape = (config.num_steps,)
            elif isinstance(value, list):
                shape = (config.num_steps, len(value))
            h5f.create_dataset(key, shape=shape, dtype="f4", compression="gzip")
            h5f[key][0] = value


def write_metrics(config, metrics, i):
    metrics = tomatos.utils.to_python_lists(metrics)
    if i == 0:
        init_metrics(config, metrics)
    else:
        with h5py.File(config.metrics_file_path, "r+") as h5f:
            for key, value in metrics.items():
                dataset = h5f[key]
                if isinstance(value, float):
                    dataset[i] = value
                else:
                    dataset[i, :] = value


def save_model(
    i,
    test_loss,
    best_test_loss,
    config,
    opt_pars,
    metrics,
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
