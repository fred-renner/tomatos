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


def init_opt_pars(config, nn_pars):

    # build opt_pars
    opt_pars = {}
    opt_pars["nn"] = nn_pars
    opt_pars["bw"] = config.bw_init
    if config.include_bins:
        # exclude boundaries
        opt_pars["bins"] = config.bins[1:-1]

    for key in config.opt_cuts:
        var_idx = config.vars.index(key)
        config.opt_cuts[key]["idx"] = var_idx
        init = config.opt_cuts[key]["init"]
        init *= config.scaler_scale[var_idx]
        init += config.scaler_min[var_idx]
        opt_pars[key + "_cut"] = init

    return opt_pars


def train_init(config):
    # init nn and opt pars
    nn_model = tomatos.nn.NeuralNetwork(n_features=config.nn_inputs_idx_end)
    # split model into parameters to optimize and the nn architecture
    nn_pars, nn_arch = eqx.partition(nn_model, eqx.is_array)
    config.nn_arch = nn_arch

    # get preprocess md
    with open(config.preprocess_md_file_path, "r") as json_file:
        config.preprocess_md = json.load(json_file)
    # for unscaling of vars
    config.scaler_scale = np.array(config.preprocess_md["scaler_scale"])
    config.scaler_min = np.array(config.preprocess_md["scaler_min"])

    opt_pars = init_opt_pars(config, nn_pars)

    # batcher
    batch = {}
    for split in ["train", "valid", "test"]:
        batch[split] = tomatos.batcher.get_generator(config, split)

    # solver

    solver = tomatos.solver.setup(config, opt_pars)
    train_data, train_sf = next(batch["train"])
    state = solver.init_state(
        opt_pars,
        data=train_data,
        config=config,
        scale=train_sf,
    )

    best_test_loss = np.inf

    return solver, state, opt_pars, batch, best_test_loss


def evaluate_losses(opt_pars, config, batch):
    """Evaluates validation and test losses."""
    valid_data, valid_sf = next(batch["valid"])
    valid_loss, valid_hists = tomatos.pipeline.loss_fn(
        opt_pars, valid_data, config, valid_sf, validate_only=True
    )

    test_data, test_sf = next(batch["test"])
    test_loss, test_hists = tomatos.pipeline.loss_fn(
        opt_pars, test_data, config, test_sf, validate_only=True
    )

    return valid_loss, valid_hists, test_loss, test_hists


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
        metrics[h_key] = h
        metrics[h_key + "_test"] = h

        # nominal hists
    logging.info("Nominal Hists:")
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
    logging.info("(Nominal hist) / (Sharp hist) Ratios:")
    for (h_key, h), (_, sharp_h) in zip(hists.items(), sharp_hists.items()):
        if config.nominal in h_key and not "STAT" in h_key:
            # hist approx ratio
            metrics[h_key + "_sharp"] = h
            logging.info(f"{h_key.ljust(25)}:  {h/sharp_h}")


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
    tomatos.utils.write_metrics(config, metrics, init=(i == 0))


def run(config):

    solver, state, opt_pars, batch, best_test_loss = train_init(config)

    metrics = {}
    # this holds optimization params like cuts for epochs used for deployment
    infer_metrics = {}
    # one step is one batch (not epoch)
    for i in alive_it(range(config.num_steps)):
        start = perf_counter()
        logging.info(f"step {i}: loss={config.objective}")

        # this holds optimization params like cuts per batch, for deployment
        infer_metrics_i = {}

        # this has to be here
        # since the optaxsolver holds step i-1, train evaluation is expensive
        valid_loss, valid_hists, test_loss, test_hists = evaluate_losses(
            opt_pars, config, batch
        )
        metrics["train_loss"] = state.value
        metrics["valid_loss"] = valid_loss
        metrics["test_loss"] = test_loss

        # gradient update
        train_data, train_sf = next(batch["train"])
        opt_pars, state = solver.update(
            opt_pars,
            state,
            data=train_data,
            config=config,
            scale=train_sf,
        )
        # apply limitations
        opt_pars = tomatos.constraints.opt_pars(config, opt_pars)

        ###### excessive logging, turn off as you please

        hists = state.aux
        logging.info(f"bw: {opt_pars['bw']}")
        # logging
        bins = (
            np.array([0, *opt_pars["bins"], 1]) if config.include_bins else config.bins
        )
        log_bins(config, metrics, bins, infer_metrics_i)
        log_hists(config, metrics, test_hists, hists)
        log_kde(config, metrics, opt_pars, train_data, train_sf, hists, bins)

        if "cls" in config.objective:
            log_cuts(config, opt_pars, metrics, infer_metrics_i)
            log_sharp_hists(opt_pars, train_data, config, train_sf, hists, metrics)

        save_model(
            i,
            test_loss,
            best_test_loss,
            config,
            opt_pars,
            metrics,
            infer_metrics,
            infer_metrics_i,
        )

        if test_loss < best_test_loss:
            best_test_loss = test_loss

        end = perf_counter()
        logging.info(f"update took {end-start:.4f}s")
        logging.info("\n")
        # if you want to run locally we need to clear the compilation caches
        # otherwise memory explodes, need to see how it scales on gpu
        tomatos.utils.clear_caches()

    return
