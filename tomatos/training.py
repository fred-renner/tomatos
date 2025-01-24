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
import tomatos.optimizer
import tomatos.initialize
import equinox as eqx
import tomatos.batcher
import tomatos.constraints
import copy
import json


def run(config):

    batch = {}
    for split in ["train", "valid", "test"]:
        batch[split] = tomatos.batcher.get_generator(config, split)

    opt_pars = config.init_pars
    optimizer = tomatos.optimizer.setup(config, opt_pars)

    best_opt_pars = opt_pars
    best_test_loss = 999

    # log
    metrics = {
        k: []
        for k in [
            "train_loss",
            "valid_loss",
            "test_loss",
            "bins",
            "bw",
            *config.opt_cuts.keys(),
        ]
    }

    # this holds optimization params like cuts for epochs, for deployment
    infer_metrics = {}

    # one step is one batch (not necessarily epoch)
    # gradient_accumulation interesting for more stable generalization in the
    # future as it reduces noise in the gradient updates
    # https://optax.readthedocs.io/en/latest/_collections/examples/gradient_accumulation.html

    for i in range(config.num_steps):

        start = perf_counter()
        logging.info(f"step {i}: loss={config.objective}")

        train_data, train_sf = next(batch["train"])
        if i == 0:
            state = optimizer.init_state(
                opt_pars,
                data=train_data,
                config=config,
                scale=train_sf,
            )

            hists = state.aux
            hists = tomatos.utils.filter_hists(config, hists)

        for k in hists.keys():
            metrics[k] = []
            metrics[k + "_test"] = []

        # Evaluate losses
        valid_data, valid_sf = next(batch["valid"])
        valid_loss, valid_hists = tomatos.pipeline.loss_fn(
            opt_pars,
            data=valid_data,
            config=config,
            scale=valid_sf,
            validate_only=True,
        )
        test_data, test_sf = next(batch["test"])
        test_loss, test_hists = tomatos.pipeline.loss_fn(
            opt_pars,
            data=test_data,
            config=config,
            scale=test_sf,
            validate_only=True,
        )

        # this has to be here
        # since the optaxsolver of step i-1, evaluation is expensive
        metrics["train_loss"].append(state.value)
        metrics["valid_loss"].append(valid_loss)
        metrics["test_loss"].append(test_loss)

        opt_pars, state = optimizer.update(
            opt_pars,
            state,
            data=train_data,
            config=config,
            scale=train_sf,
        )

        opt_pars = tomatos.constraints.opt_pars(config, opt_pars)

        ########## extra calc and logging #######
        # this is computationally chep
        infer_metrics_i = {}
        hists = state.aux
        hists = tomatos.utils.filter_hists(config, hists)
        test_hists = tomatos.utils.filter_hists(config, test_hists)
        bins = (
            np.array([0, *opt_pars["bins"], 1]) if config.include_bins else config.bins
        )

        # hists
        for hist in hists.keys():
            metrics[hist].append(hists[hist])
            metrics[hist + "_test"].append(test_hists[hist])

        # kde
        kde = sample_kde_distribution(
            config=config,
            opt_pars=opt_pars,
            data=train_data,
            scale=train_sf,
            hists=hists,
        )
        for key, h in kde.items():
            kde_key = "kde_" + key
            if kde_key not in metrics:
                metrics[kde_key] = []
            metrics[kde_key].append(h)

        if config.include_bins:
            actual_bins = (
                tomatos.utils.inverse_min_max_scale(
                    config, np.copy(bins), config.cls_var_idx
                )
                if config.objective == "cls_var"
                else bins
            )
            logging.info((f"next bin edges: {actual_bins}"))
            metrics["bins"].append(actual_bins)
            infer_metrics_i["bins"] = actual_bins

        if "cls" in config.objective:
            # cuts
            for var, cut_dict in config.opt_cuts.items():
                var_cut = f"{var}_cut"
                opt_cut = tomatos.utils.inverse_min_max_scale(
                    config, opt_pars[var_cut], cut_dict["idx"]
                )
                logging.info(f"{var_cut}: {opt_cut}")
                if var_cut not in metrics:
                    metrics[var_cut] = []
                metrics[var_cut].append(opt_cut)
                infer_metrics_i[var_cut] = opt_cut

            # sharp evaluation train data hists
            sharp_hists = tomatos.pipeline.make_hists(
                opt_pars,
                train_data,
                config,
                train_sf,
                validate_only=True,
            )
            sharp_hists = tomatos.utils.filter_hists(config, sharp_hists)

            for (h_key, h), (_, sharp_h) in zip(hists.items(), sharp_hists.items()):
                if (
                    config.nominal in key
                    and config.fit_region in key
                    and not "STAT" in key
                ):
                    # hist approx ratio
                    logging.info(f"Sharp hist ratio: {h/sharp_h}")

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
            # save metrics
            with open(config.metrics_file_path, "w") as file:
                json.dump(tomatos.utils.to_python_lists(metrics), file)
            with open(config.infer_metrics_file_path, "w") as file:
                json.dump(tomatos.utils.to_python_lists(infer_metrics), file)

        # nominal hists
        for key, h in hists.items():
            if config.nominal in key and config.fit_region in key and not "STAT" in key:
                logging.info(f"{key.ljust(30)}: {h}")
        logging.info(f"bw: {opt_pars['bw']}")

        end = perf_counter()
        logging.info(f"update took {end-start:.4f}s\n")

        # otherwise memore explodes in this jax version
        # without it, the test is 3 times faster, however this is very
        # memory expensive
        tomatos.utils.clear_caches()

    return metrics, infer_metrics


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
):
    # get kde distribution by sampling with a many bin histogram
    # enough to get kde only from the nominal ones
    sample_indices = np.arange(len(config.samples))
    nominal_data = data[sample_indices, :, :]
    kde_bins = np.linspace(0, 1, config.kde_sampling)
    kde_config = copy.deepcopy(config)
    kde_config.bins = kde_bins
    # to also collect the background estimate
    kde_dist = tomatos.pipeline.make_hists(
        opt_pars,
        nominal_data,
        kde_config,
        scale,
    )
    kde_dist = tomatos.utils.filter_hists(config, kde_dist)
    kde_dist = {
        h_key: rescale_kde(config, hists[h_key], kde_dist[h_key], opt_pars["bins"])
        for h_key in kde_dist
    }

    return kde_dist
