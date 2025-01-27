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
import tomatos.train_utils


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
        opt_pars["cut_" + key] = init

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


def run(config):

    solver, state, opt_pars, batch, best_test_loss = train_init(config)

    metrics = {}
    # this holds optimization params like cuts for epochs used for deployment
    infer_metrics = {}
    # check so not mistakenly overwrite
    tomatos.train_utils.do_metrics_exist(config)

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

        ###### excessive logging, turn off as you please ######
        # expensive is only log_kde and log_sharp_hists

        hists = state.aux
        # logging
        bins = (
            np.array([0, *opt_pars["bins"], 1]) if config.include_bins else config.bins
        )
        tomatos.train_utils.log_hists(config, metrics, test_hists, hists)
        tomatos.train_utils.log_kde(
            config, metrics, opt_pars, train_data, train_sf, hists, bins
        )

        if "cls" in config.objective:
            tomatos.train_utils.log_sharp_hists(
                opt_pars, train_data, config, train_sf, hists, metrics
            )
            tomatos.train_utils.log_bins(config, metrics, bins, infer_metrics_i)
            tomatos.train_utils.log_cuts(config, opt_pars, metrics, infer_metrics_i)
            tomatos.train_utils.log_bw(metrics, opt_pars)
        tomatos.train_utils.save_model(
            i,
            test_loss,
            best_test_loss,
            config,
            opt_pars,
            infer_metrics,
            infer_metrics_i,
        )

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            metrics["best_test_batch"] = i

        tomatos.train_utils.write_metrics(config, metrics, i)

        # if you want to run locally we need to clear the compilation caches
        # otherwise memory explodes, will see how it scales on gpu
        tomatos.utils.clear_caches()

        end = perf_counter()
        logging.info(f"train loss: {state.value}")
        logging.info(f"test loss: {test_loss}")
        logging.info(f"update took {end-start:.4f}s")
        logging.info("\n")

    logging.info("Training Done!")
    return
