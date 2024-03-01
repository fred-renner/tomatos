import gc
import logging
import sys
from functools import partial
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pyhf
import relaxed
from jaxopt import OptaxSolver

import tomatos.histograms
import tomatos.pipeline

Array = jnp.ndarray
np.set_printoptions(precision=2)


# clear caches each update otherwise memory explodes
# https://github.com/google/jax/issues/10828
# doubles computation time, still filling up memory but much slower
def clear_caches():
    # process = psutil.Process()
    # if process.memory_info().vms > 4 * 2**30:  # >4GB memory usage
    for module_name, module in sys.modules.items():
        if module_name.startswith("jax"):
            for obj_name in dir(module):
                obj = getattr(module, obj_name)
                if hasattr(obj, "cache_clear"):
                    obj.cache_clear()
    gc.collect()


def run(
    config,
    valid,
    test,
    batch_iterator,
    init_pars,
    nn,
) -> tuple[Array, dict[str, list]]:
    # even though config is passed, need to keep redundant args here as this
    # function is jitted
    loss = partial(
        tomatos.pipeline.pipeline,
        nn=nn,
        sample_names=config.data_types,
        include_bins=config.include_bins,
        do_m_hh=config.do_m_hh,
        loss_type=config.objective,
        config=config,
        bandwidth=config.bandwidth,
        do_systematics=config.do_systematics,
        do_stat_error=config.do_stat_error,
    )

    solver = OptaxSolver(loss, opt=optax.adam(config.lr), has_aux=True, jit=True)

    pyhf.set_backend("jax", default=True, precision="64b")

    params = init_pars
    best_params = init_pars
    best_sig = 999

    metrics = {
        k: []
        for k in [
            "cls_train",
            "cls_valid",
            "cls_test",
            "discovery_train",
            "discovery_valid",
            "discovery_test",
            "bce_train",
            "bce_valid",
            "bce_test",
            "Z_A",
            "bins",
            "hist_sig",
            "hist_bkg",
        ]
    }

    # one step is one batch (not necessarily epoch)
    for i in range(config.num_steps):
        logging.info(f"step {i}: loss={config.objective}")
        train, batch_num, num_batches = next(batch_iterator)
        # initialize with or without binning
        if i == 0:
            if config.include_bins:
                init_pars["bins"] = config.bins
            else:
                init_pars.pop("bins", None)

            logging.info(init_pars)
            state = solver.init_state(
                init_pars,
                data=train,
            )

        start = perf_counter()
        params, state = solver.update(
            params,
            state,
            data=train,
        )
        hists = state.aux
        end = perf_counter()
        logging.info(f"update took {end-start:.4f}s")

        bins = np.array(params["bins"]) if "bins" in params else config.bins
        # save current bins
        if config.include_bins:
            logging.info((f"next bin edges: {bins}"))
            metrics["bins"].append(bins)

        logging.info((f"hist sig: {hists['NOSYS']}"))
        logging.info((f"hist bkg: {hists['bkg']}"))
        metrics["hist_sig"].append(hists["NOSYS"])
        metrics["hist_bkg"].append(hists["bkg"])

        # Evaluate losses.
        start = perf_counter()
        for loss_type in ["cls", "bce"]:  # , "bce"]:  # , "discovery", "bce"]:
            metrics[f"{loss_type}_valid"].append(
                evaluate_loss(loss, params, valid, loss_type)
            )
            metrics[f"{loss_type}_test"].append(
                evaluate_loss(loss, params, test, loss_type)
            )
            metrics[f"{loss_type}_train"].append(
                evaluate_loss(loss, params, train, loss_type)
            )
            logging.info(f"{loss_type}: {metrics[f'{loss_type}_train'][-1]:.8f}")

        objective = config.objective + "_valid"
        # pick best training from valid.
        if metrics[objective][-1] < best_sig:
            best_params = params
            best_sig = metrics[objective][-1]

        # Z_A
        z_a = get_significance(config, nn, params, train)
        logging.info(f"Z_A: {z_a:.8f}")
        metrics["Z_A"].append(z_a)

        end = perf_counter()
        logging.info(f"metric evaluation took {end-start:.4g}s")

        # once some bin value is nan, everything breaks unrecoverable, also
        # re-init does not work
        if any(np.isnan(bins)):
            sys.exit(
                "\n\033[0;31m" + f"ERROR: I tried bad bins: {metrics['bins'][i-1]}"
            )

        logging.info("\n")
        clear_caches()

    return best_params, metrics


# Function to evaluate loss for a given data set and loss type.
def evaluate_loss(loss, params, data, loss_type):
    return loss(params, data=data, loss_type=loss_type, bandwidth=1e-8)[0]


def get_significance(config, nn, params, data):
    data_dct = {k: v for k, v in zip(config.data_types, data)}
    if config.do_m_hh:
        yields = tomatos.histograms.hists_from_mhh(
            data=data_dct,
            bandwidth=1e-8,
            bins=params["bins"] if config.include_bins else config.bins,
        )
    else:
        yields = tomatos.histograms.hists_from_nn(
            nn_pars=params["nn_pars"],
            nn=nn,
            data=data_dct,
            bandwidth=config.bandwidth,  # for the bKDEs
            bins=jnp.array([0, *params["bins"], 1])
            if config.include_bins
            else config.bins,
        )
    return relaxed.metrics.asimov_sig(s=yields["NOSYS"], b=yields["bkg"])
