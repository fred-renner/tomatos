from functools import partial
from time import perf_counter
import jax
import jax.numpy as jnp
import optax
import pyhf
import relaxed
from jaxopt import OptaxSolver
import hh_neos.pipeline
import gc
import sys
import hh_neos.histograms
import logging

Array = jnp.ndarray
import numpy as np

np.set_printoptions(precision=3)


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
    test,
    batch_iterator,
    init_pars,
    nn,
) -> tuple[Array, dict[str, list]]:
    loss = partial(
        hh_neos.pipeline.pipeline,
        nn=nn,
        sample_names=config.data_types,
        include_bins=config.include_bins,
        do_m_hh=config.do_m_hh,
        loss=config.objective,
        config=config,
    )

    # sometimes adagrad can also work
    solver = OptaxSolver(loss, opt=optax.adam(config.lr), jit=True)

    pyhf.set_backend("jax", default=True, precision="64b")

    params = init_pars
    best_params = init_pars
    best_sig = 999
    # metrics = {k: [] for k in ["cls", "discovery", "poi_uncert"]}
    metrics = {k: [] for k in ["cls"]}
    train_loss = []
    test_loss = []
    z_a = []
    bins_per_step = []

    # one step is one batch, not epoch
    for i in range(config.num_steps):
        logging.info(f"step {i}: loss={config.objective}")
        data, batch_num, num_batches = next(batch_iterator)

        if i == 0:
            if config.include_bins:
                init_pars["bins"] = config.bins[1:-1]
                logging.info(init_pars)
                state = solver.init_state(
                    init_pars,
                    data=data,
                    bandwidth=config.bandwidth,
                )
                prev_bins = config.bins[1:-1]
            else:
                if "bins" in init_pars:
                    del init_pars["bins"]
                state = solver.init_state(
                    init_pars,
                    bins=config.bins,
                    data=data,
                    bandwidth=config.bandwidth,
                )
                prev_bins = config.bins

        if "bins" in init_pars and i > 0:
            prev_bins = np.array(params["bins"])

        start = perf_counter()
        params, state = solver.update(
            params,
            state,
            bins=config.bins,
            data=data,
            bandwidth=config.bandwidth,
        )

        end = perf_counter()
        logging.info(f"update took {end-start:.4g}s")
        if "bins" in params:
            bin_edges = np.array([0, *params["bins"], 1])
            logging.info((f"bin edges: {bin_edges}"))
            bins_per_step.append(params["bins"])
        for metric in metrics:
            # evaluate loss on test set
            test_metric = loss(
                params, bins=config.bins, data=test, loss=metric, bandwidth=1e-8
            )  # small bandwidth to have "spikes"

            logging.info(f"{metric}: {test_metric:.4g}")
            metrics[metric].append(test_metric)
            test_loss.append(test_metric)

            # evaluate loss on train set
            train_metric = loss(
                params, bins=config.bins, data=data, loss=metric, bandwidth=1e-8
            )
            train_loss.append(train_metric)

        # find best training...
        if metrics[config.objective][-1] < best_sig:
            best_params = params
            best_sig = metrics[config.objective][-1]

        # for Z_A
        # becomes issue for proper batching
        z_a.append(get_significance(config, nn, params, data))

        # find best binning
        bins = np.array(params["bins"])
        corrected_bins = bin_correction(bins)
        if len(corrected_bins) != len(bins):
            params["bins"] = corrected_bins
            state = solver.init_state(
                params,
                data=data,
                bandwidth=config.bandwidth,
            )

        logging.info("\n")
        clear_caches()

    metrics["Z_A"] = z_a
    metrics["bins"] = bins_per_step
    metrics["cls_train"] = train_loss
    metrics["cls_test"] = test_loss
    return best_params, metrics


def bin_correction(bins):
    left_neighbor_larger = bins[:-1] < bins[1:]
    left_neighbor_larger = np.append(True, left_neighbor_larger)

    combined_condition = (bins < 1) & (bins > 0) & left_neighbor_larger
    corrected_bins = bins[combined_condition]

    # Ensure at least one bin remains after filtering.
    return corrected_bins if corrected_bins.size > 0 else np.array([0.5])


def get_significance(config, nn, params, data):
    data_dct = {k: v for k, v in zip(config.data_types, data)}
    if config.do_m_hh:
        yields = hh_neos.histograms.hists_from_mhh(
            data=data_dct,
            bandwidth=1e-8,
            bins=params["bins"] if config.include_bins else config.bins,
        )
    else:
        yields = hh_neos.histograms.hists_from_nn(
            pars=params["nn_pars"],
            nn=nn,
            data=data_dct,
            bandwidth=config.bandwidth,  # for the bKDEs
            bins=jnp.array([0, *params["bins"], 1])
            if config.include_bins
            else config.bins,
        )
    this_z_a = relaxed.metrics.asimov_sig(s=yields["NOSYS"], b=yields["bkg"])
    logging.info(("Z_A: ", this_z_a))
    return this_z_a
