import gc
import logging
import sys
from functools import partial
from time import perf_counter

import jax.numpy as jnp
import numpy as np
import optax
import pyhf

from jaxopt import OptaxSolver

import tomatos.histograms
import tomatos.pipeline

Array = jnp.ndarray


w_CR = 0.003785385121790652


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
    # function is jitted and used elsewhere, such that args need to be known at
    # compile time
    loss = partial(
        tomatos.pipeline.pipeline,
        nn=nn,
        sample_names=config.data_types,
        include_bins=config.include_bins,
        do_m_hh=config.do_m_hh,
        loss_type=config.objective,
        config=config,
        bandwidth=config.bandwidth,
        slope=config.slope,
        do_systematics=config.do_systematics,
        do_stat_error=config.do_stat_error,
    )

    solver = OptaxSolver(loss, opt=optax.adam(config.lr), has_aux=True, jit=True)

    pyhf.set_backend("jax", default=True, precision="64b")

    params = init_pars
    best_params = init_pars
    best_valid_loss = 999
    print(config.data_types)
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
            "vbf_cut",
            "eta_cut",
            "signal_approximation_diff",
            "bkg_approximation_diff",
            "kde_signal",
            "kde_bkg",
            "best_epoch",
        ]
    }

    # one step is one batch (not necessarily epoch)
    for i in range(config.num_steps):
        start = perf_counter()
        logging.info(f"step {i}: loss={config.objective}")
        train, batch_num, num_batches = next(batch_iterator)
        # initialize with or without binning
        if i == 0:
            init_pars["vbf_cut"] = 0.0
            init_pars["eta_cut"] = 0.0
            if config.include_bins:
                init_pars["bins"] = config.bins
            else:
                init_pars.pop("bins", None)

            logging.info(init_pars)
            state = solver.init_state(
                init_pars,
                data=train,
            )

        # warm reset doesn't seem to work
        # if i % 20 == 0:
        #     state = solver.init_state(
        #         params,
        #         data=train,
        #     )

        # since the optaxsolver holds the value from training, however it is
        # always step i-1 and evaluation is expensive --> evaluate valid and
        # test before update
        #
        # results can be checked with:
        # train_results = loss(params, data=train, loss_type=config.objective)
        # print(train_results[0])

        # Evaluate losses.
        # small bandwidth + large slope for true hists
        valid_result = loss(
            params,
            data=valid,
            loss_type=config.objective,
            bandwidth=1e-6,
            slope=1e6,
        )
        test_result = loss(
            params,
            data=test,
            loss_type=config.objective,
            bandwidth=1e-6,
            slope=1e6,
        )

        metrics[f"{config.objective}_valid"].append(valid_result[0])
        metrics[f"{config.objective}_test"].append(test_result[0])
        logging.info(
            f"{config.objective} valid: {metrics[f'{config.objective}_valid'][-1]:.8f}"
        )

        # update
        params, state = solver.update(
            params,
            state,
            data=train,
        )
        histograms = state.aux
        if i == 0:
            for k in histograms.keys():
                metrics[k] = []

        bins = np.array(params["bins"]) if "bins" in params else config.bins
        # save current bins
        if config.include_bins:
            logging.info((f"next bin edges: {bins}"))
            metrics["bins"].append(bins)
        logging.info((f"hist sig: {histograms['NOSYS']}"))
        logging.info((f"hist bkg: {histograms['bkg']}"))

        # additional_logging(config, params, histograms)

        def unscale_value(value, idx):
            # cut optimization is supported with rescaling of parameter in
            # histograms.py
            value *= 3
            value -= config.scaler_min[idx]
            value /= config.scaler_scale[idx]
            return value

        optimized_m_jj = unscale_value(params["vbf_cut"], -2)
        optimized_eta_jj = unscale_value(params["eta_cut"], -1)

        logging.info(f"vbf cut: {optimized_m_jj}")
        logging.info(f"eta cut: {optimized_eta_jj}")
        metrics["vbf_cut"].append(optimized_m_jj)
        metrics["eta_cut"].append(optimized_eta_jj)

        for hist in histograms.keys():
            metrics[hist].append(histograms[hist])

        # write train loss here, as optaxsolver state.value holds loss[i-1] and
        # evaluation is expensive
        metrics[f"{config.objective}_train"].append(state.value)
        logging.info(
            f"{config.objective} train: {metrics[f'{config.objective}_train'][-1]:.8f}"
        )

        # pick best training from valid.
        objective = config.objective + "_valid"
        if metrics[objective][-1] < best_valid_loss:
            best_params = params
            best_valid_loss = metrics[objective][-1]
            metrics["best_epoch"] = i
            logging.info(f"NEW BEST PARAMS IN EPOCH {i}")

        # get yields from sharp hists using training data set
        yields, kde = get_yields(config, nn, params, train)

        # # Z_A
        z_a = asimov_sig(s=yields["NOSYS"], b=yields["bkg"])
        logging.info(f"Z_A: {z_a:.8f}")
        metrics["Z_A"].append(z_a)

        # # # measure diff between true and estimated hist
        def safe_divide(a, b):
            return jnp.where(b == 0, 0, a / b)

        if config.objective == "cls":
            signal_approximation_diff = safe_divide(
                histograms["NOSYS"], yields["NOSYS"]
            )
            bkg_approximation_diff = safe_divide(histograms["bkg"], yields["bkg"])
            logging.info(f"signal estimate diff: {signal_approximation_diff}")
            logging.info(f"bkg estimate diff: {bkg_approximation_diff}")
            metrics["signal_approximation_diff"].append(signal_approximation_diff)
            metrics["bkg_approximation_diff"].append(bkg_approximation_diff)

            metrics["kde_signal"].append(
                rescale_kde(histograms["NOSYS"], kde["NOSYS"], bins)
            )
            metrics["kde_bkg"].append(rescale_kde(histograms["bkg"], kde["bkg"], bins))
        # once some bin value is nan, everything breaks unrecoverable, also
        # re-init does not work
        if any(np.isnan(bins)):
            sys.exit(
                "\n\033[0;31m" + f"ERROR: I tried bad bins: {metrics['bins'][i-1]}"
            )

        end = perf_counter()
        logging.info(f"update took {end-start:.4f}s")
        logging.info("\n")
        clear_caches()

    return best_params, metrics


def additional_logging(config, params, histograms):
    if config.objective == "cls":
        if config.do_stat_error:
            logging.info((f"hist stat sig bin 0: {histograms['NOSYS_stat_up_bin_0']}"))
            logging.info((f"hist stat bkg bin 0: {histograms['bkg_stat_up_bin_0']}"))
            logging.info(
                (
                    f"hist bkg unc: {1-(histograms['bkg']/histograms['bkg_shape_sys_up'])}"
                )
            )
            logging.info(
                (f"hist ps unc: {1-(histograms['NOSYS']/histograms['ps_up'])}")
            )
    logging.info(f"vbf scaled cut: {params['vbf_cut']}")
    logging.info(f"eta scaled cut: {params['eta_cut']}")


def asimov_sig(s, b) -> float:
    """Median expected significance for a counting experiment, valid in the asymptotic regime.
    Also valid for the multi-bin case.

    Parameters
    ----------
    s : Array
        Signal counts.
    b : Array
        Background counts.

    Returns
    -------
    float
        The expected significance.
    """
    # Convert s and b to numpy arrays to handle element-wise operations
    s = np.asarray(s)
    b = np.asarray(b)

    # Prevent division by zero and log of zero
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(b > 0, s / b, 0)
        term = np.where(b > 0, (s + b) * np.log1p(ratio) - s, 0)

    q0 = 2 * np.sum(term)
    return q0**0.5


def rescale_kde(hist, kde, bins):
    # find the largest bin of binned kde hist
    max_bin_idx = jnp.argmax(hist)
    # get the indices from the fined grained kde that would fall into each
    # bin
    splitted_kde = jnp.array_split(kde, len(bins) - 1)
    # integrate the fine grained ones
    kde_count = splitted_kde[max_bin_idx].sum()
    # areas for both bins must be same when integerated
    area_hist = hist[max_bin_idx] * (bins[max_bin_idx + 1] - bins[max_bin_idx])
    kde_width = (bins[-1] - bins[0]) / len(kde)
    area_kde = kde_count * kde_width

    scale_factor = area_hist / area_kde
    kde *= scale_factor

    return kde


def get_yields(config, nn, params, data):
    data_dct = {k: v for k, v in zip(config.data_types, data)}
    if config.do_m_hh:
        yields = tomatos.histograms.hists_from_mhh(
            data=data_dct,
            bandwidth=1e-6,
            bins=params["bins"] if config.include_bins else config.bins,
        )
    else:
        bins = params["bins"] if config.include_bins else config.bins
        yields = tomatos.histograms.hists_from_nn(
            nn_pars=params["nn_pars"],
            config=config,
            vbf_cut=params["vbf_cut"],
            eta_cut=params["eta_cut"],
            nn=nn,
            data=data_dct,
            bandwidth=1e-6,
            slope=1e6,
            bins=bins,
        )
        # dont need them for all
        kde_dict = dict((k, data_dct[k]) for k in ("NOSYS", "bkg"))
        kde_bins = jnp.linspace(bins[0], bins[-1], 1000)
        kde = tomatos.histograms.hists_from_nn(
            nn_pars=params["nn_pars"],
            config=config,
            vbf_cut=params["vbf_cut"],
            eta_cut=params["eta_cut"],
            nn=nn,
            data=kde_dict,
            bandwidth=config.bandwidth,
            slope=config.slope,
            bins=kde_bins,
        )
    if config.objective == "cls":
        for y in yields.keys():
            if "bkg" in y:
                yields[y] *= w_CR

    return yields, kde
