import gc
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
import tomatos.workspace
import tomatos.pipeline
import tomatos.solver
import tomatos.initialize
import equinox as eqx
import tomatos.batcher

Array = jnp.ndarray


# clear caches each update otherwise memory explodes
# https://github.com/google/jax/issues/10828
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
    init_pars,
    nn,
    nn_arch,
    args,
) -> tuple[Array, dict[str, list]]:

    # even though config is passed, need to keep redundant args here as this
    # function is jitted and used elsewhere, such that args need to be known at
    # compile time

    batch = {}
    for split in ["train", "valid", "test"]:
        batch[split] = tomatos.batcher.get_iterator(config, split)

    # this static non-optimizable vars on the loss function, need this as the
    # optimizer expects a function call loss(f(parameter),)
    # loss_fun = partial(
    #     tomatos.pipeline.data_to_loss,
    #     config=config,
    #     nn=nn,
    # )
    # solver = tomatos.solver.get(config, loss_fun)
    solver = tomatos.solver.get(config, tomatos.pipeline.loss_fun, init_pars)

    init_pars = init_pars
    params = init_pars
    best_params = init_pars
    best_test_loss = 999

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
            "bw",
            "slope",
            "best_epoch",
        ]
    }
    infer_metrics = {}

    # one step is one batch (not necessarily epoch)
    # gradient_accumulation interesting for more stable generalization in the
    # future as it reduces noise in the gradient updates
    # https://optax.readthedocs.io/en/latest/_collections/examples/gradient_accumulation.html

    for i in range(config.num_steps):
        start = perf_counter()
        logging.info(f"step {i}: loss={config.objective}")
        train, train_sf = next(batch["train"])
        print(train_sf)

        if i == 0:
            state = solver.init_state(
                init_pars,
                data=train,
                config=config,
                nn=nn,
                scale=train_sf,
            )

            histograms = state.aux

            for k in histograms.keys():
                metrics[k] = []
                metrics[k + "_test"] = []

        # since the optaxsolver holds the value from training, however it is
        # always step i-1 and evaluation is expensive --> evaluate valid and
        # test before update
        #
        # results can be checked with:
        # train_results = loss_fun(params, data=train, loss_type=config.objective)
        # print(train_results[0])

        # Evaluate losses.
        # small bandwidth + large slope for true hists

        valid, valid_sf = next(batch["valid"])
        valid_loss, valid_hists = loss_fun(
            params,
            data=valid,
            loss_type=config.objective,
            bandwidth=1e-20,
            slope=1e20,
            validate_only=True,
        )

        test, test_sf = next(batch["test"])
        test_loss, test_hists = loss_fun(
            params,
            data=test,
            loss_type=config.objective,
            bandwidth=1e-20,
            slope=1e20,
            validate_only=True,
        )

        metrics[f"{config.objective}_valid"].append(valid_loss)
        metrics[f"{config.objective}_test"].append(test_loss)
        logging.info(
            f"{config.objective} test: {metrics[f'{config.objective}_test'][-1]:.4f}"
        )

        # update
        params, state = solver.update(
            params,
            state,
            data=train,
            scale=1 / batch_frac,
            slope=config.slope,
        )
        # a large step can gow below 0 which breaks opt since
        # the cdf (and not the pdf) used for histogram calculation is flipped

        params["bw"] = np.maximum(config.bw_min, np.abs(params["bw"]))
        bw = params["bw"]
        histograms = state.aux

        if "bins" in params:
            # gradient updates basically guarantee that they are unique, so
            # sort is enough
            params["bins"] = np.clip(np.sort(np.abs(params["bins"])), 1e-6, 1 - 1e-6)
            bins = np.array([0, *params["bins"], 1])
        else:
            bins = config.bins

        # bins = np.array(params["bins"]) if "bins" in params else config.bins
        # save current bins
        if config.include_bins:
            actual_bins = (
                unscale_value(config, np.copy(bins), -3) if config.do_m_hh else bins
            )
            logging.info((f"next bin edges: {actual_bins}"))
            metrics["bins"].append(actual_bins)

        logging.info((f"bKDE hist sig: {histograms['NOSYS']}"))
        logging.info((f"bKDE hist bkg: {histograms['bkg']}"))

        logging.info(
            (f"valid ratio hist sig: {valid_hists['NOSYS']/histograms['NOSYS']}")
        )
        logging.info((f"valid ratio hist bkg: {valid_hists['bkg']/histograms['bkg']}"))

        logging.info(f"slope: {slope}")
        metrics["slope"].append(slope)
        # additional_logging(config, params, histograms)

        for hist in histograms.keys():
            metrics[hist].append(histograms[hist])
            metrics[hist + "_test"].append(test_hists[hist])

        # write train loss here, as optaxsolver state.value holds loss[i-1] and
        # evaluation is expensive
        metrics[f"{config.objective}_train"].append(state.value)

        logging.info(
            f"{config.objective} train: {metrics[f'{config.objective}_train'][-1]:.4f}"
        )

        yields, kde = get_yields(
            config,
            nn,
            params,
            train,
            bw,
            slope,
            bins,
            scale=1 / batch_frac,
        )

        # # Z_A
        z_a = asimov_sig(s=yields["NOSYS"], b=yields["bkg"])
        logging.info(f"Z_A: {z_a:.4f}")
        metrics["Z_A"].append(z_a)

        # measure diff between true and estimated hist
        def safe_divide(a, b):
            return np.where(b == 0, 0, a / b)

        if config.objective == "cls":

            # much nicer if we could the scaler transform i guess
            optimized_m_jj = unscale_value(config, params["vbf_cut"], -2)
            optimized_eta_jj = unscale_value(config, params["eta_cut"], -1)

            logging.info(f"vbf cut: {optimized_m_jj}")
            logging.info(f"eta cut: {optimized_eta_jj}")
            logging.info(f"bw: {bw}")

            metrics["vbf_cut"].append(optimized_m_jj)
            metrics["eta_cut"].append(optimized_eta_jj)
            metrics["bw"].append(bw)

            signal_approximation_diff = safe_divide(
                histograms["NOSYS"], yields["NOSYS"]
            )
            bkg_approximation_diff = safe_divide(histograms["bkg"], yields["bkg"])
            logging.info(f"signal estimate diff: {signal_approximation_diff}")
            logging.info(f"bkg estimate diff: {bkg_approximation_diff}")
            aux_info["kde_error"] = np.sum(
                np.abs(signal_approximation_diff - 1)
            ) + np.sum(np.abs(bkg_approximation_diff - 1))
            metrics["signal_approximation_diff"].append(signal_approximation_diff)
            metrics["bkg_approximation_diff"].append(bkg_approximation_diff)
            metrics["kde_signal"].append(
                rescale_kde(histograms["NOSYS"], kde["NOSYS"], bins)
            )
            metrics["kde_bkg"].append(rescale_kde(histograms["bkg"], kde["bkg"], bins))

            infer_metrics_i = {
                "inverted": is_inverted(histograms["NOSYS"]),
                "vbf_cut": metrics["vbf_cut"][-1],
                "eta_cut": metrics["eta_cut"][-1],
                "bins": bins,
            }
        else:
            infer_metrics_i = {}

        # once some bin value is nan, everything breaks unrecoverable, also
        # re-init does not work
        if any(np.isnan(bins)):
            sys.exit(
                "\n\033[0;31m" + f"ERROR: I tried bad bins: {metrics['bins'][i-1]}"
            )

        # pick best training
        objective = config.objective + "_test"
        if metrics[objective][-1] < best_test_loss:
            best_params = params
            best_test_loss = metrics[objective][-1]
            metrics["epoch_best"] = i
            infer_metrics["epoch_best"] = infer_metrics_i
            infer_metrics["epoch_best"]["epoch"] = i
        # save every 10th model to file
        if i % 10 == 0 and i != 0:
            epoch_name = f"epoch_{i:005d}"
            infer_metrics[epoch_name] = infer_metrics_i
            model = eqx.combine(params["nn_pars"], nn_arch)
            eqx.tree_serialise_leaves(config.model_path + epoch_name + ".eqx", model)
        if i == (config.num_steps - 1):
            infer_metrics["epoch_last"] = infer_metrics_i

        end = perf_counter()
        logging.info(f"update took {end-start:.4f}s")
        logging.info("\n")

        clear_caches()  # otherwise memore explodes

    return best_params, params, metrics, infer_metrics


def unscale_value(config, value, idx):
    value -= config.scaler_min[idx]
    value /= config.scaler_scale[idx]
    return value


def make_flat_bkg(config, batch_iterator, init_pars, nn):
    logging.info("Initializing Background")

    def mask_bw(params):
        # Return True only for 'bw', False for all other parameters
        return {key: key == "bw" for key in params}

    init_optimizer = optax.chain(
        # optax.zero_nans(),  # avoid NaNs in optimization
        optax.adam(0.01),  # Adam optimizer
        optax.masked(optax.clip(max_delta=0.001), mask_bw),
        # clipped_bw_optimizer,  # Clip gradients for bw
    )

    init_loss = partial(
        tomatos.initialize.pipeline,
        nn=nn,
        bandwidth=0.2,
        sample_names=config.data_types,
        config=config,
        bins=config.bins,
    )
    init_solver = OptaxSolver(init_loss, opt=init_optimizer, has_aux=True, jit=True)
    train, batch_num, num_batches = next(batch_iterator)

    init_pars["bw"] = 0.13
    state = init_solver.init_state(
        init_pars,
        data=train,
    )
    params = init_pars

    loss_value = jnp.inf
    i = 0
    previous_bw = 100
    bw_diff = []
    while np.abs(previous_bw - params["bw"]) > 0.000000001:
        train, batch_num, num_batches = next(batch_iterator)
        previous_bw = params["bw"]
        params, state = init_solver.update(
            params,
            state,
            data=train,
        )
        # print(np.abs(previous_bw - params["bw"]))
        bw_diff += [np.abs(previous_bw - params["bw"])]
        if i % 1 == 0:
            logging.info(f"Background Hist: {state.aux['bkg']}")
            logging.info(f"Signal     Hist: {state.aux['NOSYS']}")
            # print(state.value)

        loss_value = state.value
        i += 1
    # print(repr(bw_diff))

    return params


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
    """
    Median expected significance for a counting experiment, valid in the asymptotic regime.
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
    # need to upscale sampled kde hist as it is a very fine binned version of
    # the histogram,
    # note that this is still an approximation, only working fully properly for
    # the largest bin

    # use the largest bin of a binned kde hist
    max_bin_idx = np.argmax(hist)
    max_bin_edges = np.array([bins[max_bin_idx], bins[max_bin_idx + 1]])
    # integrate histogram for this bin
    hist_x_width = np.diff(max_bin_edges)
    hist_height = hist[max_bin_idx]
    area_hist = hist_x_width * hist_height

    # integrate kde for this bin
    kde_sampling = 1000
    kde_indices = (max_bin_edges * kde_sampling).astype(int)
    kde_heights = kde[kde_indices[0] : kde_indices[1]]
    kde_dx = 1 / kde_sampling
    area_kde = np.sum(kde_dx * kde_heights)

    scale_factor = area_hist / area_kde
    kde_scaled = kde * scale_factor

    return kde_scaled


def get_yields(config, nn, params, train, bw, slope, bins, scale):
    data_dct = {k: v for k, v in zip(config.data_types, train)}

    yields = tomatos.histograms.get_hists(
        nn_pars=params["nn_pars"],
        config=config,
        vbf_cut=params["vbf_cut"],
        eta_cut=params["eta_cut"],
        nn=nn,
        data=data_dct,
        bandwidth=1e-100,
        slope=1e100,
        bins=bins,
        scale=scale,
    )
    model, yields = tomatos.workspace.model_from_hists(
        do_m_hh=False,
        hists=yields,
        config=config,
        do_systematics=config.do_systematics,
        do_stat_error=config.do_stat_error,
        validate_only=False,
    )
    # dont need them for all
    kde_dict = dict((k, data_dct[k]) for k in ("NOSYS", "bkg"))
    # sample kde with 1000 bins
    kde_bins = jnp.linspace(bins[0], bins[-1], 1000)
    kde = tomatos.histograms.get_hists(
        nn_pars=params["nn_pars"],
        config=config,
        vbf_cut=params["vbf_cut"],
        eta_cut=params["eta_cut"],
        nn=nn,
        data=kde_dict,
        bandwidth=bw,
        slope=slope,
        bins=kde_bins,
        scale=scale,
    )

    return yields, kde


def is_inverted(hist):
    # Set inverted based on which half has the greater sum

    # Calculate the midpoint of the histogram
    midpoint = len(hist) // 2

    # Split the histogram into lower and upper halves
    lower_half = hist[:midpoint]
    upper_half = hist[midpoint:]

    return 1 if lower_half.sum() > upper_half.sum() else 0
