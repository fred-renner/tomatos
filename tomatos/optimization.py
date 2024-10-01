import gc
import logging
import sys
from functools import partial
from time import perf_counter

import jax.numpy as jnp
import numpy as np
import optax
import pyhf
import jax
from jaxopt import OptaxSolver

import tomatos.histograms
import tomatos.workspace
import tomatos.pipeline
import equinox as eqx

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
    valid,
    test,
    batch_iterator,
    init_pars,
    nn,
    nn_setup,
    args,
) -> tuple[Array, dict[str, list]]:
    # even though config is passed, need to keep redundant args here as this
    # function is jitted and used elsewhere, such that args need to be known at
    # compile time
    aux_info = {"kde_error": 0.0}
    loss = partial(
        tomatos.pipeline.pipeline,
        nn=nn,
        sample_names=config.data_types,
        include_bins=config.include_bins,
        do_m_hh=config.do_m_hh,
        loss_type=config.objective,
        config=config,
        bandwidth=0.2,
        slope=1000,
        do_systematics=config.do_systematics,
        do_stat_error=config.do_stat_error,
        validate_only=False,
        aux_info=aux_info,
    )
    config.aux = args.aux

    # Define a mask function that selectively scales only 'vbf_cut' and 'eta_cut'
    def mask_fn(params):
        # Return True for 'vbf_cut' and 'eta_cut', False otherwise
        return {key: key in ["vbf_cut", "eta_cut"] for key in params}

    # increase cut steps
    scale_cut_step = optax.masked(optax.scale(config.cuts_factor), mask_fn)
    # bw_decay = optax.cosine_decay_schedule(
    #     init_value=0.15,
    #     decay_steps=2000,
    #     alpha=0.01 / 0.15,
    # )
    bw_decay = optax.linear_schedule(
        init_value=0.2,
        end_value=0.01,
        transition_steps=np.maximum(2500, int(config.num_steps / 2)),
    )

    # lr_schedule = optax.linear_onecycle_schedule(
    #     transition_steps=config.num_steps,
    #     peak_value=0.002,  # Set the peak learning rate
    #     pct_start=0.3,  # Percentage of steps to reach the peak value
    #     div_factor=25,  # Factor by which to divide the peak value for the initial learning rate
    #     final_div_factor=10000,  # Factor by which to divide the peak value for the final learning rate
    #     pct_final=0.85,
    # )

    optimizer = optax.chain(
        optax.zero_nans(),  # otherwise optimization can break entirely
        optax.adam(config.lr),
        # scale_cut_step,
        # optax.masked(optax.clip(max_delta=config.aux), mask_fn),
        # optax.clip_by_global_norm(1.0),  # protect against learning spikes, however this is figthing against cosine_onecycle_schedule
    )

    solver = OptaxSolver(loss, opt=optimizer, has_aux=True, jit=True)

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
            "bw",
            "slope",
            "best_epoch",
        ]
    }
    infer_metrics = {}
    # one step is one batch (not necessarily epoch)
    for i in range(config.num_steps):
        start = perf_counter()
        logging.info(f"step {i}: loss={config.objective}")
        train, batch_num, num_batches = next(batch_iterator)
        # initialize with or without binning
        if i == 0:
            init_pars["vbf_cut"] = config.cuts_init
            init_pars["eta_cut"] = config.cuts_init

            if config.include_bins:
                init_pars["bins"] = config.bins
            else:
                init_pars.pop("bins", None)

            logging.info(init_pars)
            state = solver.init_state(
                init_pars,
                data=train,
            )

            histograms = state.aux

            for k in histograms.keys():
                metrics[k] = []
                metrics[k + "_test"] = []

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

        slope = 1000
        bw = bw_decay(i)

        valid_result, valid_hists = loss(
            params.copy(),  # let's be sure we are not influencing the training
            data=valid,
            loss_type=config.objective,
            bandwidth=1e-6,
            slope=1e6,
            validate_only=True,
        )
        test_result, test_hists = loss(
            params.copy(),  # let's be sure we are not influencing the training
            data=test,
            loss_type=config.objective,
            bandwidth=1e-6,
            slope=1e6,
            validate_only=True,
        )

        metrics[f"{config.objective}_valid"].append(valid_result)
        metrics[f"{config.objective}_test"].append(test_result)
        logging.info(
            f"{config.objective} valid: {metrics[f'{config.objective}_valid'][-1]:.4f}"
        )

        # update
        params, state = solver.update(
            params,
            state,
            bandwidth=bw,
            data=train,
            slope=slope,
        )

        histograms = state.aux
        bins = np.array(params["bins"]) if "bins" in params else config.bins
        # save current bins
        if config.include_bins:
            logging.info((f"next bin edges: {bins}"))
            metrics["bins"].append(bins)
        logging.info((f"bKDE hist sig: {histograms['NOSYS']}"))
        logging.info((f"bKDE hist bkg: {histograms['bkg']}"))

        logging.info((f"valid VR xbb 2: {valid_hists['bkg_VR_xbb_2']}"))

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

        yields, kde = get_yields(config, nn, params, train, bw, slope)

        # # Z_A
        z_a = asimov_sig(s=yields["NOSYS"], b=yields["bkg"])
        logging.info(f"Z_A: {z_a:.4f}")
        metrics["Z_A"].append(z_a)

        # measure diff between true and estimated hist
        def safe_divide(a, b):
            return jnp.where(b == 0, 0, a / b)

        if config.objective == "cls":

            def unscale_value(value, idx):
                value -= config.scaler_min[idx]
                value /= config.scaler_scale[idx]
                return value

            optimized_m_jj = unscale_value(params["vbf_cut"], -2)
            optimized_eta_jj = unscale_value(params["eta_cut"], -1)

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
            }
        else:
            infer_metrics_i = {}

        # once some bin value is nan, everything breaks unrecoverable, also
        # re-init does not work
        if any(np.isnan(bins)):
            sys.exit(
                "\n\033[0;31m" + f"ERROR: I tried bad bins: {metrics['bins'][i-1]}"
            )

        # pick best training from valid.
        objective = config.objective + "_valid"
        if metrics[objective][-1] < best_valid_loss:
            best_params = params
            best_valid_loss = metrics[objective][-1]
            metrics["epoch_best"] = i
            infer_metrics["epoch_best"] = infer_metrics_i
            infer_metrics["epoch_best"]["epoch"] = i
        # save every 10th model to file
        if i % 10 == 0 and i != 0:
            infer_metrics[str(i)] = infer_metrics_i
            model = eqx.combine(params["nn_pars"], nn_setup)
            eqx.tree_serialise_leaves(config.model_path + f"epoch_{i:004d}.eqx", model)
        if i == (config.num_steps - 1):
            infer_metrics["epoch_last"] = infer_metrics_i

        end = perf_counter()
        logging.info(f"update took {end-start:.4f}s")
        logging.info("\n")

        clear_caches()  # otherwise memore explodes

    return best_params, params, metrics, infer_metrics


def update_slope(config, i):

    # init slope
    m_0 = 50
    # final_slope, if too large can loose gradient
    m_n = 1_000
    i *= 1 / config.decay_quantile
    # the gradient scales linear with m
    m = (m_n - m_0) / config.num_steps
    slope = m * i + m_0

    # sigmoid slope m scales as 1/x with the cut uncertainty of the slope
    # however this is very steep at the training end
    # the gradient scales with m^2
    # to decrase unc linearly stepwise over the desired range
    # m_{i+1}=(1/m_i - unc)^{-1} = 1/m_0 - i*unc)^{-1}
    # unc = ((1 / m_0) - (1 / m_n) ) / n
    # delta_x = ((1 / m_0) - (1 / m_n)) / config.num_steps
    # slope = ((1 / m_0) - (i + 1) * delta_x) ** (-1)

    # truncate to largest value because want limited decay

    # if i > int(0.75 * config.num_steps):
    #     # the gradient scales linear with m
    #     m = (1000 - 100) / int(0.25 * config.num_steps)
    #     slope = m * i + 100
    # else:
    #     slope = 100

    if slope > m_n:
        return m_n

    return slope


def update_bw(config, i):
    bw_0 = 0.15
    # scale with lr
    if config.lr > 1e-4:
        m = -bw_0 / 500
    else:
        m = -bw_0 / 2500
    y = m * i + bw_0
    return np.maximum(y, 0.01)


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
    # need to upscale very fine binned version of the histogram,
    # results in much less entries per bin, to the actual bKDE to get a view
    # of the original kde

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


def get_yields(config, nn, params, train, bw, slope):
    data_dct = {k: v for k, v in zip(config.data_types, train)}
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
        kde_bins = jnp.linspace(bins[0], bins[-1], 1000)
        kde = tomatos.histograms.hists_from_nn(
            nn_pars=params["nn_pars"],
            config=config,
            vbf_cut=params["vbf_cut"],
            eta_cut=params["eta_cut"],
            nn=nn,
            data=kde_dict,
            bandwidth=bw,
            slope=slope,
            bins=kde_bins,
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
