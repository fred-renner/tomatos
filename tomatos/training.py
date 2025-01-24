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
import tomatos.optimizer
import tomatos.initialize
import equinox as eqx
import tomatos.batcher
import tomatos.constraints

# NB: outside of the opt you can use numpy everywhere


def run(config):

    batch = {}
    for split in ["train", "valid", "test"]:
        batch[split] = tomatos.batcher.get_generator(config, split)

    opt_pars = config.init_pars

    # get a reduced list of the configs necessary for opt, and make them jax
    # compatible
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
            "Z_A",
            "bins",
            "signal_approximation_diff",
            "bkg_approximation_diff",
            "kde_signal",
            "kde_bkg",
            "bw",
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
        metrics[f"valid_loss"].append(valid_loss)
        metrics[f"test_loss"].append(test_loss)
        metrics[f"train_loss"].append(state.value)

        opt_pars, state = optimizer.update(
            opt_pars,
            state,
            data=train_data,
            config=config,
            scale=train_sf,
        )

        hists = tomatos.utils.filter_hists(config, hists)

        opt_pars = tomatos.constraints.opt_pars(config, opt_pars)
        # save current bins
        if config.include_bins:
            actual_bins = (
                tomatos.utils.inverse_min_max_scale(
                    config, np.copy(opt_pars["bins"]), config.cls_var_idx
                )
                if config.objective == "cls_var"
                else config.bins
            )
            logging.info((f"next bin edges: {actual_bins}"))
            metrics["bins"].append(actual_bins)

        # for hist in hists.keys():
        #     metrics[hist].append(hists[hist])
        #     metrics[hist + "_test"].append(test_hists[hist])

        # # use train data batch to estimate hist without approximation

        # # sharp train data hists
        # sharp_hists = tomatos.pipeline.make_hists(
        #     opt_pars,
        #     train_data,
        #     config,
        #     train_sf,
        #     validate_only=True,
        # )

        # kde = sample_kde_distribution(
        #     config=config,
        #     opt_pars=opt_pars,
        #     data=train_data,
        #     scale=train_sf,
        # )

        # # # Z_A
        # z_a = asimov_sig(s=yields["NOSYS"], b=yields["bkg"])
        # logging.info(f"Z_A: {z_a:.4f}")
        # metrics["Z_A"].append(z_a)

        # # measure diff between true and estimated hist
        # def safe_divide(a, b):
        #     return np.where(b == 0, 0, a / b)

        # if config.objective == "cls":

        #     # much nicer if we could the scaler transform i guess
        #     optimized_m_jj = unscale_value(config, opt_pars["vbf_cut"], -2)
        #     optimized_eta_jj = unscale_value(config, opt_pars["eta_cut"], -1)

        #     logging.info(f"vbf cut: {optimized_m_jj}")
        #     logging.info(f"eta cut: {optimized_eta_jj}")
        #     logging.info(f"bw: {bw}")

        #     metrics["vbf_cut"].append(optimized_m_jj)
        #     metrics["eta_cut"].append(optimized_eta_jj)
        #     metrics["bw"].append(bw)

        #     signal_approximation_diff = safe_divide(hists["NOSYS"], yields["NOSYS"])
        #     bkg_approximation_diff = safe_divide(hists["bkg"], yields["bkg"])
        #     logging.info(f"signal estimate diff: {signal_approximation_diff}")
        #     logging.info(f"bkg estimate diff: {bkg_approximation_diff}")
        #     aux_info["kde_error"] = np.sum(
        #         np.abs(signal_approximation_diff - 1)
        #     ) + np.sum(np.abs(bkg_approximation_diff - 1))
        #     metrics["signal_approximation_diff"].append(signal_approximation_diff)
        #     metrics["bkg_approximation_diff"].append(bkg_approximation_diff)
        #     metrics["kde_signal"].append(
        #         rescale_kde(hists["NOSYS"], kde["NOSYS"], bins)
        #     )
        #     metrics["kde_bkg"].append(rescale_kde(hists["bkg"], kde["bkg"], bins))

        #     infer_metrics_i = {
        #         "inverted": is_inverted(hists["NOSYS"]),
        #         "vbf_cut": metrics["vbf_cut"][-1],
        #         "eta_cut": metrics["eta_cut"][-1],
        #         "bins": bins,
        #     }
        # else:
        #     infer_metrics_i = {}

        # # pick best training
        # objective = config.objective + "_test"
        # if metrics[objective][-1] < best_test_loss:
        #     best_opt_pars = opt_pars
        #     best_test_loss = metrics[objective][-1]
        #     metrics["epoch_best"] = i
        #     infer_metrics["epoch_best"] = infer_metrics_i
        #     infer_metrics["epoch_best"]["epoch"] = i
        # # save every 10th model to file
        # if i % 10 == 0 and i != 0:
        #     epoch_name = f"epoch_{i:005d}"
        #     infer_metrics[epoch_name] = infer_metrics_i
        #     model = eqx.combine(opt_pars["nn"], nn_arch)
        #     eqx.tree_serialise_leaves(config.model_path + epoch_name + ".eqx", model)
        # if i == (config.num_steps - 1):
        #     infer_metrics["epoch_last"] = infer_metrics_i

        # LOGGING
        # logging.info(
        #     f"{config.objective} train: {metrics[f'{config.objective}_train'][-1]:.4f}"
        # )
        # logging.info(f"test loss: {metrics[f'{config.objective}_test'][-1]:.4f}")

        # logging.info((f"bKDE hist sig: {hists['NOSYS']}"))
        # logging.info((f"bKDE hist bkg: {hists['bkg']}"))

        end = perf_counter()
        logging.info(f"update took {end-start:.4f}s")
        logging.info("\n")

        # otherwise memore explodes in this jax version
        tomatos.utils.clear_caches()

    return best_opt_pars, opt_pars, metrics, infer_metrics


def additional_logging(config, opt_pars, hists):
    if config.objective == "cls":
        if config.do_stat_error:
            logging.info((f"hist stat sig bin 0: {hists['NOSYS_stat_up_bin_0']}"))
            logging.info((f"hist stat bkg bin 0: {hists['bkg_stat_up_bin_0']}"))
            logging.info(
                (f"hist bkg unc: {1-(hists['bkg']/hists['bkg_shape_sys_up'])}")
            )
            logging.info((f"hist ps unc: {1-(hists['NOSYS']/hists['ps_up'])}"))
    logging.info(f"vbf scaled cut: {opt_pars['vbf_cut']}")
    logging.info(f"eta scaled cut: {opt_pars['eta_cut']}")


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
):
    # get kde distribution by sampling with a many bin histogram
    # enough to get kde only from the nominal ones
    sample_indices = np.arange(len(config.samples))
    nominal_data = data[sample_indices, :, :]
    kde_bins = jnp.linspace(0, 1, config.kde_sampling)
    kde_config = config.copy()
    kde_config.bins = kde_bins
    # to also collect the background estimate
    # this runs into trouble if hist transforms called....
    # kde_config.regions_to_sel = ["SR_btag_1", "SR_btag_2"]
    # lets actually see if thats expensive...
    kde_dist = tomatos.pipeline.make_hists(
        opt_pars,
        nominal_data,
        kde_config,
        scale,
    )

    return kde_dist
