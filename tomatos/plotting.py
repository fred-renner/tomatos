import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pyhf
import relaxed
import os
import tomatos.histograms
import tomatos.utils
import tomatos.workspace
import logging
import uproot

import os
import numpy as np
import uproot
import matplotlib.pyplot as plt
import math
import h5py


def plot_inputs(config):
    """
    Create individual plots for each variable and a combined grid of all plots.

    Args:
        config: Configuration object containing variables, file paths, and tree name.
    """
    plot_path = config.results_path + "input_plots/"
    if not os.path.isdir(plot_path):
        os.makedirs(plot_path)

    # Determine grid size for combined plot
    n_vars = len(config.vars)

    cols = math.ceil(math.sqrt(n_vars))
    rows = math.ceil(n_vars / cols)

    # Create combined grid plot
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for i, var in enumerate(config.vars):
        ax = axes[i]
        # Individual plot setup
        plt.figure(figsize=(6, 4))
        for sample, path in config.sample_files_dict.items():
            tree = uproot.open(path)[config.tree_name]
            data = tree[var].array(library="np")

            # Calculate mean and standard deviation
            mean = np.mean(data)
            std = np.std(data)

            # Filter data within 3 standard deviations
            filtered_data = data[(data > mean - (3 * std)) & (data < mean + (3 * std))]

            # Plot histogram

            with np.errstate(divide="ignore", invalid="ignore"):
                ax.hist(
                    filtered_data,
                    bins=30,
                    density=True,
                    histtype="step",
                    label=sample,
                )
                plt.hist(
                    filtered_data,
                    bins=30,
                    density=True,
                    histtype="step",
                    label=sample,
                )
        ax.set_xlabel(var)
        ax.set_ylabel("Event Density")
        ax.legend()

        # Save individual plot
        plt.xlabel(var)
        plt.ylabel("Event Density")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{plot_path}/{var}.pdf")
        plt.close()

    # Turn off any unused axes
    for ax in axes[n_vars:]:
        ax.axis("off")

    plt.tight_layout()
    logging.info("Combined input plots here: " + plot_path + "combined_input_plots.pdf")
    plt.savefig(plot_path + "combined_input_plots.pdf")
    plt.close()


def interpolate_gaps(values, limit=None):
    """
    Fill gaps using linear interpolation, optionally only fill gaps up to a
    size of `limit`.
    """
    values = np.asarray(values)
    i = np.arange(values.size)
    valid = np.isfinite(values)
    filled = np.interp(i, i[valid], values[valid])

    if limit is not None:
        invalid = ~valid
        for n in range(1, limit + 1):
            invalid[:-n] &= invalid[n:]
        filled[invalid] = np.nan

    return filled


def hist(config, metrics):

    plt.figure(figsize=config.fig_size)

    for key, value in metrics.items():
        if not key.startswith("h_"):
            continue

        if any([reg in key for reg in config.hist_skip_pattern]):
            continue

        edges = (
            metrics["bins"][i]
            if meta_data["config"]["include_bins"]
            else meta_data["config"]["bins"]
        )
        plt.stairs(
            edges=edges,
            values=metrics[key][i],
            # values=metrics[key + "_test"][i],
            # with kde!
            label=label,
            fill=None,
            linewidth=1,
            # align="edge",
        )

    if meta_data["config"]["do_m_hh"]:
        plt.xlabel("m$_{HH}$ (MeV)")
    else:
        plt.xlabel("NN score")

    if meta_data["config"]["objective"] == "cls":
        plt.plot(
            np.linspace(edges[0], edges[-1], len(metrics["kde_signal"][0]))[1:-1],
            metrics["kde_signal"][i][1:-1],
            label="kde signal",
            color="tab:orange",
        )
        plt.plot(
            np.linspace(edges[0], edges[-1], len(metrics["kde_bkg"][0]))[1:-1],
            metrics["kde_bkg"][i][1:-1],
            label="kde bkg",
            color="tab:blue",
        )

    plt.stairs(
        edges=edges,
        values=metrics["bkg"][i],
        fill=None,
        linewidth=2,
        color="tab:blue",
    )
    plt.stairs(
        edges=edges,
        values=metrics["NOSYS"][i],
        fill=None,
        linewidth=2,
        color="tab:orange",
    )

    plt.title(f"Epoch {i}")
    plt.ylabel("Events")

    if meta_data["config"]["objective"] == "bce":
        plt.legend(loc="upper right")
    else:
        plt.legend(
            prop={"size": 6.4},
            ncols=3,
            loc="upper center",
            #  loc="center left"
        )
    plt.tight_layout()
    plt.savefig(
        image_path + "/" + f"{i:005d}" + ".png",
        dpi=208,  # divisable by 16 for imageio
    )
    plt.close()


def cls(metrics, config, batch_grid, fig_size):
    if len(metrics["cls_train"]) > 0:
        metrics["cls_train"] = interpolate_gaps(np.array(metrics["cls_train"]))
        metrics["cls_valid"] = interpolate_gaps(np.array(metrics["cls_valid"]))
        metrics["cls_test"] = interpolate_gaps(np.array(metrics["cls_test"]))

        plt.figure(figsize=fig_size)
        plt.plot(batch_grid, metrics["cls_train"], label=r"$CL_s$ train")
        plt.plot(batch_grid, metrics["cls_valid"], label=r"$CL_s$ valid")
        plt.plot(batch_grid, metrics["cls_test"], label=r"$CL_s$ test")

        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.ylim([0, np.max(metrics["cls_train"]) * 1.3])
        plt.tight_layout()
        plot_path = config.results_path + "cls.pdf"
        print(plot_path)
        plt.savefig(plot_path)
        plt.close()


def bw(metrics, config, batch_grid, fig_size):
    if len(metrics["bw"]) > 0:

        plt.figure(figsize=fig_size)
        plt.plot(batch_grid, metrics["bw"])

        plt.xlabel("Epoch")
        plt.ylabel("Bandwidth")
        plt.tight_layout()
        plot_path = config.results_path + "bandwidth.pdf"
        print(plot_path)
        plt.savefig(plot_path)
        plt.close()


def bce(metrics, config, batch_grid, fig_size):
    if len(metrics["bce_train"]) > 0:
        plt.figure(figsize=fig_size)
        plt.plot(batch_grid, metrics["bce_train"], label=r"bce train")
        plt.plot(batch_grid, metrics["bce_valid"], label=r"bce valid")
        plt.plot(batch_grid, metrics["bce_test"], label=r"bce test")

        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.tight_layout()
        plot_path = config.results_path + "bce.pdf"
        print(plot_path)
        plt.savefig(plot_path)
        plt.close()


def Z_A(metrics, config, batch_grid, fig_size, ylim):
    if len(metrics["Z_A"]) > 0:
        plt.figure(figsize=fig_size)
        plt.plot(batch_grid, metrics["Z_A"])
        plt.xlabel("Epoch")
        plt.ylabel("Asimov Significance")
        plt.tight_layout()
        plt.ylim(ylim)
        plot_path = config.results_path + "Z_A.pdf"
        print(plot_path)
        plt.savefig(plot_path)
        plt.close()


def bins(metrics, config, batch_grid, fig_size):
    if len(metrics["bins"]) > 0:
        plt.figure(figsize=fig_size)
        bins = np.array(metrics["bins"])
        for i in range(len(metrics["bins"][0])):
            plt.plot(bins[:, i], np.arange(len(bins[:, i])), label=f"Bin Edge {i+1}")
            # for i, bins in enumerate(metrics["bins"]):
            if config.do_m_hh and config.include_bins:
                # bins = (np.array(bins) - config.scaler_min[-3]) / config[
                #     "scaler_scale"
                # ][-3]
                plt.xlabel("m$_{hh}$ (MeV)")
            else:
                plt.xlabel("NN score")
            # plt.vlines(x=bins, ymin=i, ymax=i + 1)
            # plt.scatter(x=bins, y=i)
            plt.ylabel("Epoch")
        plt.legend()
        plt.tight_layout()
        plot_path = config.results_path + "bins.pdf"
        print(plot_path)
        plt.savefig(plot_path)
        plt.close()


def cuts(metrics, config, batch_grid, fig_size):
    if len(np.array(metrics["vbf_cut"])) > 0:
        fig, ax1 = plt.subplots(figsize=fig_size)
        color = "tab:red"
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel(r"$m_{jj}$ (TeV)", color=color)
        ax1.plot(np.array(metrics["vbf_cut"]) * 1e-6, color=color)
        ax1.tick_params(axis="y", labelcolor=color)

        ax2 = ax1.twinx()
        color = "tab:blue"
        ax2.set_ylabel(r"$|\Delta\eta(j,j)|$", color=color)
        ax2.plot(metrics["eta_cut"], color=color)
        ax2.tick_params(axis="y", labelcolor=color)

        fig.tight_layout()
        plot_path = config.results_path + "cuts.pdf"
        print(plot_path)
        plt.savefig(plot_path)
        plt.close()


def rel_error(metrics, err_hist, err_hist_label, nom_hist, config, fig_size, ylim):
    if err_hist in metrics.keys() and len(metrics[err_hist]) > 0:
        plt.figure(figsize=fig_size)
        err = np.array(metrics[err_hist])
        nom = np.array(metrics[nom_hist])
        ratio = err / nom

        for i in range(len(metrics[err_hist][0])):
            plt.plot(ratio[:, i], label=f"Bin {i+1}")

        plt.xlabel("Epoch")
        if "NOSYS" in nom_hist:
            nom = "Nominal Signal"
        else:
            nom = "Nominal Bkg"
        # s_err_hist=err_hist.replace("_"," ")
        # s_err_hist=s_err_hist.replace("1up","")
        # s_err_hist=s_err_hist.replace("up","")
        plt.ylabel(f"({err_hist_label}) / ({nom})")
        plt.legend()
        plt.ylim(ylim)
        plt.tight_layout()
        plot_path = config.results_path + f"{err_hist}_rel_error.pdf"
        print(plot_path)
        plt.savefig(plot_path)
        plt.close()


def hists(metrics, config, batch_grid, fig_size):
    for k, m in metrics.items():
        if "kde" in k:
            continue
        m = np.array(m)
        if m.ndim == 2:
            plt.figure(figsize=fig_size)

            for i in range(len(m[0])):
                plt.plot(m[:, i], label=f"Bin {i+1}")

            plt.xlabel("Epoch")
            if "NOSYS" in k:
                y_label = r"$\kappa_\mathrm{2V}=0$ signal"
            elif "bkg" in k:
                y_label = "Background"
            else:
                y_label = k.replace("_", " ")

            plt.ylabel(y_label)
            plt.legend()
            plt.tight_layout()
            plot_path = config.results_path + f"hists/{k}.pdf"
            print(plot_path)
            plt.savefig(plot_path)
            plt.close()


def signal_approximation(metrics, config, batch_grid, fig_size):
    if len(metrics["signal_approximation_diff"]) > 0:
        plt.figure(figsize=(8, 4))
        diff = np.array(metrics["signal_approximation_diff"])
        for i in range(len(metrics["NOSYS"][0])):
            plt.plot(diff[:, i], label=f"Bin {i+1}", alpha=0.5)

        plt.xlabel("Epoch")
        plt.ylabel("Binned KDE/Nominal")
        plt.ylim([0.9, 1.1])
        plt.legend(loc="upper left")
        plt.tight_layout()
        plot_path = config.results_path + "signal_approximation_diff.pdf"
        print(plot_path)
        plt.savefig(plot_path)
        plt.close()


def bkg_approximation(metrics, config, batch_grid, fig_size):
    if len(metrics["bkg_approximation_diff"]) > 0:
        plt.figure(figsize=(8, 4))
        diff = np.array(metrics["bkg_approximation_diff"])

        for i in reversed(range(len(metrics["NOSYS"][0]))):
            plt.plot(diff[:, i], label=f"Bin {i+1}", alpha=0.5)

        plt.xlabel("Epoch")
        plt.ylabel("Binned KDE/Nominal")
        plt.ylim([0.9, 1.1])
        plt.legend(loc="upper left")
        plt.tight_layout()
        plot_path = config.results_path + "bkg_approximation_diff.pdf"
        print(plot_path)
        plt.savefig(plot_path)
        plt.close()


def total_diff(metrics, config, fig_size):
    if len(metrics["signal_approximation_diff"]) > 0:
        n_bins = len(config.bins) - 1

        plt.figure(figsize=fig_size)

        n_bins = len(metrics["signal_approximation_diff"][0])
        total_sig_diff = (
            np.max(np.abs(np.array(metrics["signal_approximation_diff"]) - 1), axis=1)
            * 100
        )
        total_bkg_diff = (
            np.max(np.abs(np.array(metrics["bkg_approximation_diff"]) - 1), axis=1)
            * 100
        )

        plt.plot(total_sig_diff / n_bins, label=r"$\kappa_\mathrm{2V}=0$ signal")
        plt.plot(total_bkg_diff / n_bins, label="Background Estimate")

        plt.xlabel("Epoch")
        plt.ylabel("Largest Bin Deviation (%)")
        plt.ylim([0, 10])
        plt.legend()
        plt.tight_layout()
        plot_path = config.results_path + "summed_diff.pdf"
        print(plot_path)
        plt.savefig(plot_path)
        plt.close()


def plots(config):
    with h5py.File(config.metrics_file_path, "r") as metrics:
        # plt.rcParams.update({"font.size": 16})
        # plt.rcParams["lines.linewidth"] = 1

        hist(config, metrics)

        rel_error(
            metrics,
            "xbb_pt_bin_3__1up",
            "xbb pt bin 3 up",
            "NOSYS",
            config,
            fig_size,
            ylim=[1, 2.0],
        )
        rel_error(
            metrics,
            "gen_up",
            "Scale Variations up",
            "NOSYS",
            config,
            fig_size,
            ylim=[1.1, 1.225],
        )
        rel_error(
            metrics,
            "ps_up",
            "Parton Shower up",
            "NOSYS",
            config,
            fig_size,
            ylim=[1, 1.35],
        )
        rel_error(
            metrics,
            "bkg_shape_sys_up",
            "bkg shape up",
            "bkg",
            config,
            fig_size,
            ylim=[1, 4],
        )
        if config.objective == "cls":
            hists(metrics, config, batch_grid, fig_size)
            cls(metrics, config, batch_grid, fig_size)
            bw(metrics, config, batch_grid, fig_size)
            bins(metrics, config, batch_grid, fig_size)
            cuts(metrics, config, batch_grid, fig_size)
            signal_approximation(metrics, config, batch_grid, fig_size)
            bkg_approximation(metrics, config, batch_grid, fig_size)
            total_diff(metrics, config, fig_size)
        elif config.objective == "bce":
            hists(metrics, config, batch_grid, fig_size)
            bce(metrics, config, batch_grid, fig_size)


# def plot():
#     config = tomatos.configuration.Setup(args)
#     results = {}
#     with open(config.metadata_file_path, "r") as file:
#         results = json.load(file)

#     with open(config.metrics_file_path, "r") as file:
#         metrics = json.load(file)
#     tomatos.plotting.main(
#         results["config"],
#         results["bins"],
#         results["yields"],
#         metrics,
#     )
