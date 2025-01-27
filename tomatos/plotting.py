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
import json


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


def loss(config, metrics):
    plt.plot(interpolate_gaps(metrics["train_loss"]), label=r"train")
    plt.plot(interpolate_gaps(metrics["valid_loss"]), label=r"valid")
    plt.plot(interpolate_gaps(metrics["test_loss"]), label=r"test")
    plt.xlabel("Batch")
    loss = r"$CL_s$" if "cls" in config.objective else "BCE"
    plt.ylabel(f"{loss} Loss")
    fig_finalize(config, "loss.pdf")


def bw(config, metrics):
    plt.plot(metrics["bw"][:])
    plt.xlabel("Batch")
    plt.ylabel("Bandwidth")
    fig_finalize(config, "bandwidth.pdf")


def bins(metrics, config):
    bins = np.array(metrics["bins"])
    for i in range(len(metrics["bins"][0])):
        plt.plot(bins[:, i], np.arange(len(bins[:, i])), label=f"Bin Edge {i+1}")

    if "var" in config.objective:
        plt.xlabel(config.cls_var)
    else:
        plt.xlabel("Neural Network Score")

    plt.ylabel("Batch")
    fig_finalize(config, "bins.pdf")


def cuts(config, metrics):
    for k in config.opt_cuts:
        cut_var = "cut_" + k
        plt.plot(metrics[cut_var][:])
        plt.ylabel(k + " cut")
        plt.xlabel("Batch")
        fig_finalize(config, cut_var + ".pdf")


def sharp_hist_deviation(config, metrics):
    # this is very large at the beginning, because of possibly empty sharp
    # bins, but it gives a nice idea how well the approximation works at later
    # batches
    for key in metrics.keys():
        # make sure we have a test hist and avoid sharp train data
        if key.startswith("h_") and key.endswith("_sharp"):
            sharp = metrics[key][:]
            key_NOSYS = key.replace("_sharp", "")
            nom = metrics[key.replace("_sharp", "")][:]
            largest_bin_deviations = np.max(np.abs(nom / sharp - 1), axis=1)

            plt.plot(largest_bin_deviations * 100, label=key)
            plt.xlabel("Batch")
            plt.ylabel("Largest Bin Deviation (%)")
            fig_finalize(config, f"{key_NOSYS}_largest_bin_deviation.pdf")


def hist(config, metrics, h_key, batch_i, add_kde):
    key = h_key.replace("h_", "")
    label = key.replace("_", " ")
    edges = metrics["bins"][batch_i] if config.include_bins else config.bins
    h_plot = plt.stairs(
        edges=edges,
        values=metrics[h_key][batch_i],
        label=label,
        # linewidth=1,
    )

    plt.ylabel("Events")
    if "var" in config.objective:
        plt.xlabel(config.cls_var)
    else:
        plt.xlabel("Neural Network Score")

    if add_kde and "NOSYS" in key:
        try:
            kde = metrics["kde_" + key][batch_i]
            plt.plot(
                config.kde_bins[:-1],
                kde,
                label="KDE",
                color=h_plot.get_edgecolor(),
            )
        except:
            pass


def fig_finalize(config, name, legend_outside=False):
    # Scale x-axis
    # with open(config.preprocess_md_file_path, "r") as json_file:
    #     config.preprocess_md = json.load(json_file)
    if legend_outside:
        # Place legend outside the plot at center-right position
        plt.legend(
            bbox_to_anchor=(1.05, 0.5),  # Center-right position
            loc="center left",  # Align the left edge of the legend to the anchor
            borderaxespad=0.0,
        )
    else:
        handles, labels = plt.gca().get_legend_handles_labels()
        if labels:
            plt.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(
        config.plot_path + "/" + name, bbox_inches="tight"
    )  # Ensure all elements are within the saved figure
    plt.close()


def test_hist(config, metrics):
    plt.figure(figsize=config.giffer_fig_size)
    for key in metrics.keys():
        # make sure we have a test hist and avoid sharp train data
        if (
            key.startswith("h_")
            and not key.endswith("_test")
            and not key.endswith("_sharp")
        ):
            hist(
                config,
                metrics,
                key,
                batch_i=metrics["best_test_batch"][-1],
                add_kde=True,
            )
    fig_finalize(config, "test_hist_best_batch.pdf", legend_outside=True)


def batch_plot(config, metrics, key, label):
    plt.plot(metrics[key])
    plt.xlabel("Batch")


def up_development(config, metrics):
    for key in metrics.keys():
        # make sure we have a test hist and avoid sharp evaluated train data
        if (
            key.startswith("h_")
            and not key.endswith("_test")
            and not key.endswith("_sharp")
            and "NOSYS" in key
        ):
            # NOSYS
            NOSYS = metrics[key][:]
            base_key = key.replace("NOSYS", "")
            for key_ in metrics.keys():
                if base_key in key_ and "1UP" in key_ and not "_test" in key_:
                    err_1UP = metrics[key_][:]
                    ratio = err_1UP / NOSYS
                    for i in range(NOSYS.shape[1]):
                        plt.plot(ratio[:, i], label=f"Bin {i+1}")

                    plt.ylabel(f'({key_.replace("_"," ")}) / ({key.replace("_"," ")})')
                    plt.xlabel("Batch")
                    fig_finalize(config, f"{key_}_relative.pdf")


def plots(config):
    with h5py.File(config.metrics_file_path, "r") as metrics:
        test_hist(config, metrics)
        loss(config, metrics)
        if "cls" in config.objective:
            bw(config, metrics)
            up_development(config, metrics)
            if config.include_bins:
                bins(metrics, config)
            cuts(config, metrics)
            sharp_hist_deviation(config, metrics)
