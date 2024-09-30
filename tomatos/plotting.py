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


def plot_hist(config, bins, yields, metrics, fig_size):
    fig = plt.figure(figsize=fig_size)
    for l, a in zip(yields, jnp.array(list(yields.values()))):
        # if "JET" in l or "GEN" in l:
        #     break
        unwanted_hists = [
            "bkg_CR_xbb_1",
            "bkg_CR_xbb_2",
            "bkg_VR_xbb_1",
            "bkg_VR_xbb_2",
            "bkg_stat_up",
            "bkg_stat_down",
            "bkg_stat_up",
            "bkg_stat_down",
            "NOSYS_stat_up",
            "NOSYS_stat_down",
            "ps",
        ]
        if any([l == h for h in unwanted_hists]):
            continue

        if "GEN" in l:
            continue
        if "protect" in l:
            continue

        l = l.replace("_", " ")

        if "bkg" == l:
            l = "Background Estimate"
        if "NOSYS" == l:
            l = r"$\kappa_\mathrm{2V}=0$ signal"

        if config["do_m_hh"]:
            if config["include_bins"]:
                bins_unscaled = (np.array(bins) - config["scaler_min"][0]) / config[
                    "scaler_scale"
                ][0]
                plt.stairs(
                    a,
                    bins_unscaled,
                    label=l,
                    alpha=0.4,
                    fill=None,
                    linewidth=2,
                )
            else:
                plt.stairs(
                    a[1:-1],
                    bins[1:-1],
                    label=l,
                    alpha=0.4,
                    fill=None,
                    linewidth=2,
                )
            plt.xlabel("m$_{hh}$ (MeV)")
        else:
            plt.stairs(
                edges=bins,
                values=a,
                label=l,
                # alpha=0.4,
                fill=None,
                # linewidth=2,
                # align="edge",
            )
            plt.xlabel("NN score")
        # this makes sig and bkg only
        # if l == "NOSYS":
        #     break

    ax = plt.gca()
    newLim = list(ax.get_ylim())
    newLim[1] = newLim[1] * 1.3
    ax.set_ylim(newLim)
    if config["objective"] == "cls":
        plt.plot(
            np.linspace(0, 1, len(metrics["kde_signal"][0])),
            metrics["kde_signal"][metrics["epoch_best"]],
            label="kde signal",
            color="tab:orange",
        )
        plt.plot(
            np.linspace(0, 1, len(metrics["kde_bkg"][0])),
            metrics["kde_bkg"][metrics["epoch_best"]],
            label="kde bkg",
            color="tab:blue",
        )
    plt.legend(fontsize=5, ncol=2, loc="upper center")

    plt.stairs(
        edges=bins,
        values=(yields["bkg"]),
        label="Background Estimate",
        fill=None,
        linewidth=2,
        # align="edge",
        color="tab:blue",
    )
    plt.stairs(
        edges=bins,
        values=yields["NOSYS"],
        label=r"$\kappa_\mathrm{2V}=0$ signal",
        fill=None,
        linewidth=2,
        # align="edge",
        color="tab:orange",
    )

    plt.ylabel("Events")
    # ax = plt.gca()
    # newLim = list(ax.get_ylim())
    # newLim[1] = newLim[1] * 1.3
    # ax.set_ylim(newLim)
    # plt.legend()  # prop={"size": 6})
    plt.tight_layout()
    print(config["results_path"] + "hist.pdf")
    plt.savefig(config["results_path"] + "hist.pdf")


def plot_cls(metrics, config, epoch_grid, fig_size):
    if len(metrics["cls_train"]) > 0:
        metrics["cls_train"] = interpolate_gaps(np.array(metrics["cls_train"]))
        metrics["cls_valid"] = interpolate_gaps(np.array(metrics["cls_valid"]))
        metrics["cls_test"] = interpolate_gaps(np.array(metrics["cls_test"]))

        plt.figure(figsize=fig_size)
        plt.plot(epoch_grid, metrics["cls_train"], label=r"$CL_s$ train")
        plt.plot(epoch_grid, metrics["cls_valid"], label=r"$CL_s$ valid")
        plt.plot(epoch_grid, metrics["cls_test"], label=r"$CL_s$ test")

        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.ylim([0, np.max(metrics["cls_train"]) * 1.3])
        plt.tight_layout()
        plot_path = config["results_path"] + "cls.pdf"
        print(plot_path)
        plt.savefig(plot_path)
        plt.close()


def plot_bw(metrics, config, epoch_grid, fig_size):
    if len(metrics["bw"]) > 0:

        plt.figure(figsize=fig_size)
        plt.plot(epoch_grid, metrics["bw"])

        plt.xlabel("Epoch")
        plt.ylabel("Bandwidth")
        plt.tight_layout()
        plot_path = config["results_path"] + "bandwidth.pdf"
        print(plot_path)
        plt.savefig(plot_path)
        plt.close()


def plot_bce(metrics, config, epoch_grid, fig_size):
    if len(metrics["bce_train"]) > 0:
        plt.figure(figsize=fig_size)
        plt.plot(epoch_grid, metrics["bce_train"], label=r"bce train")
        plt.plot(epoch_grid, metrics["bce_valid"], label=r"bce valid")
        plt.plot(epoch_grid, metrics["bce_test"], label=r"bce test")

        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.tight_layout()
        plot_path = config["results_path"] + "bce.pdf"
        print(plot_path)
        plt.savefig(plot_path)
        plt.close()


def plot_Z_A(metrics, config, epoch_grid, fig_size):
    if len(metrics["Z_A"]) > 0:
        plt.figure(figsize=fig_size)
        plt.plot(epoch_grid, metrics["Z_A"], label="Asimov Significance")

        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel(r"$Z_A$")
        plt.tight_layout()
        plot_path = config["results_path"] + "Z_A.pdf"
        print(plot_path)
        plt.savefig(plot_path)
        plt.close()


def plot_bins(metrics, config, epoch_grid, fig_size):
    if len(metrics["bins"]) > 0:
        plt.figure(figsize=fig_size)
        for i, bins in enumerate(metrics["bins"]):
            if config["do_m_hh"] and config["include_bins"]:
                bins = (np.array(bins) - config["scaler_min"][0]) / config[
                    "scaler_scale"
                ][0]
                plt.xlabel("m$_{hh}$ (MeV)")
            else:
                plt.xlabel("NN score")
            plt.vlines(x=bins, ymin=i, ymax=i + 1)
            plt.ylabel("epoch")
        plt.tight_layout()
        plot_path = config["results_path"] + "bins.pdf"
        print(plot_path)
        plt.savefig(plot_path)
        plt.close()


def plot_cuts(metrics, config, epoch_grid, fig_size):
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
        plot_path = config["results_path"] + "cuts.pdf"
        print(plot_path)
        plt.savefig(plot_path)
        plt.close()


def plot_rel_error(metrics, err_hist, nom_hist, config, epoch_grid, fig_size):
    if len(metrics[err_hist]) > 0:
        plt.figure(figsize=fig_size)
        err = np.array(metrics[err_hist])
        nom = np.array(metrics[nom_hist])
        ratio = err / nom

        for i in range(len(metrics[err_hist][0])):
            plt.plot(ratio[:, i], label=f"Bin {i+1}")

        plt.xlabel("Epoch")
        plt.ylabel("Relative Error (err/nominal)")
        plt.legend()
        plt.ylim([1, 2])
        plt.tight_layout()
        plot_path = config["results_path"] + f"{err_hist}_rel_error.pdf"
        print(plot_path)
        plt.savefig(plot_path)
        plt.close()


def plot_hists(metrics, config, epoch_grid, fig_size):
    for k, m in metrics.items():
        if "kde" in k:
            continue
        m = np.array(m)
        if m.ndim == 2:
            plt.figure(figsize=fig_size)

            for i in range(len(m[0])):
                plt.plot(m[:, i], label=f"Bin {i+1}")

            plt.xlabel("Epoch")
            plt.ylabel(k.replace("_", " "))
            plt.legend()
            plt.tight_layout()
            plot_path = config["results_path"] + f"hists/{k}.pdf"
            print(plot_path)
            plt.savefig(plot_path)
            plt.close()


def plot_signal_approximation(metrics, config, epoch_grid, fig_size):
    if len(metrics["signal_approximation_diff"]) > 0:
        plt.figure(figsize=fig_size)
        diff = np.array(metrics["signal_approximation_diff"])
        for i in range(len(metrics["NOSYS"][0])):
            plt.plot(diff[:, i], label=f"Bin {i+1}", alpha=0.75)

        plt.xlabel("Epoch")
        plt.ylabel("Binned KDE/Nominal")
        plt.ylim([0.75, 2])
        plt.legend()
        plt.tight_layout()
        plot_path = config["results_path"] + "signal_approximation_diff.pdf"
        print(plot_path)
        plt.savefig(plot_path)
        plt.close()


def plot_bkg_approximation(metrics, config, epoch_grid, fig_size):
    if len(metrics["bkg_approximation_diff"]) > 0:
        plt.figure(figsize=fig_size)
        diff = np.array(metrics["bkg_approximation_diff"])

        for i in range(len(metrics["NOSYS"][0])):
            plt.plot(diff[:, i], label=f"Bin {i+1}", alpha=0.75)

        plt.xlabel("Epoch")
        plt.ylabel("Binned KDE/Nominal")
        plt.ylim([0.75, 2])
        plt.legend()
        plt.tight_layout()
        plot_path = config["results_path"] + "bkg_approximation_diff.pdf"
        print(plot_path)
        plt.savefig(plot_path)
        plt.close()


def plot_total_diff(metrics, config, fig_size):
    if len(metrics["signal_approximation_diff"]) > 0:
        n_bins = len(config["bins"]) - 1
        
        plt.figure(figsize=fig_size)

        n_bins = len(metrics["signal_approximation_diff"][0])
        total_sig_diff = np.sum(
            np.abs(np.array(metrics["signal_approximation_diff"]) - 1), axis=1
        )
        total_bkg_diff = np.sum(
            np.abs(np.array(metrics["bkg_approximation_diff"]) - 1), axis=1
        )

        plt.plot(total_sig_diff / n_bins, label=r"$\kappa_\mathrm{2V}=0$ signal")
        plt.plot(total_bkg_diff / n_bins, label="Background Estimate")

        plt.xlabel("Epoch")
        plt.ylabel("Relative Bin Average Binned KDE/Yield")
        plt.ylim([0, 0.2])
        plt.legend()
        plt.tight_layout()
        plot_path = config["results_path"] + "summed_diff.pdf"
        print(plot_path)
        plt.savefig(plot_path)
        plt.close()


# Main function to call the plots
def main(config, bins, yields, metrics):

    if not os.path.isdir(config["results_path"] + "/hists/"):
        os.makedirs(config["results_path"] + "/hists/")

    plt.rcParams.update({"font.size": 16})
    plt.rcParams["lines.linewidth"] = 1
    epoch_grid = range(1, config["num_steps"] + 1)
    fig_size = (6, 5)

    plot_hist(config, bins, yields, metrics, fig_size)
    plot_Z_A(metrics, config, epoch_grid, fig_size)

    plot_rel_error(metrics, "xbb_pt_bin_3__1up", "NOSYS", config, epoch_grid, fig_size)
    plot_rel_error(metrics, "gen_up", "NOSYS", config, epoch_grid, fig_size)
    plot_rel_error(metrics, "ps_up", "NOSYS", config, epoch_grid, fig_size)
    plot_rel_error(metrics, "bkg_shape_sys_up", "bkg", config, epoch_grid, fig_size)
    if config["objective"] == "cls":
        plot_hists(metrics, config, epoch_grid, fig_size)
        plot_cls(metrics, config, epoch_grid, fig_size)
        plot_bw(metrics, config, epoch_grid, fig_size)
        # plot_bins(metrics, config, epoch_grid, fig_size)
        plot_cuts(metrics, config, epoch_grid, fig_size)
        plot_signal_approximation(metrics, config, epoch_grid, fig_size)
        plot_bkg_approximation(metrics, config, epoch_grid, fig_size)
        plot_total_diff(metrics, config, fig_size)
    elif config["objective"] == "bce":
        plot_hists(metrics, config, epoch_grid, fig_size)
        plot_bce(metrics, config, epoch_grid, fig_size)
