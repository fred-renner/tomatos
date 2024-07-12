import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pyhf
import relaxed

import tomatos.histograms
import tomatos.utils
import tomatos.workspace
import logging

w_CR = 0.0036312547281962607

fig_size = (5, 4)


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


def plot_metrics(metrics, config):
    plt.rcParams.update({"font.size": 12})
    plt.rcParams["lines.linewidth"] = 0.5
    epoch_grid = range(1, config["num_steps"] + 1)

    # cls
    if len(metrics["cls_train"]) > 0:
        # lets account for possible nan's
        metrics["cls_train"] = interpolate_gaps(np.array(metrics["cls_train"]))
        metrics["cls_valid"] = interpolate_gaps(np.array(metrics["cls_valid"]))
        metrics["cls_test"] = interpolate_gaps(np.array(metrics["cls_test"]))
        plt.figure(figsize=fig_size)
        plt.plot(
            epoch_grid,
            metrics["cls_train"],  # / np.max(metrics["cls_train"]),
            label=r"$CL_s$ train",
        )
        plt.plot(
            epoch_grid,
            metrics["cls_valid"],  # / np.max(metrics["cls_valid"]),
            label=r"$CL_s$ valid",
        )
        plt.plot(
            epoch_grid,
            metrics["cls_test"],  # / np.max(metrics["cls_test"]),
            label=r"$CL_s$ test",
        )

        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        # ax = plt.gca()
        # ax.set_yscale('log')
        plt.tight_layout()
        plot_path = config["results_path"] + "cls.pdf"
        ax = plt.gca()
        ax.set_yscale("log")
        print(plot_path)
        plt.savefig(plot_path)
        plt.close()

    # bce
    if len(metrics["bce_train"]) > 0:
        plt.figure(figsize=fig_size)
        # scale train test for visual comparison
        # could also do ratio, maybe better
        # scale = metrics["cls_test"][0] / metrics["cls_train"][0]
        plt.plot(epoch_grid, metrics["bce_train"], label=r"bce train")
        plt.plot(epoch_grid, metrics["bce_valid"], label=r"bce valid")
        plt.plot(epoch_grid, metrics["bce_test"], label=r"bce test")
        # plt.plot(epoch_grid, scale * metrics["cls_train"], label=r"$CL_s$ train (scaled)")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        ax = plt.gca()
        ax.set_yscale("log")
        plt.tight_layout()
        plot_path = config["results_path"] + "bce.pdf"
        print(plot_path)
        plt.savefig(plot_path)
        plt.close()

    # Z_A
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

    # bins
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
                # plt.xlim([0, 1])
            plt.vlines(x=bins, ymin=i, ymax=i + 1)
            plt.ylabel("epoch")
        plt.tight_layout()
        plot_path = config["results_path"] + "bins.pdf"
        print(plot_path)
        plt.savefig(plot_path)
        plt.close()
    plt.rcParams["lines.linewidth"] = 1

    # cuts
    if len(np.array(metrics["vbf_cut"])) > 0:
        fig, ax1 = plt.subplots(figsize=fig_size)
        color = "tab:red"
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel(r"$m_{jj}$ (TeV)", color=color)
        ax1.plot(np.array(metrics["vbf_cut"]) * 1e-6, color=color)
        ax1.tick_params(axis="y", labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

        color = "tab:blue"
        ax2.set_ylabel(
            r"$|\Delta\eta(j,j)|$", color=color
        )  # we already handled the x-label with ax1
        ax2.plot(metrics["eta_cut"], color=color)
        ax2.tick_params(axis="y", labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped

        plt.xlabel("Epoch")
        plt.tight_layout()
        plot_path = config["results_path"] + "cuts.pdf"
        print(plot_path)
        plt.savefig(plot_path)
        plt.close()

    # bkg shapesys
    if len(metrics["bkg_shape_sys_up"]) > 0:
        plt.figure(figsize=(10, 4))
        up = np.array(metrics["bkg_shape_sys_up"])
        down = np.array(metrics["bkg_shape_sys_down"])
        bkg = np.array(metrics["bkg"])
        rel_up = up / bkg
        # rel_down=down/bkg
        # only up because symmetrized
        for i in range(len(metrics["bkg_shape_sys_up"][0])):
            if i == 0 or i == 1:
                alpha = 0.5
            else:
                alpha = 1
            plt.plot(rel_up[:, i], label=f"Bin {i+1}", alpha=alpha)

        # plt.axvline(x=185,color="black",label="Epoch 185")
        plt.xlabel("Epoch")
        plt.ylabel("Relative Error (err/nominal)")
        plt.legend()
        
        # ax = plt.gca()
        # ax.set_yscale("log")     
        plt.ylim(1,3)

        plt.tight_layout()
        plot_path = config["results_path"] + "bkg_shape_sys_rel_error.pdf"
        print(plot_path)
        plt.savefig(plot_path)

    if len(metrics["ps_up"]) > 0:
        plt.figure(figsize=(10, 4))
        up = np.array(metrics["xbb_pt_bin_3__1up"])
        down = np.array(metrics["ps_down"])
        bkg = np.array(metrics["NOSYS"])
        rel_up = up / bkg
        # rel_down=down/bkg
        # only up because symmetrized
        for i in range(len(metrics["ps_up"][0])):
            if i == 0 or i == 4:
                alpha = 0.5
            else:
                alpha = 1
            plt.plot(rel_up[:, i], label=f"Bin {i+1}", alpha=alpha)

        # plt.axvline(x=185,color="black",label="Epoch 185")
        plt.xlabel("Epoch")
        plt.ylabel("Relative Error (err/nominal)")
        plt.legend()
        plt.tight_layout()

        plot_path = config["results_path"] + "ps_rel_error.pdf"
        print(plot_path)
        plt.savefig(plot_path)
        
        

def hist(config, bins, yields):
    fig = plt.figure(figsize=fig_size)
    for l, a in zip(yields, jnp.array(list(yields.values()))):
        # if "JET" in l or "GEN" in l:
        #     break

        bkg_regions = [
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
        ]
        if any([l == reg for reg in bkg_regions]):
            continue

        if "JET" in l:
            continue
        l = l.replace("_", " ")

        if "bkg" == l:
            l = "Background Estimate"
            if config["objective"] != "bce":
                a *= w_CR
        if "NOSYS" == l:
            l = r"$\kappa_\mathrm{2V}=0$ signal"

        if "ps" == l:
            l = "Parton Shower 1down"
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
    plt.legend(fontsize=5, ncol=3)

    plt.stairs(
        edges=bins,
        values=yields["bkg"] if config["objective"] == "bce" else np.array(yields["bkg"]) * w_CR,
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
    ax = plt.gca()
    newLim = list(ax.get_ylim())
    newLim[1] = newLim[1] * 1.3
    ax.set_ylim(newLim)
    # plt.legend()  # prop={"size": 6})
    plt.tight_layout()
    print(config["results_path"] + "hist.pdf")
    plt.savefig(config["results_path"] + "hist.pdf")
