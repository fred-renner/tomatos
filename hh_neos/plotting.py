import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pyhf
import relaxed

import hh_neos.histograms
import hh_neos.utils
import hh_neos.workspace
import logging


def plot_metrics(metrics, config):
    epoch_grid = range(1, config["num_steps"] + 1)

    # cls
    plt.figure()
    # scale train test for visual comparison
    # could also do ratio, maybe better
    # scale = metrics["cls_test"][0] / metrics["cls_train"][0]
    plt.plot(epoch_grid, metrics["cls_train"], label=r"$CL_s$ train")
    plt.plot(epoch_grid, metrics["cls_valid"], label=r"$CL_s$ valid")
    plt.plot(epoch_grid, metrics["cls_test"], label=r"$CL_s$ test")
    # plt.plot(epoch_grid, scale * metrics["cls_train"], label=r"$CL_s$ train (scaled)")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    # ax = plt.gca()
    # ax.set_yscale('log')
    plt.tight_layout()
    plot_path = config["results_path"] + "cls.pdf"
    logging.info(plot_path)
    plt.savefig(plot_path)
    plt.close()

    # bce
    if len(metrics["bce_train"]) > 0:
        plt.figure()
        # scale train test for visual comparison
        # could also do ratio, maybe better
        # scale = metrics["cls_test"][0] / metrics["cls_train"][0]
        plt.plot(epoch_grid, metrics["bce_train"], label=r"bce train")
        plt.plot(epoch_grid, metrics["bce_valid"], label=r"bce valid")
        plt.plot(epoch_grid, metrics["bce_test"], label=r"bce test")
        # plt.plot(epoch_grid, scale * metrics["cls_train"], label=r"$CL_s$ train (scaled)")
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel("Loss")
        # ax = plt.gca()
        # ax.set_yscale('log')
        plt.tight_layout()
        plot_path = config["results_path"] + "bce.pdf"
        logging.info(plot_path)
        plt.savefig(plot_path)
        plt.close()

    # Z_A
    plt.figure()
    plt.plot(epoch_grid, metrics["Z_A"], label=r"$Z_A$")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel(r"$Z_A$")
    plt.tight_layout()
    plot_path = config["results_path"] + "Z_A.pdf"
    logging.info(plot_path)
    plt.savefig(plot_path)
    plt.close()

    # bins
    if len(metrics["bce_train"]) > 0:
        plt.figure()
        for i, bins in enumerate(metrics["bins"]):
            if config["do_m_hh"] and config["include_bins"]:
                bins = (np.array(bins) - config["scaler_min"][0]) / config["scaler_scale"][
                    0
                ]
                plt.xlabel("m$_{hh}$ (MeV)")
            else:
                plt.xlabel("NN score")
                # plt.xlim([0, 1])
            plt.vlines(x=bins, ymin=i, ymax=i + 1)
            plt.ylabel("epoch")
        plt.tight_layout()
        plot_path = config["results_path"] + "bins.pdf"
        logging.info(plot_path)
        plt.savefig(plot_path)
        plt.close()


def hist(config, bins, yields):
    plt.figure()
    for l, a in zip(yields, jnp.array(list(yields.values()))):
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
                alpha=0.4,
                fill=None,
                linewidth=2,
                # align="edge",
            )
            plt.xlabel("NN score")
        if l == "NOSYS":
            break
    plt.ylabel("Events")
    plt.legend()  # prop={"size": 6})
    plt.tight_layout()
    logging.info(config["results_path"] + "hist.pdf")
    plt.savefig(config["results_path"] + "hist.pdf")

