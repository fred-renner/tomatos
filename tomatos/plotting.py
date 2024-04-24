import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pyhf
import relaxed

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


def plot_metrics(metrics, config):
    epoch_grid = range(1, config["num_steps"] + 1)

    # lets account for possible nan's

    for k, v in metrics.items():
        if "cls" in k and len(v) != 0:
            metrics[k] = interpolate_gaps(np.array(v))

            plt.figure()
            plt.plot(
                epoch_grid,
                metrics["cls_train"] / np.max(metrics["cls_train"]),
                label=r"$CL_s$ train",
            )
            plt.plot(
                epoch_grid,
                metrics["cls_valid"] / np.max(metrics["cls_valid"]),
                label=r"$CL_s$ valid",
            )
            plt.plot(
                epoch_grid,
                metrics["cls_test"] / np.max(metrics["cls_test"]),
                label=r"$CL_s$ test",
            )

            plt.legend()
            plt.xlabel("epoch")
            plt.ylabel("normalized loss")
            # ax = plt.gca()
            # ax.set_yscale('log')
            plt.tight_layout()
            plot_path = config["results_path"] + "cls.pdf"

            logging.info(plot_path)
            plt.savefig(plot_path)
            plt.close()
            break

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
    if len(metrics["bins"]) > 0:
        plt.figure()
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
        logging.info(plot_path)
        plt.savefig(plot_path)
        plt.close()


def hist(config, bins, yields):
    fig = plt.figure()
    for l, a in zip(yields, jnp.array(list(yields.values()))):
        if "JET" in l or "GEN" in l:
            break
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
        # this makes sig and bkg only
        # if l == "NOSYS":
        #     break
    fig.legend(loc="upper right")
    plt.ylabel("Events")
    # plt.legend()  # prop={"size": 6})
    plt.tight_layout()
    logging.info(config["results_path"] + "hist.pdf")
    plt.savefig(config["results_path"] + "hist.pdf")
