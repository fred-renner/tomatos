import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pyhf
import relaxed

import hh_neos.histograms
import hh_neos.workspace
import hh_neos.utils


def plot_metrics(metrics, config):
    epoch_grid = range(1, config.num_steps + 1)
    for k, v in metrics.items():
        # if k != "generalised_variance":
        if k == "cls" or k == "Z_A":
            plt.figure()
            plt.plot(epoch_grid, v, label=k)
            plt.legend()
            plt.xlabel("epoch")
            plt.ylabel(k)
            plt.tight_layout()
            print(config.results_path + k + ".pdf")
            plt.savefig(config.results_path + k + ".pdf")
        if k == "bins":
            plt.figure()
            for i, bins in enumerate(v):
                if config.do_m_hh and config.include_bins:
                    bins = (
                        bins * (config.data_max - config.data_min)
                    ) + config.data_min
                    plt.xlabel("m$_{hh}$ (MeV)")
                else:
                    plt.xlabel("NN score")
                    plt.xlim([0, 1])
                plt.vlines(x=bins, ymin=i, ymax=i + 1)
                plt.ylabel("epoch")
            plt.tight_layout()
            print(config.results_path + k + ".pdf")
            plt.savefig(config.results_path + k + ".pdf")
    # plt.yscale("log")
    # plt.legend()
    # plt.xlabel("epoch")
    # plt.ylabel("metric")
    # plt.tight_layout()
    # plt.savefig(results_path + "metrics.pdf")


def hist(config, bins, yields):
    plt.figure()
    for l, a in zip(yields, jnp.array(list(yields.values()))):
        if config.do_m_hh:
            if config.include_bins:
                bins_unscaled = (
                    bins * (config.data_max - config.data_min)
                ) + config.data_min
                print(bins_unscaled)
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
    plt.ylabel("Events")
    plt.legend()
    plt.tight_layout()
    print(config.results_path + "hist.pdf")
    plt.savefig(config.results_path + "hist.pdf")

    hh_neos.utils.print_cls(config, yields)
