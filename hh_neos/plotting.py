import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pyhf
import relaxed
import seaborn as sns

import hh_neos.histograms
import hh_neos.workspace


def plot_metrics(metrics, config):
    epoch_grid = range(1, config.num_steps + 1)
    print(epoch_grid)
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
    for c, (l, a) in zip(
        ["C0", "C1", "C2", "C3"], zip(yields, jnp.array(list(yields.values())))
    ):
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
                    edgecolor=c,
                    linewidth=2,
                )
            else:
                plt.stairs(
                    a[1:-1],
                    bins[1:-1],
                    label=l,
                    alpha=0.4,
                    fill=None,
                    edgecolor=c,
                    linewidth=2,
                )
            plt.xlabel("m$_{hh}$ (MeV)")
        else:
            plt.bar(
                range(len(a)),
                a,
                label=l,
                alpha=0.4,
                fill=None,
                edgecolor=c,
                linewidth=2,
            )
            plt.xlabel("NN score")
    plt.ylabel("Events")
    plt.legend()
    plt.tight_layout()
    print(config.results_path + "hist.pdf")
    plt.savefig(config.results_path + "hist.pdf")


def get_hist(config, nn, best_params, data, test):
    if config.include_bins:
        bins = jnp.array([0, *best_params["bins"], 1])
        print(best_params["bins"])
    if config.do_m_hh:
        # use whole data set to get correct norm
        yields = hh_neos.histograms.hists_from_mhh(
            data={k: v for k, v in zip(config.data_types, data)},
            bandwidth=1e-8,
            bins=bins,
        )
        model = hh_neos.workspace.model_from_hists(yields)

        # this here gives the same cls! jay
        print(model.expected_data([0, 1.0]))

        CLs_obs, CLs_exp = pyhf.infer.hypotest(
            1.0,  # null hypothesis
            model.expected_data([0, 0.0]),
            model,
            test_stat="q",
            return_expected_set=True,
        )
        print(f"      Observed CLs: {CLs_obs:.4f}")
        for expected_value, n_sigma in zip(CLs_exp, np.arange(-2, 3)):
            print(f"Expected CLs({n_sigma:2d} σ): {expected_value:.4f}")

        print(
            "relaxed cls: ",
            relaxed.infer.hypotest(
                test_poi=1.0,
                data=model.expected_data([0, 0.0]),
                model=model,
                test_stat="q",
                # expected_pars=hypothesis_pars,
                lr=0.002,
            ),
        )

    else:
        # original Asimov Significance:  2.5999689964036663
        # to get correct yields would also need to pass whole data
        yields = hh_neos.histograms.hists_from_nn(
            pars=best_params["nn_pars"],
            data={k: v + 1e-8 for k, v in zip(config.data_types, test)},
            nn=nn,
            bandwidth=1e-8,
            bins=jnp.array([0, *best_params["bins"], 1]),
        )
    print(
        bins[1:-1],
    )
    print(yields)
    print(
        "Asimov Significance: ",
        relaxed.metrics.asimov_sig(s=yields["sig"], b=yields["bkg_nominal"]),
    )

    return bins, yields
