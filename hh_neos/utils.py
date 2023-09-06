import jax.numpy as jnp
import pyhf
import relaxed


Array = jnp.ndarray
import hh_neos.histograms
import numpy as np


def get_hist(config, nn, best_params, data):
    if config.include_bins:
        bins = jnp.array([0, *best_params["bins"], 1])
        print(best_params["bins"])
    else:
        bins = config.bins
    if config.do_m_hh:
        # use whole data set to get correct norm
        yields = hh_neos.histograms.hists_from_mhh(
            data={k: v for k, v in zip(config.data_types, data)},
            bandwidth=1e-8,
            bins=bins,
        )

    else:
        # to get correct yields would also need to pass whole data
        yields = hh_neos.histograms.hists_from_nn(
            pars=best_params["nn_pars"],
            data={k: v + 1e-8 for k, v in zip(config.data_types, data)},
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


def print_cls(yields):
    model = hh_neos.workspace.model_from_hists(yields)

    CLs_obs, CLs_exp = pyhf.infer.hypotest(
        1.0,  # null hypothesis
        model.expected_data([0, 0.0]),
        model,
        test_stat="q",
        return_expected_set=True,
    )
    print(f"      Observed CLs: {CLs_obs:.6f}")
    for expected_value, n_sigma in zip(CLs_exp, np.arange(-2, 3)):
        print(f"Expected CLs({n_sigma:2d} σ): {expected_value:.6f}")
