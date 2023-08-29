from functools import partial
from time import perf_counter

import jax
import jax.numpy as jnp
import optax
import pyhf
import relaxed
from jaxopt import OptaxSolver

import hh_neos.pipeline

Array = jnp.ndarray
import hh_neos.histograms


def run(
    config,
    data,
    test,
    batch_iterator,
    init_pars,
    nn,
) -> tuple[Array, dict[str, list]]:
    loss = partial(
        hh_neos.pipeline.pipeline,
        nn=nn,
        sample_names=config.data_types,
        include_bins=config.include_bins,
        do_m_hh=config.do_m_hh,
        loss=config.objective,
    )
    solver = OptaxSolver(loss, opt=optax.adam(config.lr), jit=True)

    pyhf.set_backend("jax", default=True)

    params = init_pars
    best_params = init_pars
    best_sig = 999
    metrics = {k: [] for k in ["cls", "discovery", "poi_uncert"]}
    z_a = []
    bins_per_step = []

    for i in range(config.num_steps):
        print(f"step {i}: loss={config.objective}")
        data = next(batch_iterator)
        if i == 0:
            if config.include_bins:
                init_pars["bins"] = config.bins[
                    1:-1
                ]  # don't want to float endpoints [will account for kde spill]
                state = solver.init_state(
                    init_pars,
                    data=data,
                    bandwidth=config.bandwidth,
                )
            else:
                if "bins" in init_pars:
                    del init_pars["bins"]
                state = solver.init_state(
                    init_pars,
                    bins=config.bins,
                    data=data,
                    bandwidth=config.bandwidth,
                )

        start = perf_counter()
        params, state = solver.update(
            params,
            state,
            bins=config.bins,
            data=data,
            bandwidth=config.bandwidth,
        )
        end = perf_counter()
        print(f"update took {end-start:.4g}s")
        if "bins" in params:
            print("bin edges: [0 ", *[f"{f:.3g}" for f in params["bins"]], " 1]")
            bins_per_step.append(params["bins"])
        for metric in metrics:
            # small bandwidth to have "spikes"
            test_metric = loss(
                params, bins=config.bins, data=test, loss=metric, bandwidth=1e-8
            )
            print(f"{metric}={test_metric:.4g}")
            metrics[metric].append(test_metric)
        if metrics["discovery"][-1] < best_sig:
            best_params = params
            best_sig = metrics["discovery"][-1]
        # for Z_A

        if "bins" in params:
            data_dct = {k: v for k, v in zip(config.data_types, data)}
            if config.do_m_hh:
                yields = hh_neos.histograms.hists_from_mhh(
                    data=data_dct,
                    bandwidth=1e-8,
                    bins=params["bins"],
                )
            else:
                yields = hh_neos.histograms.hists_from_nn(
                    pars=params["nn_pars"],
                    nn=nn,
                    data=data_dct,
                    bandwidth=config.bandwidth,  # for the bKDEs
                    bins=jnp.array([0, *params["bins"], 1]),
                )
            this_z_a = relaxed.metrics.asimov_sig(
                s=yields["sig"], b=yields["bkg_nominal"]
            )
            print("Z_A: ", this_z_a)
            z_a.append(this_z_a)

    metrics["Z_A"] = z_a
    metrics["bins"] = bins_per_step
    return best_params, metrics


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
