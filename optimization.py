from jaxopt import OptaxSolver
import optax
from time import perf_counter
import jax.numpy as jnp
import pyhf
import pipeline
import relaxed
from functools import partial

Array = jnp.ndarray
from typing import Callable, Any, Generator, Iterable
import histograms


def run(
    config,
    data,
    test,
    batch_iterator,
    init_pars,
    nn,
) -> tuple[Array, dict[str, list]]:
    loss = partial(
        pipeline.pipeline,
        nn=nn,
        sample_names=config.data_types,
        include_bins=config.include_bins,
        do_m_hh=config.do_m_hh,
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
                    loss=config.objective,
                    bandwidth=config.bandwidth,
                )
            else:
                if "bins" in init_pars:
                    del init_pars["bins"]
                state = solver.init_state(
                    init_pars,
                    bins=config.bins,
                    data=data,
                    loss=config.bjective,
                    bandwidth=config.bandwidth,
                )

        start = perf_counter()
        params, state = solver.update(
            params,
            state,
            bins=config.bins,
            data=data,
            loss=config.objective,
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

        if "bins" in params and config.do_m_hh:
            yields = histograms.hists_from_mhh(
                data={k: v for k, v in zip(config.data_types, data)},
                bandwidth=1e-8,
                bins=params["bins"],
            )
            this_z_a = relaxed.metrics.asimov_sig(
                s=yields["sig"], b=yields["bkg_nominal"]
            )
            print("Z_A: ", this_z_a)
            z_a.append(this_z_a)

    metrics["Z_A"] = z_a
    metrics["bins"] = bins_per_step
    return best_params, metrics
