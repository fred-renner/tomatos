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

# import psutil
import sys
import gc


# clear caches each update otherwise memory explodes
# https://github.com/google/jax/issues/10828
# doubles computation time, still filling up memory but much slower
def clear_caches():
    # process = psutil.Process()
    # if process.memory_info().vms > 4 * 2**30:  # >4GB memory usage
    for module_name, module in sys.modules.items():
        if module_name.startswith("jax"):
            for obj_name in dir(module):
                obj = getattr(module, obj_name)
                if hasattr(obj, "cache_clear"):
                    obj.cache_clear()
    gc.collect()


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

    # adagrad is better for bad gradients
    solver = OptaxSolver(loss, opt=optax.adam(config.lr), jit=True)

    pyhf.set_backend("jax", default=True, precision="64b")

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

        data_dct = {k: v for k, v in zip(config.data_types, data)}
        if config.do_m_hh:
            yields = hh_neos.histograms.hists_from_mhh(
                data=data_dct,
                bandwidth=1e-8,
                bins=params["bins"] if config.include_bins else config.bins,
            )
        else:
            yields = hh_neos.histograms.hists_from_nn(
                pars=params["nn_pars"],
                nn=nn,
                data=data_dct,
                bandwidth=config.bandwidth,  # for the bKDEs
                bins=jnp.array([0, *params["bins"], 1]),
            )
        this_z_a = relaxed.metrics.asimov_sig(s=yields["NOSYS"], b=yields["bkg"])
        print("Z_A: ", this_z_a)
        print()
        clear_caches()
        z_a.append(this_z_a)

    metrics["Z_A"] = z_a
    metrics["bins"] = bins_per_step
    return best_params, metrics
