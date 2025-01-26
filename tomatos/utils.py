import logging
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import pyhf
import gc
import sys
import tomatos.histograms
import tomatos.training
import tomatos.workspace
import psutil
from collections import namedtuple

import h5py
import numpy as np
import json
import os

import pprint


def setup_logger(config):
    logging.basicConfig(
        filename=config.results_path + "log.txt",
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.getLogger("pyhf").setLevel(logging.WARNING)
    logging.getLogger("relaxed").setLevel(logging.WARNING)


def inverse_min_max_scale(config, arr, var_idx):
    unscaled_arr = (arr - config.scaler_min[var_idx]) / config.scaler_scale[var_idx]
    return unscaled_arr


def to_jax_static(value, top_level=True):
    """Convert values to JAX-compatible static types using namedtuples."""
    if isinstance(value, list):
        return tuple(to_jax_static(v, top_level=False) for v in value)
    elif isinstance(value, dict):
        NamedTuple = namedtuple("NamedTuple", sorted(value.keys()))
        return NamedTuple(
            **{k: to_jax_static(v, top_level=False) for k, v in value.items()}
        )
    elif isinstance(value, np.ndarray):
        return tuple(value.tolist())
    elif isinstance(value, str) and top_level:
        StringNamedTuple = namedtuple("StringNamedTuple", [f"{value}"])
        return value
    elif isinstance(value, (int, str, float, bool)):
        return value
    return value


def make_opt_config(config):
    """
    Extracts selected attributes from a Setup instance and converts them to
    JAX-compatible static types.

    Args:
        config (Setup): An instance of the Setup class.

    Returns:
        namedtuple: A namedtuple with attributes converted to static
        JAX-compatible types.
    """
    opt_attributes = [
        "weight_idx",
        "objective",
        "slope",
        "opt_cuts",
        "vars",
        "nn_inputs_idx_end",
        "sample_sys",
        "sample_sys_dict",
        "regions_to_sel",
        "include_bins",
        "bins",
        "cls_var_idx",
        "samples",
        "fit_region",
        "nominal",
        "signal_sample",
        "debug",
        "scaler_min",
        "scaler_scale",
        "nn_arch",
    ]

    static_data = {}

    for attr in opt_attributes:
        value = getattr(config, attr, None)
        static_data[attr] = to_jax_static(value)

    # Create a namedtuple with extracted static data
    StaticConfig = namedtuple("opt_config", opt_attributes)
    return StaticConfig(**static_data)


def flatten_dict(nested_dict):
    # e.g.
    # hists[sel][sample][sys] -->
    # hists[sel_sample_sys]
    flat_dict = {}

    def recurse(d, parent_key=""):
        for key, value in d.items():
            new_key = f"{parent_key}_{key}" if parent_key else key
            if isinstance(value, dict):
                recurse(value, new_key)
            elif isinstance(value, jnp.ndarray):
                flat_dict[new_key] = value

    recurse(nested_dict)
    return flat_dict


def filter_hists(config, hists):
    hists = flatten_dict(hists)

    filtered_hists = {}
    for h_key, h in hists.items():
        if any([filter_key in h_key for filter_key in config.plot_hists_filter]):
            h_key = h_key.replace(f"{config.fit_region}_", "")
            filtered_hists[h_key] = h

    return filtered_hists


def init_metrics(
    config,
    state,
):
    # init metrics and metrics.h5 for the 1d and 2d cases
    metrics = {}
    metrics = {k: -1.0 for k in ["train_loss", "valid_loss", "test_loss", "bw"]}
    metrics["bins"] = []

    for var, cut_dict in config.opt_cuts.items():
        var_cut = f"{var}_cut"
        metrics[var_cut] = -1.0
    hists = state.aux

    for k in hists.keys():
        metrics[k] = []
        metrics[k + "_test"] = []  # necessary?
        metrics["kde_" + k] = []
    print(hists)
    with h5py.File(config.metrics_file_path, "w") as h5f:
        for key, value in metrics.items():
            if isinstance(value, float):
                h5f.create_dataset(
                    key,
                    (0,),
                    maxshape=(None,),
                    dtype="f4",
                    compression="gzip",
                )
            elif isinstance(value, list):
                h5f.create_dataset(
                    key,
                    (0, len(value)),
                    maxshape=(None, len(value)),
                    dtype="f4",
                    compression="gzip",
                )
    return metrics


import h5py


def write_metrics(config, metrics, init):

    metrics = to_python_lists(metrics)

    if init:
        # Initialize HDF5 file and create datasets
        with h5py.File(config.metrics_file_path, "w") as h5f:
            for key, value in metrics.items():
                if isinstance(value, float):
                    shape = (0,)
                    maxshape = (None,)
                elif isinstance(value, list):
                    shape = (0, len(value))
                    maxshape = (None, len(value))
                h5f.create_dataset(
                    key, shape=shape, maxshape=maxshape, dtype="f4", compression="gzip"
                )

                h5f[key].resize((1,) + h5f[key].shape[1:])
                h5f[key][0] = value

    else:
        # Append new data to existing HDF5 file
        with h5py.File(config.metrics_file_path, "r+") as h5f:
            for key, value in metrics.items():
                dataset = h5f[key]
                if isinstance(value, float):
                    dataset.resize((dataset.shape[0] + 1,))
                    dataset[-1] = value
                else:
                    dataset.resize((dataset.shape[0] + 1, dataset.shape[1]))
                    dataset[-1, :] = value


def clear_caches():
    # clear caches each update otherwise memory explodes
    # https://github.com/google/jax/issues/10828
    process = psutil.Process()
    if process.memory_info().vms > 4 * 2**30:  # >4GB memory usage
        for module_name, module in sys.modules.items():
            if module_name.startswith("jax"):
                for obj_name in dir(module):
                    obj = getattr(module, obj_name)
                    if hasattr(obj, "cache_clear"):
                        obj.cache_clear()
    gc.collect()


def print_cls(config, yields):
    model = tomatos.workspace.model_from_hists(config, yields)

    CLs_obs, CLs_exp = pyhf.infer.hypotest(
        1.0,  # null hypothesis
        model.expected_data([0, 0.0]),
        model,
        test_stat="q",
        return_expected_set=True,
    )
    logging.info(f"      Observed CLs: {CLs_obs:.6f}")
    for expected_value, n_sigma in zip(CLs_exp, np.arange(-2, 3)):
        logging.info(f"Expected CLs({n_sigma:2d} σ): {expected_value:.6f}")


def to_python_lists(obj):
    """converts (also nested) nd.array or jax.array into a list living in dicts

    Parameters
    ----------
    obj : dict
        input dict

    Returns
    -------
    dict
        output dict
    """
    if isinstance(obj, (np.ndarray, jnp.ndarray)):
        # Convert arrays to Python lists
        return obj.tolist()
    elif isinstance(obj, dict):
        # Recursively process each dictionary value
        return {k: to_python_lists(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        # Recursively process each list element
        return [to_python_lists(x) for x in obj]
    else:
        # Return other objects as is
        return obj


# this is for conservative NN training tests


def binary_cross_entropy(preds, labels):
    epsilon = 1e-15  # To avoid log(0)
    preds = jnp.clip(preds, epsilon, 1 - epsilon)
    return -jnp.mean(labels * jnp.log(preds) + (1 - labels) * jnp.log(1 - preds))


def bce(pars, data, config):
    # broken
    # only need sig and bkg
    values = {k: data[k][:, 0, :] for k in ["NOSYS", "bkg"]}

    # apply the neural network to each data sample, and keep track of the
    # sample names in a dict
    nn_apply = partial(nn, pars)
    preds = {k: jax.vmap(nn_apply)(values[k]).ravel() for k in values}

    sig = preds["NOSYS"]
    bkg = preds["bkg"]
    labels = jnp.concatenate([jnp.ones_like(sig), jnp.zeros_like(bkg)])
    preds = jnp.concatenate((sig, bkg))

    return binary_cross_entropy(preds, labels)


def is_inverted(hist):
    # Set inverted based on which half has the greater sum

    # Calculate the midpoint of the histogram
    midpoint = len(hist) // 2

    # Split the histogram into lower and upper halves
    lower_half = hist[:midpoint]
    upper_half = hist[midpoint:]

    return 1 if lower_half.sum() > upper_half.sum() else 0
